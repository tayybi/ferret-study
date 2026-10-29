"""Small paired CNN, first-order MAML, and honest synthetic-change evaluation."""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

STUDY = Path(__file__).resolve().parents[1]
FAST_NAMES = ('scale', 'bias')


class ChangeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.GroupNorm(4,24),nn.SiLU(),
            nn.Conv2d(24,32,3,stride=2,padding=1),nn.GroupNorm(4,32),nn.SiLU())
        self.scale = nn.Parameter(torch.ones(1,32,1,1))
        self.bias = nn.Parameter(torch.zeros(1,32,1,1))
        self.reconstructor = nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,3,padding=1))
        self.detector = nn.Sequential(nn.Conv2d(96,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),
            nn.Conv2d(32,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
        nn.init.constant_(self.detector[-1].bias, -2.)

    def encode(self, x):
        return self.encoder(x) * self.scale + self.bias

    def reconstruct(self, x):
        features = F.interpolate(self.encode(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return torch.sigmoid(self.reconstructor(features))

    def forward(self, baseline, followup):
        a, b = self.encode(baseline), self.encode(followup)
        logits = self.detector(torch.cat([a,b,torch.abs(b-a)], dim=1))
        return F.interpolate(logits, size=baseline.shape[-2:], mode='bilinear', align_corners=False)


def foreground(image):
    # Technical evaluation/placement region, NOT an anatomical brain mask.
    h,w=image.shape
    y,x=np.mgrid[:h,:w]
    ellipse=((x-w*.5)/(w*.29))**2+((y-h*.47)/(h*.29))**2 < 1
    return ellipse & (image>.08)


def make_episode(volume, rng, batch_size=4, family='train', region_masks=None):
    indices=rng.integers(len(volume),size=batch_size)
    baseline=volume[indices].copy()
    followup=[];targets=[];regions=[]
    h,w=baseline.shape[-2:]; yy,xx=np.mgrid[:h,:w]
    for i,image in enumerate(baseline):
        region=foreground(image) if region_masks is None else region_masks[indices[i]].astype(bool)
        # Same nuisance distribution for positive and negative examples.
        dx,dy=rng.uniform(-.7,.7,size=2)
        query=shift(image,(dy,dx),order=1,mode='nearest')
        gain=rng.uniform(.85,1.15);bias=rng.uniform(-.04,.04)
        field=1+rng.uniform(-.12,.12)*(xx/w-.5)+rng.uniform(-.12,.12)*(yy/h-.5)
        query=query*gain*field+bias+rng.normal(0,rng.uniform(.005,.025),size=image.shape)
        target=np.zeros_like(image)
        if i%2==0:
            positions=np.argwhere(region)
            if not len(positions):raise ValueError('Empty synthetic-placement region')
            cy,cx=positions[rng.integers(len(positions))]
            rx,ry=rng.uniform(.035,.09,size=2)*w
            if family=='test':
                # Held-out morphology: two overlapping lobes, rather than a single ellipse.
                lesion=(((xx-cx)/rx)**2+((yy-cy)/ry)**2<1)|(((xx-cx-rx*.8)/(rx*.7))**2+((yy-cy-ry*.4)/(ry*.8))**2<1)
            else:lesion=((xx-cx)/rx)**2+((yy-cy)/ry)**2<1
            target=(lesion & region).astype(np.float32)
            soft=gaussian_filter(target,sigma=.8)
            query+=rng.choice([-1.,1.])*rng.uniform(.18,.40)*soft
        followup.append(np.clip(query,0,1).astype(np.float32));targets.append(target);regions.append(region.astype(np.float32))
    return tuple(np.stack(x)[:,None].astype(np.float32) for x in [baseline,followup,targets,regions])


def tensors(arrays, device):
    return tuple(torch.from_numpy(a).to(device) for a in arrays)


def support(volume, rng, batch_size, device):
    a=volume[rng.integers(len(volume),size=batch_size)][:,None].copy()
    clean=torch.from_numpy(a).to(device)
    # All randomness from a recorded NumPy generator, also reproducible on MPS.
    keep=torch.from_numpy((rng.random(a.shape)>.12).astype(np.float32)).to(device)
    noise=torch.from_numpy(rng.normal(0,.015,size=a.shape).astype(np.float32)).to(device)
    return clean,(clean*keep+noise).clamp(0,1)


def adapt(model, volume, rng, batch_size, device, steps=2, lr=.1):
    fast=copy.deepcopy(model)
    for _ in range(steps):
        clean,corrupt=support(volume,rng,batch_size,device)
        loss=F.mse_loss(fast.reconstruct(corrupt),clean)
        grads=torch.autograd.grad(loss,[getattr(fast,name) for name in FAST_NAMES])
        with torch.no_grad():
            for name,grad in zip(FAST_NAMES,grads):getattr(fast,name).add_(grad,alpha=-lr)
    return fast


def objective(model, arrays, device, reconstruction=None):
    baseline,query,target,region=tensors(arrays,device)
    logits=model(baseline,query)
    # Loss restricted to a fixed technical foreground region, not a validated brain mask.
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=torch.tensor(8.,device=device))*region).sum()/region.sum().clamp_min(1)
    probability=torch.sigmoid(logits)*region
    pos=target.flatten(1).sum(1)>0
    dice=1-(2*(probability*target).flatten(1).sum(1)+1)/(probability.flatten(1).sum(1)+target.flatten(1).sum(1)+1)
    loss=bce+dice[pos].mean()
    if reconstruction is not None:
        clean,corrupt=reconstruction
        loss=loss+.2*F.mse_loss(model.reconstruct(corrupt),clean)
    return loss


def evaluate(model, ids, volumes, device, adaptation, seed, family='validation', details=False, masks=None, inner_steps=2, inner_lr=.1):
    records=[];examples=[]
    for i,animal in enumerate(sorted(ids)):
        # Seeding by ordinal makes all models see identical support and query data.
        rng=np.random.default_rng(seed+i)
        net=adapt(model,volumes[animal],rng,4,device,inner_steps,inner_lr) if adaptation else model
        # Keep query generation independent of whether adaptation consumed RNG draws.
        arrays=make_episode(volumes[animal],np.random.default_rng(seed+10000+i),8,family,
                            region_masks=None if masks is None else masks[animal])
        net.eval()
        with torch.no_grad():
            loss=float(objective(net,arrays,device))
            pred=torch.sigmoid(net(*tensors(arrays[:2],device))).cpu().numpy()[:,0]
        baseline,query,target,region=[a[:,0] for a in arrays]
        truth=target.astype(bool);inside=region.astype(bool);binary=pred>=.5
        positives=truth.reshape(len(truth),-1).any(1)
        dice=[]
        for j in np.flatnonzero(positives):
            p=binary[j]&inside[j];t=truth[j]
            dice.append(float((2*(p&t).sum())/max(p.sum()+t.sum(),1)))
        healthy=inside[~positives];fp=binary[~positives]&healthy
        scores=pred[inside];labels=truth[inside]
        records.append({'animal':animal,'loss':loss,'positive_slice_dice_at_0_5':float(np.mean(dice)),
            'synthetic_pixel_auroc':float(roc_auc_score(labels,scores)),
            'synthetic_pixel_average_precision':float(average_precision_score(labels,scores)),
            'negative_foreground_fraction_flagged_at_0_5':float(fp.sum()/max(healthy.sum(),1))})
        if details:
            examples.append({'animal':animal,'baseline':baseline[0],'query':query[0],'target':target[0],'prediction':pred[0]})
        del net
    metric_names=[k for k in records[0] if k!='animal']
    result={'mean_across_animals':{k:float(np.mean([r[k] for r in records])) for k in metric_names},'per_animal':records}
    return result,examples


def train(method,args,volumes,split,device,out,initial,masks=None):
    model=ChangeModel().to(device);model.load_state_dict(initial)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(args.seed)
    best=float('inf');log=[];start=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        animal=split['train'][int(rng.integers(len(split['train'])))];volume=volumes[animal]
        # Identical query episodes across algorithms at each training step.
        query_rng=np.random.default_rng(args.seed+step*101)
        arrays=make_episode(volume,query_rng,args.batch_size,region_masks=None if masks is None else masks[animal])
        support_rng=np.random.default_rng(args.seed+step*211)
        if method=='fomaml':
            fast=adapt(model,volume,support_rng,args.batch_size,device,args.inner_steps,args.inner_lr)
            loss=objective(fast,arrays,device,support(volume,np.random.default_rng(args.seed+step*307),args.batch_size,device))
            loss.backward()
            # First-order MAML: transfer query gradients, ignoring derivatives through adaptation.
            for original,temporary in zip(model.parameters(),fast.parameters()):
                if temporary.grad is not None:original.grad=temporary.grad.detach().clone()
            del fast
        else:
            loss=objective(model,arrays,device,support(volume,np.random.default_rng(args.seed+step*307),args.batch_size,device))
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==1 or step%args.eval_interval==0 or step==args.steps:
            metrics,_=evaluate(model,split['validation'],volumes,device,method=='fomaml',args.seed+500000,
                               masks=masks,inner_steps=args.inner_steps,inner_lr=args.inner_lr)
            val=metrics['mean_across_animals'];row={'step':step,'training_loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-start,**{'validation_'+k:v for k,v in val.items()}}
            log.append(row)
            (out/(method+'_history.json')).write_text(json.dumps(log,indent=2)+'\n')
            print(json.dumps({'method':method,**row}),flush=True)
            if val['loss']<best:
                best=val['loss']
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save({'model':state,'method':method,'step':step,'validation_loss':best,'config':vars(args)},out/(method+'_best.pt'))
    checkpoint=torch.load(out/(method+'_best.pt'),map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    return model


def make_report(out,results,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method in ['erm','fomaml']:
        rows=json.loads((out/(method+'_history.json')).read_text())
        axes[0].plot([r['step'] for r in rows],[r['training_loss'] for r in rows],label=method)
        axes[1].plot([r['step'] for r in rows],[r['validation_loss'] for r in rows],label=method)
    for ax,title in zip(axes,['Training objective','Validation objective']):
        ax.set(xlabel='Training episode',ylabel='Loss (dimensionless)',title=title);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(examples),4,figsize=(10,2.7*len(examples)),squeeze=False)
    for row,e in zip(axes,examples):
        for ax,key,title in zip(row,['baseline','query','target','prediction'],['Baseline','Synthetic follow-up','Known synthetic mask','Predicted score']):
            ax.imshow(e[key],cmap='gray' if key in ['baseline','query'] else 'magma',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
        row[0].set_ylabel(e['animal'])
    fig.suptitle('Development holdouts — simulated changes only; NOT validated injury maps')
    fig.tight_layout();fig.savefig(out/'synthetic_test_examples.png',dpi=130);plt.close(fig)
    lines=['# Initial technical training run','',
        '**This is a synthetic-change feasibility experiment, not injury validation or evidence of novelty.**','',
        'Scanner baseline labels are provisional. No verified brain masks, longitudinal sham calibration, or real lesion references were used. No real follow-up images were used to train or evaluate this pilot.','',
        '| Method | Synthetic AUROC | Synthetic AP | Positive-slice Dice at 0.5 | Negative foreground fraction flagged at 0.5 |',
        '|---|---:|---:|---:|---:|']
    for name,result in results.items():
        m=result['mean_across_animals'];lines.append(f"| {name} | {m['synthetic_pixel_auroc']:.3f} | {m['synthetic_pixel_average_precision']:.3f} | {m['positive_slice_dice_at_0_5']:.3f} | {m['negative_foreground_fraction_flagged_at_0_5']:.3%} |")
    lines += ['', 'Values are unweighted means across held-out animals. Pixel AUROC/AP are first computed within each animal; they are not independent animal sample sizes. Threshold 0.5 was fixed, not calibrated to a false-alarm target. The foreground region is a heuristic central signal region, not an anatomical brain mask.', '',
        'Test simulated abnormalities use a two-lobe shape; training uses ellipses. Both share a simple synthetic intensity model, so this is only a limited morphology holdout, not proof of transfer to real injury.', '',
        'The ERM and FOMAML models start with identical weights and see identical synthetic training query episodes. Both train the reconstruction head. The additional ERM + adaptation result tests ordinary baseline adaptation without meta-training. Best checkpoints are selected on validation loss only. Final test metrics are not used for checkpoint selection.', '',
        'Required before real study: confirmed animal/health/exposure records, reviewed ferret brain extraction, registration quality control, healthy follow-up controls, independent expert evaluation, and broader validation across animals/seeds.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--inner-steps',type=int,default=2)
    parser.add_argument('--inner-lr',type=float,default=.1)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--eval-interval',type=int,default=25)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--run',default='pilot-001')
    parser.add_argument('--data-dir',default=str(STUDY/'data'))
    parser.add_argument('--mask-cache',default=None)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device='mps' if args.device=='auto' and torch.backends.mps.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in this execution context')
    data_dir=Path(args.data_dir)
    manifest=json.loads((data_dir/'manifest.json').read_text())
    split={name:sorted(r['animal_id'] for r in manifest['animals'] if r['split']==name) for name in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if set(split[a])&set(split[b]):raise ValueError('Animal split leakage')
    with np.load(data_dir/'baseline_t2.npz') as f:volumes={k:f[k] for k in f.files}
    masks=None
    if args.mask_cache:
        with np.load(args.mask_cache) as f:masks={k:f[k].astype(bool) for k in f.files}
        if set(masks)!=set(volumes):raise ValueError('Mask/image animal mismatch')
        if any(masks[k].shape!=volumes[k].shape for k in volumes):raise ValueError('Mask/image shape mismatch')
    out=STUDY/'runs'/args.run
    if out.exists():raise FileExistsError('Choose a new run name; refusing to overwrite results')
    out.mkdir(parents=True)
    initial=copy.deepcopy(ChangeModel().state_dict())
    metadata={'args':vars(args),'device':device,'torch':torch.__version__,'split':split,
        'parameters':sum(x.numel() for x in ChangeModel().parameters()),'scope':'Synthetic changes ONLY; no real injury claims',
        'manifest_sha256':__import__('hashlib').sha256((data_dir/'manifest.json').read_bytes()).hexdigest(),
        'region_type':'Provisional atlas mask, NOT expert ground truth' if masks else 'Heuristic head foreground',
        'evaluation_status':'Development holdouts; not a confirmatory injury test'}
    (out/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    start=time.perf_counter();results={};example=[]
    for method in ['erm','fomaml']:
        model=train(method,args,volumes,split,device,out,initial,masks=masks)
        modes=[False,True] if method=='erm' else [True]
        for adaptation in modes:
            name=method+('_baseline_adaptation' if adaptation else '_no_adaptation')
            results[name],examples=evaluate(model,split['test'],volumes,device,adaptation,args.seed+900000,'test',details=method=='fomaml',
                                            masks=masks,inner_steps=args.inner_steps,inner_lr=args.inner_lr)
            if examples:example=examples
        del model
    metadata['elapsed_seconds']=time.perf_counter()-start
    if device=='mps':
        torch.mps.synchronize();metadata['mps_allocated_bytes']=torch.mps.current_allocated_memory()
    (out/'results.json').write_text(json.dumps({'run':metadata,'synthetic_test_results':results},indent=2)+'\n')
    make_report(out,results,example)
    if masks:
        report=(out/'REPORT.md').read_text()
        report=report.replace('No verified brain masks, longitudinal sham calibration, or real lesion references were used.',
            'Provisional public-atlas masks were used; no expert-verified masks, longitudinal sham calibration, or real lesion references were available.')
        report=report.replace('The foreground region is a heuristic central signal region, not an anatomical brain mask.',
            'The evaluation region is a provisional atlas-derived mask, not expert anatomical ground truth.')
        report=report.replace('Validation controls checkpoint selection; final-test scores do not.',
            'Validation controls checkpoint selection; development-holdout scores do not.')
        (out/'REPORT.md').write_text(report)
    print(json.dumps({'completed':str(out),'device':device,'elapsed_seconds':metadata['elapsed_seconds'],'results':results}),flush=True)


if __name__=='__main__':main()
