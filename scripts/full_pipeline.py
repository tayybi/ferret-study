"""Reproducible provisional longitudinal processing and inference, with explicit QC."""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ferret-mpl')
import argparse
import copy
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, binary_closing, binary_erosion, label
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare_data import ROOT, decode, parameters

STUDY=ROOT/'ferret-study'
FULL=STUDY/'full-pipeline'
SIZE=160


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_normalize(a,mask=None):
    v=a[np.isfinite(a)] if mask is None else a[mask]
    lo,hi=np.percentile(v,[1,99.5])
    return np.clip((a-lo)/max(hi-lo,1e-6),0,1).astype(np.float32)


def scan_image(path):
    a,p=decode(path)
    orientation=np.array([float(x) for x in p['VisuCoreOrientation'].split()]).reshape(-1,3,3)
    if not np.allclose(orientation,np.eye(3)[None],atol=.001):
        raise ValueError('This pipeline requires the inspected identity scanner orientation; conversion needs review')
    positions=np.array([float(x) for x in p['VisuCorePosition'].split()]).reshape(-1,3)
    delta=np.diff(positions,axis=0)
    dz=np.median(delta[:,2])
    if dz<=0 or not np.allclose(delta,[0,0,dz],atol=.01):raise ValueError('Nonuniform slice geometry')
    extent=[float(x) for x in p['VisuCoreExtent'].split()]
    im=ants.from_numpy(a.transpose(2,1,0).copy(),origin=tuple(positions[0]),
        spacing=(extent[0]/a.shape[2],extent[1]/a.shape[1],float(dz)))
    im=ants.resample_image(im,(SIZE,SIZE,a.shape[0]),use_voxels=True,interp_type=0)
    # Internal coordinates preserve scanner physical geometry; anatomical RAS is unverified.
    return im,p


def normalized(im,mask=None):
    a=scalar_normalize(im.numpy(),None if mask is None else mask.numpy()>.5)
    return ants.from_numpy(a,origin=im.origin,spacing=im.spacing,direction=im.direction)


def atlas():
    t=nib.load(STUDY/'reference/Template4D_TE040.nii.gz').get_fdata().astype(np.float32)
    m=nib.load(STUDY/'reference/evDTI_SEGMENTATION.nii.gz').get_fdata()>0
    m=binary_fill_holes(m)
    # Atlas anatomical x/AP/DV -> inspected scanner x/DV/AP; DV reversed for stored row direction.
    t=t.transpose(0,2,1)[:,::-1,:].copy();m=m.transpose(0,2,1)[:,::-1,:].copy()
    t=scalar_normalize(t,m)*m
    im=ants.from_numpy(t,spacing=(.25,.25,.25))
    mask=ants.from_numpy(m.astype(np.float32),spacing=(.25,.25,.25))
    im=ants.resample_image(im,(.5,.5,.5))
    mask=ants.resample_image_to_target(mask,im,interp_type='nearestNeighbor')
    return im,mask


def provisional_mask(fixed,template,template_mask,out):
    x,y,z=np.indices(fixed.shape)
    rough=(((x-fixed.shape[0]*.5)/(fixed.shape[0]*.3))**2+
           ((y-fixed.shape[1]*.52)/(fixed.shape[1]*.26))**2<1)
    fm=ants.from_numpy(rough.astype(np.float32),origin=fixed.origin,spacing=fixed.spacing,direction=fixed.direction)
    mc=np.array(ants.get_center_of_mass(template_mask))
    fc=np.array(fixed.origin)+np.array(fixed.spacing)*np.array([fixed.shape[0]*.5,fixed.shape[1]*.52,(fixed.shape[2]-1)/2])
    init=ants.create_ants_transform('AffineTransform',dimension=3,center=mc,translation=mc-fc)
    ants.write_transform(init,str(out/'atlas_initial.mat'))
    reg=ants.registration(normalized(fixed),template,type_of_transform='Rigid',
        initial_transform=str(out/'atlas_initial.mat'),mask=fm,mask_all_stages=True,
        aff_metric='mattes',aff_random_sampling_rate=1,
        aff_iterations=(300,150,50),aff_shrink_factors=(4,2,1),aff_smoothing_sigmas=(2,1,0),
        outprefix=str(out/'atlas_'),random_seed=42,verbose=False)
    warped=ants.apply_transforms(fixed,template_mask,reg['fwdtransforms'],interpolator='nearestNeighbor')
    a=warped.numpy()>.5
    # Fill internal gaps without labeling this as expert segmentation.
    for z in range(a.shape[2]):a[:,:,z]=binary_fill_holes(binary_closing(a[:,:,z],iterations=1))
    components,n=label(a)
    if n:a=components==np.argmax(np.bincount(components.ravel())[1:])+1
    mask=ants.from_numpy(a.astype(np.float32),origin=fixed.origin,spacing=fixed.spacing,direction=fixed.direction)
    volume=float(a.sum()*np.prod(fixed.spacing))
    if not 2000<volume<14000:raise ValueError(f'Implausible provisional mask volume: {volume:.0f} mm3')
    return mask,volume


def correlation(a,b,mask):
    x=a[mask];y=b[mask]
    if len(x)<100 or x.std()<1e-8 or y.std()<1e-8:return -1.
    return float(np.corrcoef(x,y)[0,1])


def save_qc(animal,base,mask,raw_follow,registered,coverage,out):
    n=base.shape[2];indices=np.linspace(2,n-3,6).astype(int)
    b=normalized(base).numpy();m=mask.numpy()>.5
    rows=2 if registered is None else 4
    fig,axs=plt.subplots(rows,6,figsize=(15,rows*2.6),squeeze=False)
    for j,z in enumerate(indices):
        axs[0,j].imshow(b[:,:,z].T,cmap='gray',vmin=0,vmax=1)
        axs[0,j].contour(m[:,:,z].T,levels=[.5],colors='orange',linewidths=.7)
        axs[0,j].set_title(f'{animal}: slice {z+1}')
        axs[1,j].imshow((b*m)[:,:,z].T,cmap='gray',vmin=0,vmax=1)
        if registered is not None:
            r=normalized(registered).numpy();axs[2,j].imshow(r[:,:,z].T,cmap='gray',vmin=0,vmax=1)
            axs[2,j].contour(m[:,:,z].T,levels=[.5],colors='orange',linewidths=.6)
            # Alternating tiles allow anatomical inspection without judging difference as pathology.
            yy,xx=np.indices(b.shape[:2]);checker=((xx//12+yy//12)%2).astype(bool)
            mixed=np.where(checker,b[:,:,z],r[:,:,z]);axs[3,j].imshow(mixed.T,cmap='gray',vmin=0,vmax=1)
        for ax in axs[:,j]:ax.axis('off')
    title='Provisional atlas mask / extracted baseline'
    if registered is not None:title+=' / registered follow-up / checkerboard'
    fig.suptitle(title+' — expert QC required')
    fig.tight_layout();fig.savefig(out/'processing_qc.png',dpi=110);plt.close(fig)


def resize_stack(a,size=128,mask=False):
    return np.stack([np.asarray(Image.fromarray(frame.astype(np.float32)).resize((size,size),
        Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR)) for frame in a]).astype(np.float32)


def prepare():
    FULL.mkdir(exist_ok=True)
    inventory=json.loads((ROOT/'Ferret/visual-overview/session-inventory.json').read_text())
    old=json.loads((STUDY/'data/manifest.json').read_text())
    template,template_mask=atlas()
    manifest=[];all_pairs=[];images={};masks={};started=time.perf_counter()
    for row in sorted(old['animals'],key=lambda r:r['animal_id']):
        animal=row['animal_id'];out=FULL/'subjects'/animal;out.mkdir(parents=True,exist_ok=True)
        cached=out/'processing.json'
        if cached.exists():
            info=json.loads(cached.read_text());base=ants.image_read(str(out/'baseline.nii.gz'));mask=ants.image_read(str(out/'provisional_mask.nii.gz'))
        else:
            base,p=scan_image(ROOT/row['source']);mask,volume=provisional_mask(base,template,template_mask,out)
            ants.image_write(base,str(out/'baseline.nii.gz'));ants.image_write(mask,str(out/'provisional_mask.nii.gz'))
            info={'animal_id':animal,'split':row['split'],'baseline_source':row['source'],
                  'baseline_sha256':row['source_sha256'],'baseline_date':row['date'],
                  'mask_volume_mm3':volume,'mask_status':'Public-atlas rigid transfer; provisional, not expert ground truth',
                  'coordinate_status':'Scanner physical geometry preserved; anatomical RAS not verified',
                  'followup':None,'status':'baseline_only','registration_qc':'not_applicable'}
            matches=row['candidate_followups']
            if len(matches)==1:
                match=matches[0];session=next(s for s in inventory if s['path']==match['path'])
                scans=[s for s in session['scans'] if s['method']=='<Bruker:RARE>' and s['protocol']=='04_RARE_T2_axial']
                if len(scans)!=1:raise ValueError('Ambiguous follow-up T2: '+animal)
                path=ROOT/'Ferret'/session['path']/scans[0]['number']/'pdata/1'
                follow,p=scan_image(path)
                ants.image_write(follow,str(out/'followup_original.nii.gz'))
                reg=ants.registration(normalized(base,mask),normalized(follow),type_of_transform='Rigid',mask=mask,
                    mask_all_stages=True,aff_metric='mattes',aff_random_sampling_rate=1,
                    aff_iterations=(400,200,100),aff_shrink_factors=(4,2,1),aff_smoothing_sigmas=(2,1,0),
                    outprefix=str(out/'followup_'),random_seed=42,verbose=False)
                registered=ants.apply_transforms(base,follow,reg['fwdtransforms'])
                ones=ants.from_numpy(np.ones(follow.shape,dtype=np.float32),origin=follow.origin,spacing=follow.spacing,direction=follow.direction)
                coverage=ants.apply_transforms(base,ones,reg['fwdtransforms'],interpolator='nearestNeighbor')
                unregistered=ants.resample_image_to_target(follow,base)
                common=(mask.numpy()>.5)&(coverage.numpy()>.5)
                before=correlation(base.numpy(),unregistered.numpy(),common)
                after=correlation(base.numpy(),registered.numpy(),common)
                overlap=float(common.sum()/max((mask.numpy()>.5).sum(),1))
                flags=[]
                if after<.65:flags.append('low_structural_correlation')
                if after<before-.03:flags.append('correlation_decreased')
                if overlap<.85:flags.append('incomplete_followup_coverage')
                info.update(status='paired',followup={'label':match['label'],'date':session['date'],
                    'elapsed_days':(datetime.fromisoformat(session['date'])-datetime.fromisoformat(row['date'])).days,
                    'source':str(path.relative_to(ROOT)),'sha256':digest(path/'2dseq')},
                    registration_correlation_before=before,registration_correlation_after=after,
                    coverage_fraction=overlap,registration_qc='flagged' if flags else 'automated_checks_passed_expert_review_pending',
                    qc_flags=flags,transforms=reg['fwdtransforms'])
                ants.image_write(registered,str(out/'followup_registered.nii.gz'));ants.image_write(coverage,str(out/'followup_coverage.nii.gz'))
                save_qc(animal,base,mask,unregistered,registered,coverage,out)
            else:save_qc(animal,base,mask,None,None,None,out)
            cached.write_text(json.dumps(info,indent=2)+'\n')
        # Provisional masks are used for all baseline training data; source intensities remain unchanged.
        b=base.numpy().transpose(2,1,0);m=(mask.numpy()>.5).transpose(2,1,0)
        selected=np.asarray(row['stored_slice_indices_zero_based'])
        b=scalar_normalize(b,m);small=resize_stack(b);sm=resize_stack(m,mask=True)>.5
        sm=np.stack([binary_erosion(v,iterations=1) for v in sm])
        if min(sm[selected].sum((1,2)))<30:raise ValueError('Insufficient brain-mask area for training '+animal)
        images[animal]=small[selected]*sm[selected];masks[animal]=sm[selected].astype(np.uint8)
        new=copy.deepcopy(row);new.update(processed_shape=list(images[animal].shape),
            source_processing=str(cached.relative_to(ROOT)),mask_status=info['mask_status'],normalization='Per-baseline atlas-mask percentiles')
        manifest.append(new)
        if info['status']=='paired':all_pairs.append(info)
        print(json.dumps({'prepared':animal,'paired':info['status']=='paired','qc':info['registration_qc'],
                          'elapsed_seconds':round(time.perf_counter()-started,1)}),flush=True)
    data=FULL/'data';data.mkdir(exist_ok=True)
    np.savez_compressed(data/'baseline_t2.npz',**images);np.savez_compressed(data/'brain_masks.npz',**masks)
    (data/'manifest.json').write_text(json.dumps({'scope':'Provisional atlas-masked development pipeline; not injury validation',
        'animals':manifest,'image_size':128,'mask_status':'All masks require expert review'},indent=2)+'\n')
    (FULL/'pair_inventory.json').write_text(json.dumps(all_pairs,indent=2)+'\n')
    (FULL/'reference_provenance.json').write_text(json.dumps({'atlas':'Hutchinson et al. 2017 adult ferret atlas',
        'source_page':'https://scalablebrainatlas.incf.org/ferret/HSRetal17',
        'source_paper':'https://doi.org/10.1016/j.neuroimage.2017.03.009',
        'files':{p.name:digest(p) for p in (STUDY/'reference').glob('*.nii.gz')},
        'method':'In-vivo TE40 template with transferred ex-vivo label union; cross-template alignment provisional',
        'review_status':'Not validated against expert masks; no anatomical regional labels analyzed'},indent=2)+'\n')
    print(json.dumps({'preparation_complete':True,'baselines':len(manifest),'paired_followups':len(all_pairs),
                     'flagged_pairs':[r['animal_id'] for r in all_pairs if r['qc_flags']]}),flush=True)


def inference(run_names):
    import torch
    from study import ChangeModel,adapt
    device='mps' if torch.backends.mps.is_available() else 'cpu';torch.set_num_threads(4)
    rows=json.loads((FULL/'pair_inventory.json').read_text());models=[]
    for name in run_names:
        checkpoint=torch.load(STUDY/'runs'/name/'fomaml_best.pt',map_location='cpu',weights_only=False)
        model=ChangeModel().to(device);model.load_state_dict(checkpoint['model']);model.eval()
        models.append((name,model,checkpoint['config']))
    summary=[]
    for i,row in enumerate(rows):
        animal=row['animal_id'];out=FULL/'subjects'/animal
        if row['qc_flags']:
            summary.append({'animal_id':animal,'status':'inference_withheld_registration_qc','flags':';'.join(row['qc_flags'])});continue
        base=ants.image_read(str(out/'baseline.nii.gz'));follow=ants.image_read(str(out/'followup_registered.nii.gz'))
        mask=ants.image_read(str(out/'provisional_mask.nii.gz')).numpy()>.5
        coverage=ants.image_read(str(out/'followup_coverage.nii.gz')).numpy()>.5
        common=mask&coverage
        b=base.numpy();f=follow.numpy()
        # Robust per-visit intensity standardization, estimated within the common provisional mask.
        bn=scalar_normalize(b,common);fn=scalar_normalize(f,common)
        bsmall=resize_stack(bn.transpose(2,1,0));fsmall=resize_stack(fn.transpose(2,1,0));msmall=resize_stack(common.transpose(2,1,0),mask=True)>.5
        bsmall*=msmall;fsmall*=msmall
        scores=[]
        for name,model,config in models:
            fast=adapt(model,bsmall,np.random.default_rng(config['seed']+800000+i),4,device,config['inner_steps'],config['inner_lr'])
            fast.eval();parts=[]
            with torch.no_grad():
                for j in range(0,len(bsmall),4):
                    a=torch.from_numpy(bsmall[j:j+4,None]).to(device);q=torch.from_numpy(fsmall[j:j+4,None]).to(device)
                    parts.append(torch.sigmoid(fast(a,q)).cpu().numpy()[:,0])
            scores.append(np.concatenate(parts)*msmall);del fast
        average=np.mean(scores,axis=0);disagreement=np.std(scores,axis=0)
        mapped=resize_stack(average,SIZE).transpose(2,1,0)*common
        uncertainty=resize_stack(disagreement,SIZE).transpose(2,1,0)*common
        for name,a in [('change_score',mapped),('seed_disagreement',uncertainty),('normalized_subtraction',(fn-bn)*common)]:
            image=ants.from_numpy(a.astype(np.float32),origin=base.origin,spacing=base.spacing,direction=base.direction)
            ants.image_write(image,str(out/(name+'.nii.gz')))
        # Use a predetermined slice selection and shared display windows; no cherry-picking anomalies.
        indices=np.linspace(3,len(bsmall)-4,5).astype(int)
        fig,axs=plt.subplots(4,5,figsize=(13,10))
        for j,z in enumerate(indices):
            axs[0,j].imshow(bsmall[z],cmap='gray',vmin=0,vmax=1);axs[0,j].set_title(f'{animal}: slice {z+1}')
            axs[1,j].imshow(fsmall[z],cmap='gray',vmin=0,vmax=1)
            axs[2,j].imshow((fsmall-bsmall)[z],cmap='coolwarm',vmin=-.3,vmax=.3)
            axs[3,j].imshow(bsmall[z],cmap='gray',vmin=0,vmax=1)
            axs[3,j].imshow(np.ma.masked_where(~msmall[z],average[z]),cmap='magma',vmin=0,vmax=1,alpha=.65)
            for ax in axs[:,j]:ax.axis('off')
        fig.suptitle('Baseline / aligned follow-up / subtraction / synthetic-trained score\nExploratory output — not validated injury detection')
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'change_maps.png',dpi=110);plt.close(fig)
        record={'animal_id':animal,'status':'exploratory_maps_generated_expert_review_pending','split':row['split'],
            'followup_label':row['followup']['label'],'elapsed_days':row['followup']['elapsed_days'],
            'registration_correlation':row['registration_correlation_after'],'coverage_fraction':row['coverage_fraction'],
            'mean_change_score':float(mapped[common].mean()),'fraction_above_uncalibrated_0_5':float((mapped[common]>.5).mean()),
            'mean_seed_disagreement':float(uncertainty[common].mean())}
        summary.append(record);print(json.dumps(record),flush=True)
    (FULL/'inference_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    names=sorted({k for row in summary for k in row})
    with (FULL/'inference_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader();writer.writerows(summary)
    write_report(run_names,summary)


def write_report(run_names,summary):
    lines=['# Ferret end-to-end development pipeline','',
        '**Status: computational development workflow completed. Injury detection and anatomical masks are not clinically or experimentally validated.**','',
        '## Completed stages','',
        '- Decoded scanner T2 data with stored intensity scaling and checked geometry.',
        '- Preserved scanner physical geometry on a 160 × 160 processing grid. Anatomical RAS orientation remains unverified.',
        '- Generated provisional public-atlas brain masks for 20 baseline-labeled animals.',
        '- Matched and rigidly registered 19 labeled follow-ups; ambiguous BCF94/BCR94 naming was not silently resolved.',
        '- Created extraction/registration QC sheets and withheld inference for automated registration failures.',
        '- Trained ordinary and first-order meta-learning models across multiple random seeds, using synthetic changes restricted to eroded provisional masks.',
        '- Selected checkpoints using validation animals; scored synthetic examples on animal-separated development holdouts.',
        '- Adapted each meta-trained model using each animal baseline only and generated exploratory real follow-up score maps.',
        '- Saved transforms, images, masks, score maps, seed disagreement, configuration, source hashes, and summary tables.','',
        '## Synthetic development results','',
        '| Run | Method | Synthetic Dice at 0.5 | Synthetic AP |', '|---|---|---:|---:|']
    for name in run_names:
        r=json.loads((STUDY/'runs'/name/'results.json').read_text())
        for method,result in r['synthetic_test_results'].items():
            m=result['mean_across_animals'];lines.append(f"| {name} | {method} | {m['positive_slice_dice_at_0_5']:.3f} | {m['synthetic_pixel_average_precision']:.3f} |")
    done=[r for r in summary if r['status'].startswith('exploratory')]
    lines+=['','## Real follow-up outputs','',f'Exploratory maps generated for {len(done)} of {len(summary)} matched pairs. Any withheld pairs are listed in inference_summary.csv.',
        '', 'The score is learned from synthetic changes and is not a calibrated probability of injury. The 0.5 score threshold has no false-alarm guarantee. Seed disagreement is model variability, not a validated uncertainty interval. Outputs for training animals are in-sample exploratory inference; development-holdout animals are distinguished in the summary.','',
        'Visual review of selected maps showed boundary-heavy and broad intensity-related scores. These patterns may arise from registration, residual contrast differences, masking, or normal longitudinal changes; they are not evidence of injury. Conservative atlas masks omit parts of the brain, so whole-brain lesion-volume claims are not supported.','',
        '## Review before scientific claims','',
        'All atlas masks require expert anatomical review. Registration metrics are engineering QC and are not proof of alignment accuracy; inspect processing_qc.png. Exposure, sham status, pre-injury timing, stable animal IDs, and real lesion references remain unverified. Normal growth, acquisition differences, masking, and registration errors can produce change scores. No real injury AUROC, lesion Dice, sensitivity, specificity, or conformal control is claimed.','',
        'This pipeline uses rigid longitudinal registration to reduce the risk of deforming away biological changes. Baseline masking and per-visit percentile normalization can still suppress or distort genuine changes and need sensitivity analysis.','',
        'The pilot holdouts have already been inspected during development. They are not a locked confirmatory publication test set. Plan independent evaluation or a properly nested animal-level study before publication.','',
        '## Files','',
        '- subjects/<animal>/processing_qc.png: baseline-mask, follow-up, checkerboard QC.',
        '- subjects/<animal>/change_maps.png: baseline, follow-up, subtraction, score overlay.',
        '- subjects/<animal>/*.nii.gz: intermediate volumes, provisional masks, score maps.',
        '- pair_inventory.json and inference_summary.csv: pairing, processing, QC, inference.',
        '- data/manifest.json: frozen animal partitions and source provenance.',
        '- ../runs/<run>/: checkpoints, learning curves, synthetic examples, metrics.',
        '- reference_provenance.json: adult-ferret atlas attribution and hashes.','',
        'Atlas: Hutchinson et al. (2017), https://doi.org/10.1016/j.neuroimage.2017.03.009; downloaded from https://scalablebrainatlas.incf.org/ferret/HSRetal17.']
    (FULL/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer']);parser.add_argument('--runs',nargs='+',default=[])
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        if not args.runs:raise ValueError('Supply trained run names')
        inference(args.runs)
