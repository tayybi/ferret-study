from pathlib import Path
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ferret-mpl')
import sys,json,io,base64
import ants,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
ROOT=Path(__file__).resolve().parents[3]
FULL=ROOT/'full-pipeline'
# This script lives under ferret-study/full-pipeline/visual-summary.
STUDY=Path(__file__).resolve().parents[2]
FULL=STUDY/'full-pipeline'
sys.path.insert(0,str(STUDY/'scripts'))
from full_pipeline import scalar_normalize
from study import ChangeModel,adapt,make_episode
import torch

def encoded(image):
 im=Image.fromarray(image) if isinstance(image,np.ndarray) else image.copy()
 im.thumbnail((160,160))
 b=io.BytesIO();im.save(b,format='JPEG',quality=62)
 return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
def gray(a):return np.uint8(np.clip(a,0,1)*255)
def overlay(a,score,mask):
 rgb=np.repeat(a[...,None],3,axis=-1)
 color=colormaps['magma'](np.clip(score,0,1))[...,:3]
 alpha=(np.clip(score,0,1)*.8*mask)[...,None]
 return np.uint8(np.clip(rgb*(1-alpha)+color*alpha,0,1)*255)
def cropper(mask):
 positions=np.argwhere(mask)
 y0,x0=np.maximum(positions.min(0)-9,0);y1,x1=np.minimum(positions.max(0)+10,mask.shape)
 return np.s_[y0:y1,x0:x1]

manifest=json.loads((FULL/'data/manifest.json').read_text())['animals']
summary={x['animal_id']:x for x in json.loads((FULL/'inference_summary.json').read_text())}
rows=[]
for row in sorted(manifest,key=lambda x:int(x['animal_id'][3:])):
 animal=row['animal_id'];folder=FULL/'subjects'/animal
 info=json.loads((folder/'processing.json').read_text())
 base=ants.image_read(str(folder/'baseline.nii.gz')).numpy().transpose(2,1,0)
 mask=ants.image_read(str(folder/'provisional_mask.nii.gz')).numpy().transpose(2,1,0)>.5
 b=scalar_normalize(base,mask)
 r={'animal':animal,'partition':{'train':'Training','validation':'Validation','test':'Development test'}[row['split']],
    'outcome':'map' if summary.get(animal,{}).get('status','').startswith('exploratory') else ('withheld' if animal in summary else 'unmatched'),
    'followup':info['followup']['label'] if info['followup'] else 'No unambiguous follow-up',
    'frames':[],'source':str(folder.relative_to(STUDY))}
 if r['outcome']=='map':
  f=ants.image_read(str(folder/'followup_registered.nii.gz')).numpy().transpose(2,1,0)
  coverage=ants.image_read(str(folder/'followup_coverage.nii.gz')).numpy().transpose(2,1,0)>.5
  common=mask&coverage;b=scalar_normalize(base,common);f=scalar_normalize(f,common)
  score=ants.image_read(str(folder/'change_score.nii.gz')).numpy().transpose(2,1,0)
  selected=[4,8,12]
  c=cropper(np.any(mask[selected],axis=0))
  for z in selected:
   r['frames'].append({'slice':z+1,'baseline':encoded(gray(b[z][c])),
                      'followup':encoded(gray(f[z][c])),
                      'overlay':encoded(overlay(f[z][c],score[z][c],common[z][c]))})
  r['detail']='Change map generated; no expert injury reference available.'
 elif r['outcome']=='withheld':
  c=cropper(np.any(mask[[4,8,12]],axis=0))
  r['baseline']=encoded(gray(b[8][c]))
  r['detail']='Map withheld: alignment correlation '+str(round(info['registration_correlation_after'],3))+' below the 0.65 engineering cutoff.'
 else:
  c=cropper(np.any(mask[[4,8,12]],axis=0));r['baseline']=encoded(gray(b[8][c]));r['detail']='Baseline processed; follow-up identity is unresolved.'
 rows.append(r)

# Recreate the fixed seed-42 development example; no new model fitting or tuning.
with np.load(FULL/'data/baseline_t2.npz') as cache:volume=cache['BCF101']
with np.load(FULL/'data/brain_masks.npz') as cache:masks=cache['BCF101'].astype(bool)
checkpoint=torch.load(STUDY/'runs/atlas-development-seed42/fomaml_best.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(2);model=ChangeModel();model.load_state_dict(checkpoint['model'])
seed=42+900000
fast=adapt(model,volume,np.random.default_rng(seed),4,'cpu')
arrays=make_episode(volume,np.random.default_rng(seed+10000),8,'test',region_masks=masks)
with torch.no_grad():prediction=torch.sigmoid(fast(torch.from_numpy(arrays[0]),torch.from_numpy(arrays[1]))).numpy()[0,0]
base,query,target,region=[a[0,0] for a in arrays]
c=cropper(region.astype(bool))
synthetic={'animal':'BCF101','followup':encoded(gray(query[c])),
           'truth':encoded(gray(target[c])),'prediction':encoded(overlay(query[c],prediction[c],region[c])),
           'note':'Fixed seed-42 synthetic development example; added abnormality mask is known.'}
assets={'animals':rows,'synthetic':synthetic,'metrics':json.loads((FULL/'aggregate_synthetic_results.json').read_text())}
out=Path(__file__).resolve().parent
(out/'assets.json').write_text(json.dumps(assets,separators=(',',':')))
# Side-by-side visual QA, with labels kept in the inline view.
real=next(r for r in rows if r['animal']=='BCF101')['frames'][1]
sheet=Image.new('RGB',(900,600),'white')
from PIL import ImageDraw,ImageFont
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
draw=ImageDraw.Draw(sheet)
for j,(encoded_image,label) in enumerate([(synthetic['followup'],'Simulated follow-up'),(synthetic['truth'],'Known added region'),(synthetic['prediction'],'Model highlight'),(real['baseline'],'Real baseline'),(real['followup'],'Real follow-up'),(real['overlay'],'Unverified change score')]):
 im=Image.open(io.BytesIO(base64.b64decode(encoded_image.split(',')[1]))).convert('RGB');im.thumbnail((270,250));im=im.resize((270,250))
 x=(j%3)*300;y=(j//3)*300;sheet.paste(im,(x+10,y+35));draw.text((x+10,y+8),label,fill='black',font=font)
sheet.save(out/'example-comparison.png')
print('Animals',len(rows),'embedded bytes',(out/'assets.json').stat().st_size)
