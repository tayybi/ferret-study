"""Trace one actual training episode without modifying data or training any model."""
from pathlib import Path
import sys,json,io,base64
import numpy as np
from scipy.ndimage import shift,gaussian_filter
from PIL import Image,ImageDraw,ImageFont
STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY/'scripts'))
from study import make_episode
manifest=json.loads((STUDY/'full-pipeline/data/manifest.json').read_text())['animals']
train=sorted(r['animal_id'] for r in manifest if r['split']=='train')
animal=train[int(np.random.default_rng(42).integers(len(train)))];row=next(r for r in manifest if r['animal_id']==animal)
with np.load(STUDY/'full-pipeline/data/baseline_t2.npz') as cache:volume=cache[animal]
with np.load(STUDY/'full-pipeline/data/brain_masks.npz') as cache:masks=cache[animal].astype(bool)
rng=np.random.default_rng(42+101);indices=rng.integers(len(volume),size=4)
images=volume[indices].copy();h,w=images.shape[-2:];yy,xx=np.mgrid[:h,:w]
followup=[];targets=[];details=[]
for i,image in enumerate(images):
 region=masks[indices[i]];dx,dy=rng.uniform(-.7,.7,size=2)
 query=shift(image,(dy,dx),order=1,mode='nearest');gain=rng.uniform(.85,1.15);bias=rng.uniform(-.04,.04)
 field=1+rng.uniform(-.12,.12)*(xx/w-.5)+rng.uniform(-.12,.12)*(yy/h-.5)
 query=query*gain*field+bias+rng.normal(0,rng.uniform(.005,.025),size=image.shape)
 nuisance=query.copy();target=np.zeros_like(image);signed_amplitude=0
 if i%2==0:
  positions=np.argwhere(region);cy,cx=positions[rng.integers(len(positions))];rx,ry=rng.uniform(.035,.09,size=2)*w
  lesion=((xx-cx)/rx)**2+((yy-cy)/ry)**2<1;target=(lesion&region).astype(np.float32)
  soft=gaussian_filter(target,sigma=.8);signed_amplitude=rng.choice([-1.,1.])*rng.uniform(.18,.40);query+=signed_amplitude*soft
 followup.append(np.clip(query,0,1).astype(np.float32));targets.append(target)
 details.append({'nuisance':np.clip(nuisance,0,1),'amplitude':float(signed_amplitude),'gain':float(gain),'shift_pixels':[float(dx),float(dy)]})
expected=make_episode(volume,np.random.default_rng(42+101),4,region_masks=masks)
assert np.array_equal(expected[0][:,0],images)
assert np.allclose(expected[1][:,0],np.stack(followup),atol=1e-7)
assert np.array_equal(expected[2][:,0],np.stack(targets))
i=0;positions=np.argwhere(masks[indices[i]])
y0,x0=np.maximum(positions.min(0)-6,0);y1,x1=np.minimum(positions.max(0)+7,(h,w));crop=np.s_[y0:y1,x0:x1]
arrays={'baseline':images[i][crop],'variation':details[i]['nuisance'][crop],'simulated':followup[i][crop],'mask':targets[i][crop]}
def uri(a):
 im=Image.fromarray(np.uint8(np.clip(a,0,1)*255));stream=io.BytesIO();im.save(stream,format='PNG')
 return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
metadata={'animal':animal,'training_seed':42,'episode':1,'stored_slice':row['stored_slice_indices_zero_based'][int(indices[i])]+1,
 'amplitude':details[i]['amplitude'],'direction':'brighter' if details[i]['amplitude']>0 else 'darker',
 'trace_matches_training_generator':True,'images':{k:uri(a) for k,a in arrays.items()},
 'scope':'Artificial intensity patch; not a real injury or validated injury simulation.'}
out=Path(__file__).resolve().parent;(out/'simulated-change-assets.json').write_text(json.dumps(metadata,separators=(',',':')))
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
sheet=Image.new('RGB',(1120,330),'white');draw=ImageDraw.Draw(sheet)
for j,(key,title) in enumerate([('baseline','1 Original MRI'),('variation','2 Scan-like variation'),('simulated','3 Added local patch'),('mask','4 Known patch mask')]):
 im=Image.fromarray(np.uint8(np.clip(arrays[key],0,1)*255)).convert('RGB').resize((260,260))
 sheet.paste(im,(j*280+5,50));draw.text((j*280+5,12),title,fill='black',font=font)
sheet.save(out/'simulated-change-steps.png')
print(json.dumps({k:v for k,v in metadata.items() if k!='images'},indent=2))
