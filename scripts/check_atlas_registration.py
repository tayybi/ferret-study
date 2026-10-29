import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ferret-mpl')
import sys,json,numpy as np,nibabel as nib,ants
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from prepare_data import ROOT,decode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=ROOT/'ferret-study/reference/registration_trial';out.mkdir(exist_ok=True)
r=next(r for r in json.load(open(ROOT/'ferret-study/data/manifest.json'))['animals'] if r['animal_id']=='BCF100')
a,p=decode(ROOT/r['source']);a=np.clip(a/np.percentile(a,99.5),0,1)
fixed=ants.from_numpy(np.transpose(a,(2,1,0)),spacing=(55/366,55/366,1.5))
fixed=ants.resample_image(fixed,(160,160,18),use_voxels=True)
t=nib.load(ROOT/'ferret-study/reference/Template4D_TE040.nii.gz').get_fdata().astype('float32')
m=nib.load(ROOT/'ferret-study/reference/evDTI_SEGMENTATION.nii.gz').get_fdata()>0
m=binary_fill_holes(m)
t=t.transpose(0,2,1)[:,::-1,:].copy();m=m.transpose(0,2,1)[:,::-1,:].copy()
t=np.clip(t/np.percentile(t[m],99),0,1)*m
moving=ants.from_numpy(t,spacing=(.25,.25,.25));mask=ants.from_numpy(m.astype('float32'),spacing=(.25,.25,.25))
moving=ants.resample_image(moving,(.5,.5,.5));mask=ants.resample_image_to_target(mask,moving,interp_type='nearestNeighbor')
x,y,z=np.indices(fixed.shape);roi=(((x-fixed.shape[0]*.5)/(fixed.shape[0]*.3))**2+((y-fixed.shape[1]*.52)/(fixed.shape[1]*.26))**2<1).astype('float32')
fm=ants.from_numpy(roi,spacing=fixed.spacing)
mc=np.array(ants.get_center_of_mass(mask));fc=np.array([fixed.shape[0]*.5*fixed.spacing[0],fixed.shape[1]*.52*fixed.spacing[1],8.5*1.5])
init=ants.create_ants_transform(transform_type='AffineTransform',dimension=3,center=mc,translation=mc-fc)
ants.write_transform(init,str(out/'initial.mat'))
reg=ants.registration(fixed,moving,type_of_transform='Rigid',initial_transform=str(out/'initial.mat'),mask=fm,mask_all_stages=True,aff_metric='mattes',aff_random_sampling_rate=1,aff_iterations=(300,150,50),aff_shrink_factors=(4,2,1),aff_smoothing_sigmas=(2,1,0),outprefix=str(out/'atlas_'),verbose=False)
warpedmask=ants.apply_transforms(fixed,mask,reg['fwdtransforms'],interpolator='nearestNeighbor')
for name,im in [('fixed',fixed),('atlas',reg['warpedmovout']),('mask',warpedmask)]:ants.image_write(im,str(out/(name+'.nii.gz')))
fig,axs=plt.subplots(3,6,figsize=(15,8))
for j,z in enumerate([1,4,7,10,13,16]):
 f=fixed.numpy()[:,:,z].T;w=reg['warpedmovout'].numpy()[:,:,z].T;b=warpedmask.numpy()[:,:,z].T
 axs[0,j].imshow(f,cmap='gray');axs[0,j].contour(b,levels=[.5],colors='red',linewidths=.7);axs[1,j].imshow(w,cmap='gray');axs[2,j].imshow(f,cmap='gray');axs[2,j].imshow(np.ma.masked_where(b<.5,b),cmap='autumn',alpha=.3)
 for ax in axs[:,j]:ax.axis('off')
 axs[0,j].set_title('Baseline '+str(z))
fig.tight_layout();fig.savefig(out/'atlas_mask_check.png',dpi=120)
print('mask voxels',int(warpedmask.numpy().sum()),'volume mm3',warpedmask.numpy().sum()*np.prod(fixed.spacing),flush=True)
