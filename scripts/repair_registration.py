"""Re-run flagged longitudinal registration with correlation and physical initialization."""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ferret-mpl')
from full_pipeline import *

def run():
 rows=json.loads((FULL/'pair_inventory.json').read_text())
 for row in rows:
  if not row['qc_flags'] or row.get('registration_retry_candidates'):continue
  out=FULL/'subjects'/row['animal_id'];base=ants.image_read(str(out/'baseline.nii.gz'));follow=ants.image_read(str(out/'followup_original.nii.gz'));mask=ants.image_read(str(out/'provisional_mask.nii.gz'))
  ones=ants.from_numpy(np.ones(follow.shape,dtype=np.float32),origin=follow.origin,spacing=follow.spacing,direction=follow.direction)
  candidates=[]
  for number,metric in enumerate(['GC','meansquares','mattes']):
   reg=ants.registration(normalized(base,mask),normalized(follow),type_of_transform='Rigid',initial_transform='Identity',mask=mask,mask_all_stages=True,
      aff_metric=metric,aff_random_sampling_rate=1,aff_iterations=(300,150,75),aff_shrink_factors=(4,2,1),aff_smoothing_sigmas=(2,1,0),
      outprefix=str(out/f'physical_retry_{metric}_'),random_seed=42,verbose=False)
   registered=ants.apply_transforms(base,follow,reg['fwdtransforms']);coverage=ants.apply_transforms(base,ones,reg['fwdtransforms'],interpolator='nearestNeighbor')
   common=(mask.numpy()>.5)&(coverage.numpy()>.5);overlap=float(common.sum()/max((mask.numpy()>.5).sum(),1));corr=correlation(base.numpy(),registered.numpy(),common)
   candidates.append((corr if overlap>=.85 else corr-2,metric,corr,overlap,reg,registered,coverage))
  score,metric,corr,overlap,reg,registered,coverage=max(candidates,key=lambda x:x[0])
  row['first_registration_attempt']={k:row[k] for k in ['registration_correlation_before','registration_correlation_after','coverage_fraction','registration_qc','qc_flags','transforms']}
  row['registration_retry_candidates']=[{'metric':m,'correlation':c,'coverage':v} for _,m,c,v,*_ in candidates]
  common=(mask.numpy()>.5)&(coverage.numpy()>.5);unregistered=ants.resample_image_to_target(follow,base)
  before=correlation(base.numpy(),unregistered.numpy(),common)
  flags=[]
  if corr<.65:flags.append('low_structural_correlation')
  if corr<before-.03:flags.append('correlation_decreased')
  if overlap<.85:flags.append('incomplete_followup_coverage')
  row.update(registration_correlation_before=before,registration_correlation_after=corr,coverage_fraction=overlap,
      registration_qc='flagged' if flags else 'automated_checks_passed_expert_review_pending',qc_flags=flags,transforms=reg['fwdtransforms'],registration_metric=metric)
  ants.image_write(registered,str(out/'followup_registered.nii.gz'));ants.image_write(coverage,str(out/'followup_coverage.nii.gz'))
  save_qc(row['animal_id'],base,mask,unregistered,registered,coverage,out)
  (out/'processing.json').write_text(json.dumps(row,indent=2)+'\n')
  print(json.dumps({'animal':row['animal_id'],'selected_metric':metric,'correlation':corr,'coverage':overlap,'flags':flags}),flush=True)
 (FULL/'pair_inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':run()
