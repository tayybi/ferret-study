"""Redraw saved maps for readable report layout; does not rerun or tune models."""
from full_pipeline import *
rows=json.loads((FULL/'inference_summary.json').read_text())
for row in rows:
 if not row['status'].startswith('exploratory'):continue
 animal=row['animal_id'];out=FULL/'subjects'/animal
 base=ants.image_read(str(out/'baseline.nii.gz'));follow=ants.image_read(str(out/'followup_registered.nii.gz'))
 mask=ants.image_read(str(out/'provisional_mask.nii.gz')).numpy()>.5
 coverage=ants.image_read(str(out/'followup_coverage.nii.gz')).numpy()>.5;common=mask&coverage
 b=scalar_normalize(base.numpy(),common);f=scalar_normalize(follow.numpy(),common)
 score=ants.image_read(str(out/'change_score.nii.gz')).numpy()
 indices=np.linspace(3,base.shape[2]-4,5).astype(int)
 fig,axs=plt.subplots(4,5,figsize=(13,10))
 for j,z in enumerate(indices):
  a=b[:,:,z].T;c=f[:,:,z].T;m=common[:,:,z].T;s=score[:,:,z].T
  axs[0,j].imshow(a*m,cmap='gray',vmin=0,vmax=1);axs[0,j].set_title(f'{animal}: slice {z+1}')
  axs[1,j].imshow(c*m,cmap='gray',vmin=0,vmax=1)
  axs[2,j].imshow((c-a)*m,cmap='coolwarm',vmin=-.3,vmax=.3)
  axs[3,j].imshow(a*m,cmap='gray',vmin=0,vmax=1);axs[3,j].imshow(np.ma.masked_where(~m,s),cmap='magma',vmin=0,vmax=1,alpha=.65)
  for ax in axs[:,j]:ax.axis('off')
 fig.suptitle('Baseline / aligned follow-up / subtraction / synthetic-trained score\nExploratory output — not validated injury detection')
 fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'change_maps.png',dpi=110);plt.close(fig)
write_report(['atlas-development-seed42','atlas-development-seed43','atlas-development-seed44'],rows)
print('Redrew 15 saved map sets; no training or inference rerun.')
