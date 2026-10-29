"""Provisional engineering triage; never assigns injury diagnoses or reviewer labels."""
from pathlib import Path
from collections import Counter
import csv, json, hashlib, html
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_candidate_review import FULL, load, normalize, write_csv

REVIEW = FULL/'candidate-review'
OUT = REVIEW/'screening'
CONFIG = dict(score_threshold=.5, higher_score_threshold=.6, minimum_voxels=30,
    minimum_slices=2, minimum_interior_fraction=.5, minimum_score_retention=.5,
    minimum_absolute_mean_difference=.1, minimum_registration_correlation=.7,
    shift_minimum_contrast_retention=.5, shift_minimum_coverage=.9,
    artifact_boundary_fraction=.8, boundary_width_inplane_voxels=2,
    shifts_processing_voxels=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])


def classify(r):
    # These are declared screening rules, not validated clinical criteria.
    shift_concern = (r['minimum_shift_coverage'] < CONFIG['shift_minimum_coverage'] or
                     r['minimum_shift_contrast_retention'] < CONFIG['shift_minimum_contrast_retention'])
    if r['boundary_fraction_2px'] >= CONFIG['artifact_boundary_fraction'] and shift_concern:
        return 'likely processing artifact', 'Boundary-dominated with contrast or coverage sensitivity to small shifts; heuristic flag, not a confirmed cause.'
    tests = {
        'sufficient component size': r['voxel_count'] >= CONFIG['minimum_voxels'],
        'present on adjacent slices': r['slice_count'] >= CONFIG['minimum_slices'],
        'interior support': 1-r['boundary_fraction_2px'] >= CONFIG['minimum_interior_fraction'],
        'higher-threshold retention': r['fraction_score_above_0_6'] >= CONFIG['minimum_score_retention'],
        'measurable normalized contrast': abs(r['mean_difference']) >= CONFIG['minimum_absolute_mean_difference'],
        'alignment perturbation stability': r['shift_stable'],
        'global registration check': r['registration_correlation'] >= CONFIG['minimum_registration_correlation'],
    }
    if all(tests.values()):
        return 'possible change', 'Passed all engineering shortlist rules; biological origin and native-image visibility remain unverified.'
    return 'uncertain', 'Not enough support under these rules: '+', '.join(k for k,v in tests.items() if not v)+'. This does not exclude real abnormality.'


def figure(folder, labels, r, bn, fn, common):
    component=labels==r['component_number']
    peak=int(np.argmax(component.sum(axis=(0,1))))
    zs=list(range(max(0,peak-1),min(bn.shape[2],peak+2)))
    coords=np.argwhere(common.any(axis=2))
    xmin,ymin=np.maximum(coords.min(axis=0)-12,0)
    xmax,ymax=np.minimum(coords.max(axis=0)+13,bn.shape[:2])
    fig,axs=plt.subplots(len(zs),4,figsize=(12,3.3*len(zs)),squeeze=False)
    for j,title in enumerate(['Baseline (aligned grid)','Follow-up (aligned grid)','Follow-up + candidate','Signed normalized difference']):
        axs[0,j].set_title(title,fontsize=10)
    for i,z in enumerate(zs):
        a=bn[xmin:xmax,ymin:ymax,z].T
        b=fn[xmin:xmax,ymin:ymax,z].T
        m=component[xmin:xmax,ymin:ymax,z].T
        for j,im in enumerate([a,b,b,b-a]):
            axs[i,j].imshow(im,cmap='gray' if j<3 else 'coolwarm',vmin=0 if j<3 else -.4,vmax=1 if j<3 else .4)
            axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
        if m.any(): axs[i,2].contour(m,levels=[.5],colors=['#ffad33'],linewidths=1)
        axs[i,0].set_ylabel(f'Slice {z+1}')
    fig.suptitle(f'{r["candidate_id"]} | {r["screening_label"]} (automated, provisional)\n'
        'Peak slice and neighbors | No injury diagnosis | Left/right unverified',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(OUT/(r['candidate_id']+'.png'),dpi=115)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    original_forms={name:hashlib.sha256((REVIEW/name).read_bytes()).hexdigest() for name in ['candidate_review.csv','subject_review.csv']}
    manifest=json.loads((REVIEW/'manifest.json').read_text())
    for name,expected in manifest['source_map_sha256'].items():
        assert hashlib.sha256((FULL/name).read_bytes()).hexdigest()==expected, 'Frozen score map changed: '+name
    subjects=manifest['subjects']
    records=[]
    for s in subjects:
        if s['status']!='unconfirmed_candidates_for_review': continue
        animal=s['animal']; folder=FULL/'subjects'/animal
        labels=load(REVIEW, animal+'_candidate_ids').astype(int)
        mask=load(folder,'provisional_mask')>.5
        common=mask&(load(folder,'followup_coverage')>.5)
        interior=ndi.binary_erosion(common,structure=np.ones((3,3,1)),iterations=2)
        bn=normalize(load(folder,'baseline'),common)
        fn=normalize(load(folder,'followup_registered'),common)
        difference=fn-bn
        score=load(folder,'change_score')
        shifted=[]
        for shift in CONFIG['shifts_processing_voxels']:
            shifted.append((ndi.shift(fn,shift,order=0,mode='constant',cval=0)-bn,
                common&ndi.shift(common,shift,order=0,mode='constant',cval=0)))
        animal_records=[]
        for n in range(1,int(labels.max())+1):
            m=labels==n;coords=np.argwhere(m)
            delta=float(difference[m].mean())
            coverages=[float(valid[m].mean()) for _,valid in shifted]
            contrasts=[float(d[m&valid].mean()) if (m&valid).any() else 0. for d,valid in shifted]
            retention=min(np.sign(delta)*d for d in contrasts)/max(abs(delta),1e-8)
            stable=bool(abs(delta)>=CONFIG['minimum_absolute_mean_difference'] and
                min(coverages)>=CONFIG['shift_minimum_coverage'] and retention>=CONFIG['shift_minimum_contrast_retention'])
            r=dict(candidate_id=f'{animal}-C{n:03d}',animal_id=animal,component_number=n,split=s['split'],
                voxel_count=int(m.sum()),slice_count=len(np.unique(coords[:,2])),
                first_slice_1based=int(coords[:,2].min()+1),last_slice_1based=int(coords[:,2].max()+1),
                boundary_fraction_2px=float((~interior)[m].mean()),fraction_score_above_0_6=float((score[m]>.6).mean()),
                mean_difference=delta,minimum_shift_contrast_retention=float(retention),
                minimum_shift_coverage=min(coverages),shift_stable=stable,
                registration_correlation=s['registration_correlation'],
                native_visibility='not assessed',other_sequence_support='not assessed',
                biological_validation='not established',label_source='automated engineering rules v1')
            r['screening_label'],r['reason']=classify(r)
            r['shortlisted']=False
            animal_records.append(r)
        # At most two per category per animal: diversity of review examples, not severity ranking.
        for category in ['possible change','likely processing artifact','uncertain']:
            eligible=sorted((r for r in animal_records if r['screening_label']==category),
                key=lambda r:(-r['voxel_count'],r['candidate_id']))
            for r in eligible[:2]:
                r['shortlisted']=True
                figure(folder,labels,r,bn,fn,common)
        records.extend(animal_records)
        print(animal,dict(Counter(r['screening_label'] for r in animal_records)),flush=True)
    assert len(records)==210 and len({r['candidate_id'] for r in records})==210
    write_csv(OUT/'all_candidate_screening.csv',records,list(records[0]))
    selected=[r for r in records if r['shortlisted']]
    write_csv(OUT/'shortlist.csv',selected,list(records[0]))
    counts=dict(Counter(r['screening_label'] for r in records))
    summary=dict(config=CONFIG,counts=counts,total_candidates=len(records),shortlist_count=len(selected),
        confirmed_injuries='not established',reviewer_fields_preserved=True,
        rule_validation='not validated; declared engineering screening heuristics',
        perturbation_scope='normalized subtraction only; no model or registration rerun',
        shortlist_selection='up to two largest components per label per animal; not severity ranking')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'all_candidate_screening.json').write_text(json.dumps(records,indent=2))
    build_html(selected,counts)
    assert all(hashlib.sha256((REVIEW/k).read_bytes()).hexdigest()==v for k,v in original_forms.items())
    print(json.dumps(summary,indent=2))


def build_html(rows,counts):
    cards=[]
    for r in rows:
        animal=r['animal_id'];first=r['first_slice_1based'];last=r['last_slice_1based']
        page=((first-1)//6)*6+1
        cards.append(f'''<article data-label="{r['screening_label']}"><h2>{r['candidate_id']} — {r['screening_label']}</h2>
<p>{html.escape(r['reason'])}</p><p>Split: {r['split']} · slices {first}–{last} · {r['voxel_count']} processing voxels (not lesion volume)</p>
<p>Within two pixels of mask/coverage edge: {r['boundary_fraction_2px']:.0%} · retained above score 0.6: {r['fraction_score_above_0_6']:.0%} · small-shift check: {'passed' if r['shift_stable'] else 'not passed'}</p>
<a href="{r['candidate_id']}.png"><img loading="lazy" src="{r['candidate_id']}.png" alt="{r['candidate_id']} baseline, follow-up, candidate outline, and difference"></a>
<p><a href="../{animal}_slices_{page:02d}_{page+5:02d}.png">Native-scan sheet at first candidate slice</a> · <a href="../index.html">All slices and other pages</a></p></article>''')
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Provisional Ferret shortlist</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:25px auto;padding:0 18px;color:#182e3e;background:white}img{width:100%;height:auto}article{margin:35px 0;border-top:1px solid #aaa}select{font:inherit;padding:8px;max-width:100%}.note{background:#fff1d0;padding:16px}a{color:#1459aa}[hidden]{display:none}</style>
<h1>Provisional shortlist: changes and artifact concerns</h1>
<p class="note"><strong>Automated research screening, not radiologist review.</strong> None of these labels confirms injury or normality. “Likely processing artifact” is a rule-based concern, not a verified explanation. Native-image visibility and other-sequence support have not been assessed for individual candidates.</p>
<p>__COUNTS__</p><p>This page contains __N__ review examples: up to two largest components per label per animal. All 210 candidates remain in the screening table. Size controls display selection, not injury severity.</p>
<p><a href="all_candidate_screening.csv">All screening labels and evidence</a> · <a href="shortlist.csv">Shortlist CSV</a> · <a href="METHODS.md">Exact rules and limitations</a> · <a href="PUBLICATION_READINESS.md">Study status</a> · <a href="../index.html">Full scan review</a></p>
<label for="filter">Show </label><select id="filter"><option value="all">All screening labels</option><option>possible change</option><option>likely processing artifact</option><option>uncertain</option></select><p id="shown" aria-live="polite"></p>
__CARDS__
<script>(()=>{const f=document.getElementById('filter');function draw(){let n=0;document.querySelectorAll('article[data-label]').forEach(a=>{a.hidden=f.value!=='all'&&a.dataset.label!==f.value;if(!a.hidden)n++});document.getElementById('shown').textContent=n+' review examples shown'}f.addEventListener('change',draw);draw()})();</script></html>'''
    (OUT/'index.html').write_text(text.replace('__COUNTS__',html.escape(' · '.join(f'{v} {k}' for k,v in counts.items()))).replace('__N__',str(len(rows))).replace('__CARDS__','\n'.join(cards)))


if __name__=='__main__': main()
