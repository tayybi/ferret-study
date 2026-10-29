"""Assemble unconfirmed-change review materials from frozen pipeline outputs.

No training, diagnostic labeling, source modification, or threshold optimization.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_data import decode, ROOT

FULL = ROOT / 'ferret-study/full-pipeline'
OUT = FULL / 'candidate-review'
OUT.mkdir(exist_ok=True)


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(a, mask=None):
    values = a[mask] if mask is not None else a[np.isfinite(a)]
    lo, hi = np.percentile(values, [1, 99.5])
    return np.clip((a-lo)/max(hi-lo, 1e-6), 0, 1)


def load(folder, name):
    return nib.load(folder / (name + '.nii.gz')).get_fdata()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records, subjects, hashes = [], [], {}
    summaries = {r['animal_id']: r for r in json.loads((FULL/'inference_summary.json').read_text())}
    for folder in sorted((FULL/'subjects').iterdir()):
        if not folder.is_dir():
            continue
        animal = folder.name
        info = json.loads((folder/'processing.json').read_text())
        baseline_path = ROOT/info['baseline_source']
        assert sha(baseline_path/'2dseq') == info['baseline_sha256']
        raw_b, _ = decode(baseline_path)
        raw_b = normalize(raw_b)
        raw_f = None
        if info['followup']:
            follow_path = ROOT/info['followup']['source']
            assert sha(follow_path/'2dseq') == info['followup']['sha256']
            raw_f, _ = decode(follow_path)
            raw_f = normalize(raw_f)
        b = load(folder, 'baseline')
        mask = load(folder, 'provisional_mask') > .5
        f = load(folder, 'followup_registered') if raw_f is not None else None
        common = mask & (load(folder, 'followup_coverage') > .5) if f is not None else mask
        bn = normalize(b, common)
        fn = normalize(f, common) if f is not None else None
        accepted = summaries.get(animal, {}).get('status', '').startswith('exploratory')
        labels = np.zeros(b.shape, dtype=np.int32)
        components = 0
        if accepted:
            score = load(folder, 'change_score')
            disagreement = load(folder, 'seed_disagreement')
            assert np.isfinite(score).all() and score.min() >= 0 and score.max() <= 1
            # Fixed existing threshold; all 26-connected components retained, including tiny ones.
            labels, components = ndi.label((score > .5) & common, np.ones((3,3,3)))
            edge = common & ~ndi.binary_erosion(common, structure=np.ones((3,3,1)))
            for number, bounds in enumerate(ndi.find_objects(labels), 1):
                component = labels[bounds] == number
                coords = np.argwhere(component) + np.array([s.start for s in bounds])
                values = score[bounds][component]
                records.append(dict(candidate_id=f'{animal}-C{number:03d}', animal_id=animal,
                    status='unconfirmed_model_candidate', split=info['split'],
                    first_slice_1based=int(coords[:,2].min()+1), last_slice_1based=int(coords[:,2].max()+1),
                    slice_count=len(np.unique(coords[:,2])), voxel_count=int(component.sum()),
                    centroid_x_index_0based=round(float(coords[:,0].mean()),2),
                    centroid_y_index_0based=round(float(coords[:,1].mean()),2),
                    centroid_z_index_0based=round(float(coords[:,2].mean()),2),
                    mean_score=round(float(values.mean()),5), max_score=round(float(values.max()),5),
                    mean_seed_sd=round(float(disagreement[bounds][component].mean()),5),
                    fraction_on_inplane_mask_or_coverage_edge=round(float(edge[bounds][component].mean()),5),
                    mean_normalized_followup_minus_baseline=round(float((fn-bn)[bounds][component].mean()),5),
                    reviewer='', review_date='', reviewer_assessment='', visible_in_native_scan='',
                    adjacent_slice_support='', other_sequence_support='', alignment_acceptable='',
                    exposure_verified='', notes=''))
            label_image = nib.Nifti1Image(labels, nib.load(folder/'baseline.nii.gz').affine)
            label_image.set_data_dtype(np.int32)
            nib.save(label_image, OUT/f'{animal}_candidate_ids.nii.gz')
            assert np.array_equal(nib.load(OUT/f'{animal}_candidate_ids.nii.gz').get_fdata(), labels)
            hashes[str((folder/'change_score.nii.gz').relative_to(FULL))] = sha(folder/'change_score.nii.gz')
        pages = []
        for start in range(0, b.shape[2], 6):
            zs = list(range(start, min(start+6, b.shape[2])))
            fig, axes = plt.subplots(len(zs), 5, figsize=(15, 3*len(zs)), squeeze=False)
            titles = ['Native baseline', 'Native follow-up*', 'Aligned baseline', 'Aligned follow-up', 'Candidate outlines']
            for j, title in enumerate(titles):
                axes[0,j].set_title(title, fontsize=12)
            for row, z in enumerate(zs):
                images = [raw_b[z], raw_f[z] if raw_f is not None and z < len(raw_f) else None,
                          bn[:,:,z].T, fn[:,:,z].T if fn is not None else None,
                          fn[:,:,z].T if fn is not None else bn[:,:,z].T]
                for j, a in enumerate(images):
                    ax = axes[row,j]
                    if a is not None:
                        ax.imshow(a, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                    else:
                        ax.text(.5,.5,'No matched follow-up',ha='center',transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                axes[row,0].set_ylabel(f'Stored slice {z+1}', fontsize=11)
                lab = labels[:,:,z].T
                if lab.any():
                    axes[row,4].contour(lab>0, levels=[.5], colors=['#ffad33'], linewidths=.7)
                    counts = [(int((lab == n).sum()), n) for n in np.unique(lab) if n]
                    # Label up to three largest per slice for legibility; all are in CSV/NIfTI.
                    for _, n in sorted(counts, reverse=True)[:3]:
                        y,x = np.argwhere(lab==n).mean(axis=0)
                        axes[row,4].text(x,y,f'C{n:03d}',color='cyan',fontsize=7)
                elif not accepted:
                    axes[row,4].text(.02,.02,'No model interpretation: withheld / unmatched',
                        fontsize=7,color='white',backgroundcolor='black',transform=axes[row,4].transAxes)
            fig.suptitle(f'{animal} | {info["split"]} | UNCONFIRMED candidates — not injury labels\n'
                         '*Native follow-up uses its own slice index, not anatomical correspondence. Display windows normalized per volume.', fontsize=12)
            fig.tight_layout(rect=(0,0,1,.96))
            name=f'{animal}_slices_{zs[0]+1:02d}_{zs[-1]+1:02d}.png'
            fig.savefig(OUT/name, dpi=95)
            plt.close(fig)
            pages.append(name)
        subjects.append(dict(animal=animal, split=info['split'], pages=pages, candidates=components,
            status='unconfirmed_candidates_for_review' if accepted else ('withheld_registration_qc' if raw_f is not None else 'unmatched_baseline'),
            followup=info['followup']['label'] if info['followup'] else None,
            registration_correlation=info.get('registration_correlation_after'),
            baseline_source=info['baseline_source'], followup_source=info['followup']['source'] if info['followup'] else '',
            reviewer='', reviewer_assessment='', notes=''))
        print(animal, components, 'components;', len(pages), 'pages', flush=True)
    assert len(subjects)==20 and sum(s['status']=='unconfirmed_candidates_for_review' for s in subjects)==15
    assert len({r['candidate_id'] for r in records}) == len(records)
    write_csv(OUT/'candidate_review.csv', records, list(records[0]))
    write_csv(OUT/'subject_review.csv', [{k:v for k,v in r.items() if k!='pages'} for r in subjects], [k for k in subjects[0] if k!='pages'])
    (OUT/'manifest.json').write_text(json.dumps(dict(subjects=subjects, candidate_count=len(records),
        confirmed_injuries=None, threshold=.5, threshold_calibrated=False,
        connectivity=26, minimum_component_size=1, source_map_sha256=hashes),indent=2))
    build_html(subjects)
    print('Review package complete:', OUT, flush=True)


def build_html(subjects):
    options=''.join(f'<option value="{i}">{s["animal"]} — {s["status"]}</option>' for i,s in enumerate(subjects))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ferret candidate review</title><style>body{font:16px/1.5 system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#172b38;background:white}a{color:#1459aa}select,button{font:inherit;padding:8px;max-width:100%}img{width:100%;height:auto}nav{display:flex;gap:18px;flex-wrap:wrap}.note{background:#fff3d4;padding:15px}#status{margin:15px 0}label{display:inline-block;margin:10px 18px 10px 0}</style>
<h1>Ferret: candidate changes for independent review</h1>
<p class="note"><strong>No injuries are confirmed by this package.</strong> Orange contours mark model scores above the existing, uncalibrated 0.5 threshold. These may reflect artifacts, normal anatomy or biological changes. Regions without contours have not been cleared as normal. Provisional masks omit some brain tissue.</p>
<nav><a href="screening/index.html">Provisional screening shortlist</a><a href="../review.html">Pipeline report</a><a href="README.md">Review instructions</a><a href="candidate_review.csv">Candidate review form (CSV)</a><a href="subject_review.csv">Whole-animal review form (CSV)</a><a href="manifest.json">Methods and provenance</a></nav>
<p>20 baseline animals · 19 paired follow-ups · 15 scored pairs · 4 withheld pairs · 1 unmatched baseline. All stored slices shown. No additional model training or threshold tuning.</p>
<label>Animal <select id="animal">__OPTIONS__</select></label><label>Slice page <select id="page"></select></label>
<div id="status" aria-live="polite"></div>
<p>Native images are decoded directly from scanner data without brain masking or registration; grayscale windows are adjusted per volume. Native baseline/follow-up slice numbers are not necessarily matching anatomy. Aligned images use the existing 160 × 160 processing grid. Anatomical left/right is unverified.</p>
<p>Review images without outlines first. Cyan IDs label up to three largest regions per slice; every component is retained in the CSV and candidate-ID NIfTI, including tiny components. Candidate counts are not injury counts.</p>
<a id="full" target="_blank" rel="noopener"><img id="sheet" alt="All-slice native and aligned MRI review sheet"></a>
<p id="ids"></p><script id="data" type="application/json">__DATA__</script>
<script>(()=>{const data=JSON.parse(document.getElementById('data').textContent);const animal=document.getElementById('animal'),page=document.getElementById('page');function show(){const s=data[Number(animal.value)],file=s.pages[Number(page.value)];document.getElementById('sheet').src=file;document.getElementById('full').href=file;document.getElementById('status').textContent=s.animal+' | '+s.split+' | '+s.status+' | '+s.candidates+' model components (not lesions)';const ids=document.getElementById('ids');ids.replaceChildren();if(s.status==='unconfirmed_candidates_for_review'){const a=document.createElement('a');a.href=s.animal+'_candidate_ids.nii.gz';a.textContent='Candidate-ID volume for '+s.animal;ids.append(a)}}function choose(){page.replaceChildren();data[Number(animal.value)].pages.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent='Slices '+(i*6+1)+'–'+(i*6+6);page.append(o)});show()}animal.addEventListener('change',choose);page.addEventListener('change',show);choose()})();</script></html>'''
    (OUT/'index.html').write_text(page.replace('__OPTIONS__',options).replace('__DATA__',json.dumps(subjects).replace('</','<\\/')))


if __name__ == '__main__':
    main()
