"""Build an offline review report from completed outputs; no network or inference."""
from pathlib import Path
import json,html
root=Path(__file__).resolve().parents[1]/'full-pipeline'
pairs=json.loads((root/'pair_inventory.json').read_text())
summary=json.loads((root/'inference_summary.json').read_text())
scores={r['animal_id']:r for r in summary}
animals=sorted(p.name for p in (root/'subjects').iterdir() if p.is_dir())
rows=[]
for animal in animals:
 info=json.loads((root/'subjects'/animal/'processing.json').read_text())
 s=scores.get(animal,{})
 rows.append({'animal':animal,'split':info['split'],'status':s.get('status','No unambiguous follow-up'),
              'flags':'; '.join(info.get('qc_flags',[])) or 'Automatic checks passed; expert review pending',
              'followup':info['followup']['label'] if info['followup'] else 'None confirmed',
              'maps':(root/'subjects'/animal/'change_maps.png').exists()})
data=json.dumps(rows).replace('</','<\\/')
options=''.join(f'<option value="{i}">{r["animal"]} — {html.escape(r["followup"])}</option>' for i,r in enumerate(rows))
fragment='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Ferret development pipeline review</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1150px;margin:25px auto;padding:0 20px;color:#192a3a;background:#fff}h1,h2{line-height:1.25}select{font:inherit;max-width:100%;padding:10px}img{display:block;width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #d2d8de}a{color:#1459aa}.note{padding:14px;background:#fff4db}.status{padding:12px 0}nav{display:flex;gap:20px;flex-wrap:wrap}code{overflow-wrap:anywhere}</style>
<h1>Ferret pipeline: development review</h1>
<p class="note"><strong>Exploratory outputs, not validated injury detection.</strong> Atlas masks need expert correction; study labels and controls remain unverified. Scores are trained on simulated changes. No injury probability, false-alarm guarantee, or whole-brain lesion volume is established.</p>
<p><strong>20 baselines · 19 follow-up pairs · 15 exploratory map sets · 4 pairs withheld</strong></p>
<nav><a href="candidate-review/screening/index.html">Provisional screening shortlist</a><a href="candidate-review/index.html">Highlighted candidates for independent review</a><a href="REPORT.md">Full report</a><a href="inference_summary.csv">Summary CSV</a><a href="pair_inventory.json">Pairing and QC</a><a href="reference_provenance.json">Atlas provenance</a></nav>
<h2>Inspect an animal</h2><label for="animal">Animal and scanner follow-up label</label><br><select id="animal">__OPTIONS__</select>
<div id="status" class="status" aria-live="polite"></div>
<h3>Extraction and registration</h3><p>Rows: baseline with provisional mask; extracted baseline; aligned follow-up; checkerboard. Baseline-only animals have two rows.</p><img id="qc" alt="Selected animal extraction and registration checks">
<div id="map-section"><h3>Exploratory change maps</h3><p>Rows: baseline; aligned follow-up; normalized subtraction; ensemble synthetic-trained change score. Bright scores can reflect artifacts, normal changes, or abnormalities.</p><img id="maps" alt="Selected animal exploratory longitudinal change maps"></div>
<h2>Synthetic development comparisons</h2><p>Animal-separated development holdouts, not real injury accuracy. Three training seeds do not increase the number of independent evaluation animals.</p>
<table><thead><tr><th>Method</th><th>Mean synthetic Dice</th><th>SD across three seeds</th></tr></thead><tbody>__METRICS__</tbody></table>
<h2>Remaining work for a publication study</h2><ul><li>Verify animal, exposure, sham, and baseline records.</li><li>Expert review/correction of masks and longitudinal registration.</li><li>Independent real-abnormality reference information and suitable controls.</li><li>Confirmatory animal-level evaluation and a focused novelty comparison.</li></ul>
<script id="review-data" type="application/json">__DATA__</script>
<script>(()=>{const rows=JSON.parse(document.getElementById('review-data').textContent);const select=document.getElementById('animal');function draw(){const row=rows[Number(select.value)];document.getElementById('status').textContent=row.animal+' | '+row.split+' | '+row.status+' | '+row.flags;document.getElementById('qc').src='subjects/'+row.animal+'/processing_qc.png';document.getElementById('map-section').hidden=!row.maps;if(row.maps)document.getElementById('maps').src='subjects/'+row.animal+'/change_maps.png';else document.getElementById('maps').removeAttribute('src')}select.addEventListener('change',draw);draw()})();</script></html>'''
import numpy as np
metrics={}
for run in ['atlas-development-seed42','atlas-development-seed43','atlas-development-seed44']:
 result=json.loads((root.parent/'runs'/run/'results.json').read_text())
 for method,r in result['synthetic_test_results'].items():metrics.setdefault(method,[]).append(r['mean_across_animals']['positive_slice_dice_at_0_5'])
labels={'erm_no_adaptation':'Ordinary training','erm_baseline_adaptation':'Ordinary training + baseline adaptation','fomaml_baseline_adaptation':'Meta-learning + baseline adaptation'}
table=''.join(f'<tr><td>{labels[k]}</td><td>{np.mean(v):.3f}</td><td>{np.std(v,ddof=1):.3f}</td></tr>' for k,v in metrics.items())
(root/'review.html').write_text(fragment.replace('__OPTIONS__',options).replace('__METRICS__',table).replace('__DATA__',data))
(root/'aggregate_synthetic_results.json').write_text(json.dumps({k:{'mean_dice':float(np.mean(v)),'sd_across_seeds':float(np.std(v,ddof=1)),'seed_values':v} for k,v in metrics.items()},indent=2)+'\n')
print(root/'review.html')
