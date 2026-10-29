"""Run fixed, predeclared development configurations without tuning on holdouts."""
import os,subprocess,sys,json,time
from pathlib import Path
study=Path(__file__).resolve().parents[1]
logs=study/'full-pipeline/training_logs';logs.mkdir(exist_ok=True)
started=time.perf_counter()
for seed in [42,43,44]:
 name=f'atlas-development-seed{seed}'
 command=[sys.executable,'-u',str(study/'scripts/study.py'),'--device','mps','--steps','1000','--eval-interval','100','--seed',str(seed),
          '--run',name,'--data-dir',str(study/'full-pipeline/data'),'--mask-cache',str(study/'full-pipeline/data/brain_masks.npz')]
 print(json.dumps({'starting':name,'episodes_per_method':1000}),flush=True)
 with (logs/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 if result.returncode:
  print((logs/(name+'.log')).read_text()[-6000:],flush=True)
  raise SystemExit(result.returncode)
 print(json.dumps({'completed':name,'total_elapsed_seconds':round(time.perf_counter()-started,1)}),flush=True)
