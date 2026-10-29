"""One entry point for the computational development pipeline; preserves completed runs."""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ferret-mpl')
from pathlib import Path
import subprocess
import sys

scripts=Path(__file__).resolve().parent
study=scripts.parent
def run(script,*args):
    subprocess.run([sys.executable,'-u',str(scripts/script),*args],check=True)

if __name__=='__main__':
    run('full_pipeline.py','prepare')
    run('repair_registration.py')
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(scripts),'-p','test_*.py','-v'],check=True)
    # Individual training runs deliberately refuse to overwrite a partial run.
    seeds=[42,43,44]
    for seed in seeds:
        name=f'atlas-development-seed{seed}'
        if (study/'runs'/name/'results.json').exists():
            print('Using completed fixed run:',name,flush=True)
            continue
        run('study.py','--device','mps','--steps','1000','--eval-interval','100',
            '--seed',str(seed),'--run',name,'--data-dir',str(study/'full-pipeline/data'),
            '--mask-cache',str(study/'full-pipeline/data/brain_masks.npz'))
    run('full_pipeline.py','infer','--runs',*[f'atlas-development-seed{s}' for s in seeds])
    run('build_review.py')
