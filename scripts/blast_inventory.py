"""Inventory the blast cohort: locate T2, DTI, reverse-PE b0 and PFC MRS per session; never edits sources."""
import json
import re
from pathlib import Path

from prepare_data import ROOT, parameters

OUT = ROOT / 'ferret-study/blast-pipeline'
GROUPS = ROOT / 'Ferret/MRI groups'


def scans(session):
    found = {}
    for d in sorted((p for p in session.iterdir() if p.name.isdigit()), key=lambda p: int(p.name)):
        if not (d / 'acqp').exists():
            continue
        name = parameters(d / 'acqp').get('ACQ_protocol_name', '')
        complete = (d / 'pdata/1/2dseq').exists()
        for key, pattern in [('t2', r'04_RARE_T2_axial$'), ('dti', r'05_DTI_EPI_2Bval_30dir'),
                             ('dti_neg', r'05b_DTI_EPI_B0_neg'), ('mrs', r'08c_sTE_PRESS_1H_PFC$')]:
            if re.search(pattern, name) and complete:
                found.setdefault(key, []).append(d.name)
    return found


def main():
    rows = []
    for session in sorted(p for p in GROUPS.rglob('subject')):
        session = session.parent
        rel = session.relative_to(GROUPS).parts
        time, group = rel[0], (rel[1] if len(rel) > 2 else None)
        subj = parameters(session / 'subject')
        found = scans(session)
        csvs = sorted(str(p.relative_to(session)) for p in session.rglob('*.csv'))
        row = {'session': str(session.relative_to(ROOT)), 'folder_time': time, 'group': group,
               'animal': session.name, 'study_name': subj.get('SUBJECT_study_name'),
               'sex': subj.get('SUBJECT_sex_animal'), 'weight': subj.get('SUBJECT_weight'),
               'date': (subj.get('SUBJECT_date') or '')[:30], 'scans': found, 'lcmodel_csv': csvs}
        if found.get('dti'):
            m = parameters(session / found['dti'][-1] / 'method')
            row['dti_params'] = {k: m.get(k) for k in ['PVM_DwAoImages', 'PVM_DwNDiffDir', 'PVM_DwBvalEach',
                                 'PVM_NRepetitions', 'PVM_EncNReceivers', 'PVM_SPackArrSliceOrient',
                                 'PVM_SPackArrReadOrient', 'PVM_EffSWh', 'PVM_EchoTime']}
        if found.get('mrs'):
            m = parameters(session / found['mrs'][-1] / 'method')
            row['mrs_voxel'] = {k: m.get(k) for k in ['PVM_VoxArrPosition', 'PVM_VoxArrSize', 'PVM_VoxArrGradOrient']}
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'inventory.json').write_text(json.dumps(rows, indent=1) + '\n')
    for r in rows:
        print(r['folder_time'], r['group'], r['animal'], r['sex'], r['weight'],
              {k: v for k, v in r['scans'].items()}, len(r['lcmodel_csv']))


if __name__ == '__main__':
    main()
