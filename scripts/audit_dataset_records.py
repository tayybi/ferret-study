"""Audit original Ferret records; save evidence without altering scientific labels."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date
import csv, hashlib, json, re
from prepare_data import ROOT, parameters

OUT=ROOT/'ferret-study/full-pipeline/record-audit'
SOURCE=ROOT/'Ferret'

def rel(p): return str(p.relative_to(ROOT))
def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    files=sorted(p for collection in ['Fiskum','MRI groups'] for p in (SOURCE/collection).rglob('*') if p.is_file() and not p.name.startswith('.'))
    sessions=[];by_path={}
    for p in files:
        if p.name!='subject':continue
        d=parameters(p)
        row={'source':rel(p),'label':d.get('SUBJECT_study_name',''),'date':d.get('SUBJECT_date',''),
             'subject_id':d.get('SUBJECT_id',''),'subject_uid':d.get('SUBJECT_instance_uid',''),
             'study_uid':d.get('SUBJECT_study_instance_uid',''),'birth_field':d.get('SUBJECT_dbirth',''),
             'sex':d.get('SUBJECT_sex',''),'weight':d.get('SUBJECT_weight',''),
             'remarks':d.get('SUBJECT_remarks',''),'purpose':d.get('SUBJECT_purpose',''),
             'comment':d.get('SUBJECT_comment',''),'folder_group':next((x for x in p.parts if x in ['Sham','10psi','20psi']),''),
             'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        sessions.append(row);by_path[p.parent]=row
    write_csv('all_sessions.csv',sessions)
    # Search explicit text/metadata files, not scanner image or raw signal binaries.
    names={'subject','acqp','method','visu_pars','reco','id','configscan','specpar','ResultState','AdjStatePerStudy','AdjStatePerScan'}
    suffixes={'.txt','.csv','.table','.control','.coord','.jcamp','.par','.info'}
    keyword=re.compile(r'BC[FR]\d+|blast|sham|injur\w*|expos\w*|\b(?:10|20)\s*psi\b',re.I)
    hits=[];searched=0
    for p in files:
        if p.name not in names and p.suffix.lower() not in suffixes:continue
        searched+=1
        for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
            if keyword.search(line):hits.append({'source':rel(p),'line':i,'text':line[:2000]})
    (OUT/'metadata_search_hits.json').write_text(json.dumps(hits,indent=2))
    pairs=[];checks=[]
    for folder in sorted((ROOT/'ferret-study/full-pipeline/subjects').iterdir()):
        if not folder.is_dir():continue
        info=json.loads((folder/'processing.json').read_text())
        visits=[]
        for kind,source in [('baseline',info['baseline_source']),('followup',info['followup']['source'] if info['followup'] else None)]:
            if not source:continue
            recon=ROOT/source;session=by_path[recon.parents[2]]
            v=parameters(recon/'visu_pars');a=parameters(recon.parents[1]/'acqp')
            checks.append({'animal':folder.name,'visit':kind,'subject_file':session['source'],
                'subject_label':session['label'],'visu_file':rel(recon/'visu_pars'),
                'visu_label':v.get('VisuStudyId',''),'subject_date':session['date'],
                'visu_date':v.get('VisuStudyDate',''),'acquisition_time':a.get('ACQ_time',''),
                'label_matches':session['label']==v.get('VisuStudyId',''),
                'study_uid_matches':session['study_uid']==v.get('VisuStudyUid',''),
                'date_matches':session['date']==v.get('VisuStudyDate',''),
                'acquisition_same_date':session['date'][:10]==a.get('ACQ_time','')[:10]})
            visits.append(session)
        b=visits[0];f=visits[1] if len(visits)>1 else None
        pairs.append({'animal':folder.name,'baseline_label':b['label'],'baseline_date':b['date'][:10],
            'followup_label':f['label'] if f else '', 'followup_date':f['date'][:10] if f else '',
            'elapsed_calendar_days':(date.fromisoformat(f['date'][:10])-date.fromisoformat(b['date'][:10])).days if f else '',
            'matching_animal_label':bool(f and b['label'].split('_')[0]==f['label'].split('_')[0]),
            'same_subject_uid':bool(f and b['subject_uid']==f['subject_uid']),
            'pairing_evidence':'matching recorded BCF label; no independent identity confirmation' if f else 'unmatched BCF94; BCR94 follow-up exists',
            'exposure_date':'not found in searched records','sham_or_injury_group':'not found for BCF ID',
            'baseline_subject_file':b['source'],'followup_subject_file':f['source'] if f else ''})
    write_csv('selected_pairs.csv',pairs);write_csv('selected_scan_crosschecks.csv',checks)
    ids=defaultdict(list)
    for s in sessions:ids[s['subject_id']].append(s['label'])
    duplicate_ids={k:v for k,v in ids.items() if len(set(v))>1}
    fiskum_hits=[h for h in hits if '/Fiskum/' in h['source'] and re.search(r'blast|sham|injur\w*|expos\w*|\b(?:10|20)\s*psi\b',h['text'],re.I)]
    crosslinks=[h for h in hits if '/MRI groups/' in h['source'] and re.search(r'BC[FR]\d+',h['text'],re.I)]
    summary={'sessions':len(sessions),'source_files_inventoried':len(files),'text_metadata_files_searched':searched,
        'selected_scans_crosschecked':len(checks),'pairs_with_matching_BCF_labels':sum(r['matching_animal_label'] for r in pairs),
        'crosscheck_failures':[c for c in checks if not all(c[k] for k in ['label_matches','study_uid_matches','date_matches','acquisition_same_date'])],
        'generic_subject_ids_reused_for_different_study_labels':duplicate_ids,
        'fiskum_exposure_keyword_hits':fiskum_hits,'BCF_BCR_links_in_MRI_groups':crosslinks,
        'group_folder_counts':dict(Counter(s['folder_group'] for s in sessions if s['folder_group'])),
        'document_extension_counts':dict(Counter(p.suffix.lower() for p in files if p.suffix.lower() in {'.pdf','.xlsx','.xls','.doc','.docx','.txt'})),
        'search_scope':'Original local Ferret/Fiskum and Ferret/MRI groups. Generated reports excluded. Text and metadata search; binary payloads not searched as text.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    assert len(sessions)==112 and len(checks)==39 and len(pairs)==20
    print(json.dumps({k:v for k,v in summary.items() if k!='generic_subject_ids_reused_for_different_study_labels'},indent=2))

if __name__=='__main__':main()
