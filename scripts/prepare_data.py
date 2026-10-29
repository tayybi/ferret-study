"""Prepare a small, provisional Ferret T2 technical benchmark; never edits sources."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'ferret-study'


def parameters(path):
    text = path.read_text(errors='replace')
    result = {}
    for key, value in re.findall(r'^##\$([^=]+)=(.*?)(?=^##|\Z)', text, re.M | re.S):
        value = '\n'.join(line for line in value.strip().splitlines() if not line.startswith('$$'))
        result[key] = re.sub(r'^\([^)]*\)\s*', '', value).strip().strip('<>')
    return result


def decode(path):
    p = parameters(path / 'visu_pars')
    shape = [int(x) for x in p['VisuCoreSize'].split()]
    if p['VisuCoreDim'] != '2':
        raise ValueError('This pilot accepts 2D multislice anatomy only')
    count = int(p['VisuCoreFrameCount'])
    kinds = {'_16BIT_SGN_INT': 'i2', '_32BIT_SGN_INT': 'i4', '_32BIT_FLOAT': 'f4', '_8BIT_UNSGN_INT': 'u1'}
    dtype = np.dtype(('<' if p['VisuCoreByteOrder'] == 'littleEndian' else '>') + kinds[p['VisuCoreWordType']])
    a = np.fromfile(path / '2dseq', dtype=dtype)
    if a.size != count * int(np.prod(shape)):
        raise ValueError('Binary/metadata size mismatch: ' + str(path))
    a = a.reshape((count, *shape[::-1])).astype(np.float32)
    for key, default, op in [('VisuCoreDataSlope', '1', np.multiply), ('VisuCoreDataOffs', '0', np.add)]:
        v = np.asarray([float(x) for x in p.get(key, default).split()], dtype=np.float32)
        if len(v) not in [1, count]:
            raise ValueError('Unsupported intensity scaling')
        a = op(a, v.reshape((-1, 1, 1)))
    return a, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128)
    args = parser.parse_args()
    inventory = json.loads((ROOT / 'Ferret/visual-overview/session-inventory.json').read_text())
    rows, groups = [], defaultdict(list)
    for session in inventory:
        match = re.fullmatch(r'(BCF\d+)_baseline', session['label'], re.I)
        if not match:
            continue
        animal = match[1].upper()
        candidates = [s for s in session['scans'] if s['method'] == '<Bruker:RARE>' and s['protocol'] == '04_RARE_T2_axial']
        if len(candidates) != 1:
            raise ValueError('Expected one baseline T2 axial acquisition: ' + animal)
        scan = candidates[0]
        reconstruction = ROOT / 'Ferret' / session['path'] / scan['number'] / 'pdata/1'
        rows.append({'animal_id': animal, 'label': session['label'], 'date': session['date'],
                     'collection': session['collection'], 'scan': scan['number'],
                     'source': str(reconstruction.relative_to(ROOT)), 'protocol': scan['protocol'],
                     'status': 'Scanner baseline label only; health and exposure not verified',
                     'candidate_followups': [{'label': s['label'], 'date': s['date'], 'path': s['path']}
                         for s in inventory if re.fullmatch(re.escape(animal) + r'_(1month|6month)', s['label'], re.I)]})
        groups[session['collection']].append(animal)
    ids = [r['animal_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated baseline animal identifier')
    # Freeze splits before training; balance acquisition year without claiming independent sites.
    split = {}
    rng = np.random.default_rng(20261026)
    for collection in sorted(groups):
        animals = sorted(groups[collection])
        rng.shuffle(animals)
        n = len(animals)
        if n < 5:
            raise ValueError('Too few animals for this pilot split')
        n_val = n_test = max(1, round(n * .2))
        for index, animal in enumerate(animals):
            split[animal] = 'validation' if index < n_val else ('test' if index < n_val + n_test else 'train')
    arrays = {}
    for row in sorted(rows, key=lambda r: r['animal_id']):
        path = ROOT / row['source']
        a, p = decode(path)
        row['original_shape'] = list(a.shape)
        row['extent_mm'] = p['VisuCoreExtent']
        row['thickness_mm'] = p['VisuCoreFrameThickness']
        row['orientation'] = p.get('VisuCoreOrientation', '')
        row['source_sha256'] = hashlib.sha256((path / '2dseq').read_bytes()).hexdigest()
        # Fixed central-slice selection; no outcomes or test-derived population statistics used.
        indices = np.arange(3, min(len(a) - 3, 15))
        if len(indices) < 4:
            raise ValueError('Insufficient central slices')
        a = a[indices]
        lo, hi = np.percentile(a, [1, 99.5])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        reduced = np.stack([np.asarray(Image.fromarray(frame).resize((args.size, args.size), Image.Resampling.BILINEAR)) for frame in a]).astype(np.float32)
        arrays[row['animal_id']] = reduced
        row.update(split=split[row['animal_id']], stored_slice_indices_zero_based=indices.tolist(),
                   processed_shape=list(reduced.shape), normalization_percentiles=[float(lo), float(hi)])
    out = STUDY / 'data'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'baseline_t2.npz', **arrays)
    (out / 'manifest.json').write_text(json.dumps({'scope': 'Provisional synthetic-change technical pilot; NOT injury validation',
        'seed': 20261026, 'image_size': args.size, 'brain_extraction': 'Not performed: no validated ferret brain-extraction model supplied',
        'registration': 'Not needed for synthetic pairs from the same baseline; real follow-up evaluation requires registration',
        'animals': rows}, indent=2) + '\n')
    print(json.dumps({'animals': len(rows), 'splits': {name: sum(r['split'] == name for r in rows) for name in ['train','validation','test']},
                      'slices': sum(len(x) for x in arrays.values()), 'cache_MiB': (out/'baseline_t2.npz').stat().st_size/2**20}, indent=2))


if __name__ == '__main__':
    main()
