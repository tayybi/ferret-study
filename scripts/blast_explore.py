"""Post-hoc exploratory analyses E1-E4 (ANALYSIS_ADDENDUM_EXPLORATORY.md). Results are labelled exploratory."""
import json

import numpy as np
import nibabel as nib
import ants
from scipy import stats

import blast_stats as bs
from prepare_data import ROOT

BP, MD = bs.BP, bs.MD


def atlas_labels(brain):
    lab = nib.load(ROOT / 'ferret-study/reference/evDTI_SEGMENTATION.nii.gz').get_fdata().astype(np.float32)
    lab = lab.transpose(0, 2, 1)[:, ::-1, :].copy()  # same axis convention as full_pipeline.atlas()
    im = ants.from_numpy(lab, spacing=(.25, .25, .25))
    ref = ants.from_numpy(np.zeros(brain.shape, np.float32), spacing=(.5, .5, .5))
    return ants.resample_image_to_target(im, ref, interp_type='nearestNeighbor').numpy().astype(int)


def welch_perm(a, b, rng, n=5000):
    t0 = stats.ttest_ind(a, b, equal_var=False).statistic
    x = np.concatenate([a, b]); k = len(a); c = 0
    for _ in range(n):
        p = rng.permutation(x); c += abs(stats.ttest_ind(p[:k], p[k:], equal_var=False).statistic) >= abs(t0)
    return float(t0), (c + 1) / (n + 1)


def main():
    rng = np.random.default_rng(7)
    rows = json.loads((MD / 'rows.json').read_text())
    brain = np.load(MD / 'brain_mask.npy'); vol = bs.Vol(brain)
    g = np.array([r['group'] for r in rows]); t = np.array([str(r['time']) for r in rows])
    cohort = np.array([r['cohort'] for r in rows])
    sham = [i for i in range(len(rows)) if cohort[i] == 'blast' and g[i] == 'Sham']
    inv = {r['session']: r for r in json.loads((BP / 'inventory.json').read_text())}
    import re
    years = np.array([int(re.search(r'(20\d\d)', inv.get(r['session'], {}).get('date') or '2023')[1]) for r in rows])
    X = np.stack([np.stack([np.load(BP / 'sessions' / r['id'] / 'channels.npz')[c].astype(np.float32)[brain] for c in bs.ALL]) for r in rows])
    out = {'label': 'EXPLORATORY (post hoc); see ANALYSIS_ADDENDUM_EXPLORATORY.md'}

    # E1
    R, _ = bs.load_residuals('main')
    e1 = {}
    for name, src, absolute in [('model', R, False), ('B1_raw_z', X, True)]:
        S = bs.zscores(src, sham, vol, absolute=absolute)
        top = np.sort(S, 1)[:, -int(.01 * S.shape[1]):].mean(1)
        blast = cohort == 'blast'
        res = bs.endpoints(top[blast], [rows[i] for i in np.nonzero(blast)[0]], years[blast], rng)
        res['fiskum_vs_sham'] = {'fiskum_mean': float(top[cohort != 'blast'].mean()), 'sham_mean': float(top[sham].mean()),
                                 'hedges_g': bs.hedges(top[cohort != 'blast'], top[sham]),
                                 'p_perm_mc': bs.perm_mc(top[cohort != 'blast'], top[sham], rng)}
        res['per_animal'] = {rows[i]['id']: float(top[i]) for i in range(len(rows))}
        e1[name] = res
        print('E1', name, json.dumps(res['primary']), 'dose', res['dose_trend'], 'fiskum', res['fiskum_vs_sham'], flush=True)
    out['E1_top1pct_score'] = e1

    # E2
    lab = atlas_labels(brain)[brain]
    regions = [l for l in np.unique(lab) if l > 0 and (lab == l).sum() >= 200]
    e2 = []
    for l in regions:
        m = lab == l
        for c in ['FA', 'MD', 'T2', 'logJ']:
            v = X[:, bs.ALL.index(c)][:, m].mean(1)
            for tm in ['3d', '7d']:
                a = v[(g == '20psi') & (t == tm)]; b = v[(g == 'Sham') & (t == tm)]
                tt, p = welch_perm(a, b, rng)
                e2.append({'region_label': int(l), 'n_voxels': int(m.sum()), 'metric': c, 'time': tm,
                           'mean_20psi': float(a.mean()), 'mean_sham': float(b.mean()),
                           'pct_diff': float(100 * (a.mean() - b.mean()) / abs(b.mean())) if c != 'logJ' else None,
                           'hedges_g': bs.hedges(a, b), 't': tt, 'p_perm': p})
    q = bs.bh(np.array([e['p_perm'] for e in e2]), .05)
    ps = np.array([e['p_perm'] for e in e2]); o = np.argsort(ps); m_ = len(ps)
    qv = np.minimum.accumulate((ps[o] * m_ / np.arange(1, m_ + 1))[::-1])[::-1]
    qvals = np.empty(m_); qvals[o] = np.minimum(qv, 1)
    for e, r_, qq in zip(e2, q, qvals):
        e['q_bh'] = float(qq); e['significant_fdr05'] = bool(r_)
    e2.sort(key=lambda e: e['p_perm'])
    out['E2_regions'] = {'n_tests': len(e2), 'n_regions': len(regions), 'n_significant_fdr05': int(q.sum()), 'top10': e2[:10]}
    print('E2 tests', len(e2), 'sig', int(q.sum()), json.dumps(e2[:5]), flush=True)

    # E3
    z = np.load(MD / 'main' / 'synthetic.npz')
    res_syn = z['residuals'].astype(np.float32).mean(0); raw = z['raw'].astype(np.float32)
    les = z['lesion']; sidx = list(z['sham_index']); amps = z['amps']
    e3 = {}
    for name, src, base, absolute in [('model', res_syn, R, False), ('B1_raw_z', raw, X, True)]:
        curve = []
        for a, amp in enumerate(amps):
            sens = []
            for j, i in enumerate(sidx):
                R2 = base.copy(); R2[i] = src[j, a]
                S2 = bs.zscores(R2, sham, vol, absolute=absolute)
                pool = np.concatenate([S2[k] for k in sham if k != i])
                thr = np.percentile(pool, 99.9)
                sens.append(float((S2[i][les[j]] > thr).mean()))
            curve.append({'amp_sd': int(amp), 'voxel_sensitivity_at_null_99.9pct': float(np.mean(sens)),
                          'sd_across_shams': float(np.std(sens))})
        e3[name] = curve
        print('E3', name, curve, flush=True)
    out['E3_uncorrected_sensitivity'] = e3

    # E4
    e4 = {}
    for c in ['FA', 'MD', 'T2']:
        v = X[:, bs.ALL.index(c)].mean(1)
        a, b = v[cohort != 'blast'], v[sham]
        e4[c] = {'fiskum_mean': float(a.mean()), 'sham_mean': float(b.mean()), 'hedges_g': bs.hedges(a, b),
                 'p_perm_mc': bs.perm_mc(a, b, rng)}
    out['E4_domain_shift'] = e4
    print('E4', e4)
    (BP / 'results_exploratory.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
