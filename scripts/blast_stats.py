"""Scores, conformal calibration, locked endpoints, comparators, figures. Follows blast-pipeline/ANALYSIS_PLAN.md."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ferret-mpl')
import hashlib
import itertools
import json
import re

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare_data import ROOT

BP = ROOT / 'ferret-study/blast-pipeline'
MD = BP / 'models'
FIG = BP / 'figures'
ALL = ['T2', 'FA', 'MD', 'AD', 'RD', 'logJ']
Q = .05
METABOLITES = ['NAA+NAAG/Cr+PCr', 'Glu/Cr+PCr', 'Ins/Cr+PCr', 'Lac/Cr+PCr']
B = 10000


def check_lock():
    want = (BP / 'analysis_plan.sha256').read_text().split()[0]
    got = hashlib.sha256((BP / 'ANALYSIS_PLAN.md').read_bytes()).hexdigest()
    if want != got:
        raise SystemExit('ANALYSIS_PLAN.md changed after locking; refusing to compute endpoints')


class Vol:
    """Maps flattened analysis-mask vectors to volumes for smoothing."""

    def __init__(self, brain):
        self.brain = brain
        self.norm = gaussian_filter(brain.astype(np.float32), 1.0)
        self.norm15 = gaussian_filter(brain.astype(np.float32), 1.5)

    def smooth(self, v, sigma=1.0):
        a = np.zeros(self.brain.shape, np.float32)
        a[self.brain] = v
        n = self.norm if sigma == 1.0 else self.norm15
        return (gaussian_filter(a, sigma) / np.maximum(n, 1e-6))[self.brain]


def zscores(R, sham, vol, absolute=False):
    """Per-animal max-over-channel z against sham statistics (leave-one-out for shams), smoothed sigma=1."""
    n, C, V = R.shape
    S1, S2 = R[sham].sum(0), (R[sham] ** 2).sum(0)
    out = np.zeros((n, V), np.float32)
    cache = {}
    for i in range(n):
        if i in sham:
            m = len(sham) - 1
            mu, s2 = (S1 - R[i]) / m, (S2 - R[i] ** 2) / m
        else:
            m = len(sham)
            mu, s2 = S1 / m, S2 / m
        key = i if i in sham else -1
        if key not in cache:
            sd = np.sqrt(np.maximum(s2 - mu ** 2, 0) * m / (m - 1))
            sd = np.stack([vol.smooth(sd[c], 1.5) for c in range(C)])
            floor = np.percentile(sd, 10, axis=1, keepdims=True)
            cache[key] = (mu, np.maximum(sd, floor))
        mu, sd = cache[key]
        z = (R[i] - mu) / sd
        z = np.abs(z) if absolute else z
        out[i] = vol.smooth(z.max(0), 1.0)
    return out


def conformal(S, sham):
    """Per-voxel conformal p-values vs pooled sham scores (excluding the animal), BH within animal."""
    pools = {}
    full = np.sort(S[sham].ravel())
    P, rej = np.zeros_like(S), np.zeros(S.shape, bool)
    for i in range(len(S)):
        pool = full if i not in sham else np.sort(np.concatenate([S[j] for j in sham if j != i]))
        p = (1 + len(pool) - np.searchsorted(pool, S[i], side='left')) / (len(pool) + 1)
        P[i] = p
        rej[i] = bh(p, Q)
    return P, rej


def bh(p, q):
    o = np.argsort(p)
    thr = q * np.arange(1, len(p) + 1) / len(p)
    ok = p[o] <= thr
    r = np.zeros(len(p), bool)
    if ok.any():
        r[o[:np.max(np.nonzero(ok)[0]) + 1]] = True
    return r


def hedges(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    d = (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def exact_perm(a, b):
    x = np.concatenate([a, b]); obs = abs(a.mean() - b.mean()); n = len(a)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(x)), n):
        m = np.zeros(len(x), bool); m[list(idx)] = True
        cnt += abs(x[m].mean() - x[~m].mean()) >= obs - 1e-12; tot += 1
    return cnt / tot


def boot_ci(a, b, rng):
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def freedman_lane(y, full, reduced, col, strata, rng):
    """Permutation p for coefficient `col` of `full` using Freedman-Lane, permuting within strata."""
    def tstat(X, y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        s2 = r @ r / (len(y) - X.shape[1])
        cov = s2 * np.linalg.pinv(X.T @ X)
        return beta[col] / np.sqrt(cov[col, col]), beta[col]
    t0, b0 = tstat(full, y)
    br, *_ = np.linalg.lstsq(reduced, y, rcond=None)
    fit, res = reduced @ br, y - reduced @ br
    cnt = 0
    for _ in range(B):
        rp = res.copy()
        for s in np.unique(strata):
            ix = np.nonzero(strata == s)[0]; rp[ix] = res[rng.permutation(ix)]
        cnt += abs(tstat(full, fit + rp)[0]) >= abs(t0)
    return float(b0), float(t0), (cnt + 1) / (B + 1)


def spearman_perm(x, y, rng):
    r = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(B)])
    return float(r), float((np.sum(np.abs(null) >= abs(r)) + 1) / (B + 1))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj.tolist()


def load_residuals(name, seeds=None):
    files = sorted((MD / name).glob('residuals_seed*.npy'))
    if seeds is not None:
        files = [f for f in files if int(re.search(r'seed(\d+)', f.name)[1]) in seeds]
    return np.mean([np.load(f).astype(np.float32) for f in files], 0), [f.name for f in files]


def endpoints(burden, rows, years, rng):
    g = np.array([r['group'] for r in rows]); t = np.array([str(r['time']) for r in rows])
    blast = np.array([r['cohort'] == 'blast' for r in rows])
    a = burden[(g == '20psi') & (t == '7d')]; b = burden[(g == 'Sham') & (t == '7d')]
    res = {'primary': {'mean_20psi_7d': float(a.mean()), 'mean_sham_7d': float(b.mean()), 'n': [len(a), len(b)],
                       'diff': float(a.mean() - b.mean()), 'ci95': boot_ci(a, b, rng),
                       'hedges_g': hedges(a, b), 'p_exact': exact_perm(a, b)}}
    dose = np.array([{'Sham': 0, '10psi': 1, '20psi': 2}.get(x, -1) for x in g], float)[blast]
    tt = (t[blast] == '7d').astype(float); yr = years[blast]
    yrs = [(yr == y).astype(float) for y in np.unique(yr)[1:]]
    y = burden[blast]; one = np.ones_like(y)
    full = np.column_stack([one, dose, tt, *yrs]); red = np.column_stack([one, tt, *yrs])
    bcoef, tval, p = freedman_lane(y, full, red, 1, tt, rng)
    res['dose_trend'] = {'coef_per_step': bcoef, 't': tval, 'p_perm': p, 'n': int(len(y))}
    full2 = np.column_stack([one, dose, tt, dose * tt, *yrs])
    bcoef, tval, p = freedman_lane(y, full2, full, 3, tt, rng)
    res['dose_x_time'] = {'coef': bcoef, 't': tval, 'p_perm': p}
    res['group_means'] = {f'{tm}-{gr}': {'mean': float(burden[(g == gr) & (t == tm)].mean()),
                                         'sd': float(burden[(g == gr) & (t == tm)].std(ddof=1)),
                                         'n': int(((g == gr) & (t == tm)).sum())}
                          for tm in ['3d', '7d'] for gr in ['Sham', '10psi', '20psi']}
    for tm in ['3d']:
        a3 = burden[(g == '20psi') & (t == tm)]; b3 = burden[(g == 'Sham') & (t == tm)]
        res[f'secondary_20psi_vs_sham_{tm}'] = {'diff': float(a3.mean() - b3.mean()), 'hedges_g': hedges(a3, b3),
                                                 'p_perm_mc': perm_mc(a3, b3, rng)}
    return res


def perm_mc(a, b, rng):
    x = np.concatenate([a, b]); obs = abs(a.mean() - b.mean()); n = len(a); c = 0
    for _ in range(B):
        p = rng.permutation(x); c += abs(p[:n].mean() - p[n:].mean()) >= obs - 1e-12
    return (c + 1) / (B + 1)


def b2_maxT(X, rows, brain, vol, rng, nperm=5000):
    """Comparator B2: voxelwise Welch t (smoothed FA, MD), sham vs 20 psi per time point, max-T FWER."""
    out = {}
    g = np.array([r['group'] for r in rows]); t = np.array([str(r['time']) for r in rows])
    for ch in ['FA', 'MD']:
        c = ALL.index(ch)
        V = np.stack([vol.smooth(X[i, c], 1.0) for i in range(len(rows))])
        for tm in ['3d', '7d']:
            ia = np.nonzero((g == '20psi') & (t == tm))[0]; ib = np.nonzero((g == 'Sham') & (t == tm))[0]
            data = V[np.concatenate([ia, ib])]; na = len(ia)

            def tmap(d):
                a, b = d[:na], d[na:]
                return (a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-12)
            obs = tmap(data)
            mx = np.array([np.abs(tmap(data[rng.permutation(len(data))])).max() for _ in range(nperm)])
            thr = np.percentile(mx, 95)
            out[f'{ch}_{tm}'] = {'n_sig_voxels_fwer05': int((np.abs(obs) > thr).sum()), 'max_abs_t': float(np.abs(obs).max()),
                                 'fwer_p_min': float((np.sum(mx >= np.abs(obs).max()) + 1) / (nperm + 1)),
                                 'direction_of_peak': 'up' if obs[np.argmax(np.abs(obs))] > 0 else 'down'}
    return out


def main():
    check_lock()
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = json.loads((MD / 'rows.json').read_text())
    brain = np.load(MD / 'brain_mask.npy')
    vol = Vol(brain)
    inv = {r['session']: r for r in json.loads((BP / 'inventory.json').read_text())}
    years = np.array([int(re.search(r'(20\d\d)', inv.get(r['session'], {}).get('date') or '2023')[1]) for r in rows])
    blast_sham = [i for i, r in enumerate(rows) if r['cohort'] == 'blast' and r['group'] == 'Sham']
    procs = [json.loads((BP / 'sessions' / r['id'] / 'processing.json').read_text()) for r in rows]
    results = {'n_sessions': len(rows), 'n_voxels': int(brain.sum()), 'analysis_volume_mm3': float(brain.sum() * .125),
               'n_by_group': {}}
    for r in rows:
        k = f"{r['time']}-{r['group']}" if r['cohort'] == 'blast' else 'fiskum_baseline'
        results['n_by_group'][k] = results['n_by_group'].get(k, 0) + 1
    excluded = [json.loads(p.read_text()) for p in (BP / 'sessions').glob('*/processing.json')]
    results['excluded_sessions'] = [{'id': e['id'], 'reason': e.get('error') or 'template_correlation<0.35'}
                                    for e in excluded if e['id'] not in {r['id'] for r in rows}]

    # Raw channels within mask (for B1, B2, synthetic sd)
    X = np.stack([np.stack([np.load(BP / 'sessions' / r['id'] / 'channels.npz')[c].astype(np.float32)[brain] for c in ALL])
                  for r in rows])

    variants = {}
    for name in ['main', 'fiskum_pool', 't2_only', 'dti_only', 'no_logj', 'main_adapt20']:
        if not (MD / name).exists():
            continue
        R, files = load_residuals(name)
        variants[name] = (R, files)
    for s in [0, 1, 2]:
        if (MD / 'main' / f'residuals_seed{s}.npy').exists():
            variants[f'main_seed{s}'] = load_residuals('main', [s])
    # B1: raw-value z (two-sided)
    variants['B1_raw_z'] = (X, ['raw channels'])

    scored = {}
    for name, (R, files) in variants.items():
        S = zscores(R, blast_sham, vol, absolute=name.startswith('B1'))
        P, rej = conformal(S, blast_sham)
        burden = rej.mean(1)
        sham_b = burden[blast_sham]
        animal_p = np.array([(1 + np.sum(np.delete(sham_b, blast_sham.index(i)) >= burden[i])) / len(sham_b) if i in blast_sham
                             else (1 + np.sum(sham_b >= burden[i])) / (len(sham_b) + 1) for i in range(len(rows))])
        blast_rows = [i for i, r in enumerate(rows) if r['cohort'] == 'blast']
        res = endpoints(burden[blast_rows], [rows[i] for i in blast_rows], years[blast_rows], rng)
        res['false_alarm_shams'] = {'mean_voxel_fp_fraction': float(sham_b.mean()),
                                    'share_with_any_rejection': float((rej[blast_sham].sum(1) > 0).mean()),
                                    'max_voxel_fp_fraction': float(sham_b.max())}
        fis = [i for i, r in enumerate(rows) if r['cohort'] != 'blast']
        if fis and name not in ('fiskum_pool',):
            res['fiskum_external_normals'] = {'mean_burden': float(burden[fis].mean()),
                                              'share_animal_p_below_0.05': float((animal_p[fis] < .05).mean()), 'n': len(fis)}
        res['sources'] = files
        scored[name] = dict(S=S, rej=rej, burden=burden, animal_p=animal_p)
        results.setdefault('variants', {})[name] = res
        print(name, json.dumps(res['primary']), 'dose p', res['dose_trend']['p_perm'], 'FA', res['false_alarm_shams'], flush=True)

    primary = 'main' if 'main' in scored else next(iter(scored))
    results['primary_variant'] = primary
    S, rej, burden = scored[primary]['S'], scored[primary]['rej'], scored[primary]['burden']

    # MRS: choose PRESS convention by cohort-wide fraction in brain (decided before correlation).
    conv = {k: np.mean([p['mrs_voxel_qc'][k]['fraction_in_brain'] for p in procs if p.get('mrs_voxel_qc')])
            for k in ['as_stored', 'flip_xy']}
    chosen = max(conv, key=conv.get)
    mrs_rows = []
    for i, (r, p) in enumerate(zip(rows, procs)):
        m = p.get('mrs')
        if r['cohort'] != 'blast' or not m:
            continue
        vm = np.load(BP / 'sessions' / r['id'] / 'channels.npz')[f'mrs_{chosen}'][brain]
        qc_ok = (m.get('FWHM_ppm', 9) <= .08) and (m.get('SNR', 0) >= 5)
        mrs_rows.append({'i': i, 'id': r['id'], 'group': r['group'], 'time': r['time'], 'variant': m['variant'],
                         'FWHM_ppm': m.get('FWHM_ppm'), 'SNR': m.get('SNR'), 'qc_ok': bool(qc_ok),
                         'voxel_in_mask_voxels': int(vm.sum()),
                         'anomaly_in_voxel': float(S[i][vm].mean()) if vm.sum() >= 5 else None,
                         **{mt: m.get(mt) for mt in METABOLITES},
                         **{mt.split('/')[0] + ' %SD': m.get(mt.split('/')[0] + ' %SD') for mt in METABOLITES}})
    corr = {}
    for mt in METABOLITES:
        sdk = mt.split('/')[0] + ' %SD'
        ok = [d for d in mrs_rows if d['qc_ok'] and d['anomaly_in_voxel'] is not None and d[mt] is not None
              and (d[sdk] is not None and d[sdk] <= 20)]
        if len(ok) >= 8:
            rho, p = spearman_perm(np.array([d['anomaly_in_voxel'] for d in ok]), np.array([d[mt] for d in ok]), rng)
            corr[mt] = {'rho': rho, 'p_perm': p, 'n': len(ok)}
        else:
            corr[mt] = {'rho': None, 'p_perm': None, 'n': len(ok)}
    tested = [m for m in METABOLITES if corr[m]['p_perm'] is not None]
    for m, a in zip(tested, holm([corr[m]['p_perm'] for m in tested])):
        corr[m]['p_holm'] = a
    results['mrs'] = {'convention_fraction_in_brain': conv, 'convention': chosen, 'n_with_mrs': len(mrs_rows),
                      'n_qc_pass': sum(d['qc_ok'] for d in mrs_rows),
                      'variants_used': {v: sum(d['variant'] == v for d in mrs_rows) for v in {d['variant'] for d in mrs_rows}},
                      'correlations': corr}
    # Metabolite group differences (descriptive)
    results['mrs']['group_means'] = {f'{tm}-{gr}': {mt: float(np.nanmean([d[mt] for d in mrs_rows if d['qc_ok'] and d['group'] == gr and d['time'] == tm and d[mt] is not None] or [np.nan]))
                                                  for mt in METABOLITES} for tm in ['3d', '7d'] for gr in ['Sham', '10psi', '20psi']}

    # B2 comparator
    results['B2_maxT'] = b2_maxT(X, rows, brain, vol, rng)

    # Synthetic lesions
    syn = MD / primary / 'synthetic.npz'
    if syn.exists():
        z = np.load(syn)
        res_syn = z['residuals'].astype(np.float32).mean(0)  # mean over seeds: (sham, amp, C, V)
        raw = z['raw'].astype(np.float32); les = z['lesion']; sidx = list(z['sham_index']); amps = z['amps']
        Rp, _ = variants[primary]
        out = {'model': [], 'B1_raw_z': []}
        for name, src in [('model', res_syn), ('B1_raw_z', raw)]:
            base = Rp if name == 'model' else X
            for a, amp in enumerate(amps):
                sens, det = [], []
                for j, i in enumerate(sidx):
                    R2 = base.copy(); R2[i] = src[j, a]
                    S2 = zscores(R2, blast_sham, vol, absolute=(name != 'model'))
                    pool = np.sort(np.concatenate([S2[k] for k in blast_sham if k != i]))
                    p = (1 + len(pool) - np.searchsorted(pool, S2[i], side='left')) / (len(pool) + 1)
                    r = bh(p, Q)
                    sens.append(float(r[les[j]].mean())); det.append(float(r[les[j]].mean() >= .1))
                out[name].append({'amp_sd': int(amp), 'voxel_sensitivity': float(np.mean(sens)),
                                  'detection_rate': float(np.mean(det)), 'n': len(sens)})
                print('synthetic', name, amp, out[name][-1], flush=True)
        results['synthetic_lesions'] = out

    # Per-animal table
    table = [{'id': r['id'], 'cohort': r['cohort'], 'group': r['group'], 'time': r['time'],
              'burden': float(burden[i]), 'mean_score': float(S[i].mean()), 'animal_conformal_p': float(scored[primary]['animal_p'][i]),
              'template_correlation': procs[i].get('template_correlation'), 'sdc': procs[i].get('sdc'),
              'motion_max_mm': procs[i].get('motion_max_translation_mm')} for i, r in enumerate(rows)]
    (BP / 'per_animal.json').write_text(json.dumps(table, indent=1) + '\n')
    (BP / 'mrs_table.json').write_text(json.dumps(mrs_rows, indent=1) + '\n')
    (BP / 'results.json').write_text(json.dumps(results, indent=1, default=float) + '\n')
    figures(rows, S, burden, brain, scored, mrs_rows, results)


def figures(rows, S, burden, brain, scored, mrs_rows, results):
    from full_pipeline import atlas
    tpl = atlas()[0].numpy()
    g = np.array([r['group'] for r in rows]); t = np.array([str(r['time']) for r in rows])
    zs = [z for z in range(brain.shape[2]) if brain[:, :, z].sum() > 300]
    pick = np.array(zs)[np.linspace(0, len(zs) - 1, 5).astype(int)]
    fig, ax = plt.subplots(6, 5, figsize=(11, 10.5))
    for row, (tm, gr) in enumerate([(tm, gr) for tm in ['3d', '7d'] for gr in ['Sham', '10psi', '20psi']]):
        idx = np.nonzero((g == gr) & (t == tm))[0]
        m = np.zeros(brain.shape); m[brain] = S[idx].mean(0)
        for c, z in enumerate(pick):
            ax[row, c].imshow(tpl[:, :, z].T, cmap='gray')
            ov = np.ma.masked_where(~brain[:, :, z].T, m[:, :, z].T)
            im = ax[row, c].imshow(ov, cmap='magma', vmin=0, vmax=2.5, alpha=.85)
            ax[row, c].axis('off')
            if c == 0: ax[row, c].set_title(f'{tm} {gr} (n={len(idx)})', fontsize=9, loc='left')
    fig.colorbar(im, ax=ax, shrink=.5, label='mean anomaly score s(v)')
    fig.savefig(FIG / 'group_maps.png', dpi=110, bbox_inches='tight'); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, name in zip(axs, ['main', 'B1_raw_z']):
        if name not in scored: continue
        b = scored[name]['burden'] * 100
        for xi, (tm, gr) in enumerate([(tm, gr) for tm in ['3d', '7d'] for gr in ['Sham', '10psi', '20psi']]):
            idx = np.nonzero((g == gr) & (t == tm))[0]
            jit = np.random.default_rng(xi).uniform(-.12, .12, len(idx))
            ax.scatter(xi + jit, b[idx], s=18, color=['#1d5f86', '#e0904c', '#c2410c'][xi % 3])
            ax.hlines(b[idx].mean(), xi - .25, xi + .25, color='k', lw=1.5)
        fis = np.nonzero(np.array([r['cohort'] for r in rows]) != 'blast')[0]
        if len(fis) and name != 'fiskum_pool':
            ax.scatter(6 + np.random.default_rng(9).uniform(-.12, .12, len(fis)), b[fis], s=18, color='#7a8794')
        ax.set_xticks(range(7), ['3d\nsham', '3d\n10psi', '3d\n20psi', '7d\nsham', '7d\n10psi', '7d\n20psi', 'Fiskum\nbaseline'], fontsize=8)
        ax.set_ylabel('anomaly burden (% of mask, BH q=0.05)'); ax.set_title(name, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'burden.png', dpi=110); plt.close(fig)

    ok = [d for d in mrs_rows if d['qc_ok'] and d['anomaly_in_voxel'] is not None]
    if ok:
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
        col = {'Sham': '#1d5f86', '10psi': '#e0904c', '20psi': '#c2410c'}
        for ax, mt in zip(axs, METABOLITES):
            d = [x for x in ok if x[mt] is not None]
            ax.scatter([x['anomaly_in_voxel'] for x in d], [x[mt] for x in d], c=[col[x['group']] for x in d], s=18)
            c = results['mrs']['correlations'][mt]
            ax.set_title(f"{mt}\nρ={c['rho']:.2f}, n={c['n']}" if c['rho'] is not None else mt, fontsize=9)
            ax.set_xlabel('anomaly score in PRESS voxel')
        fig.tight_layout(); fig.savefig(FIG / 'mrs.png', dpi=110); plt.close(fig)

    syn = results.get('synthetic_lesions')
    if syn:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for name, c in [('model', '#c2410c'), ('B1_raw_z', '#1d5f86')]:
            ax.plot([d['amp_sd'] for d in syn[name]], [d['voxel_sensitivity'] for d in syn[name]], 'o-', color=c, label=name)
        ax.set_xlabel('lesion amplitude (× between-sham SD)'); ax.set_ylabel('voxel sensitivity at FDR 5%'); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / 'synthetic.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
