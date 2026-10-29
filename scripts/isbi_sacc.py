"""Session-aware conformal calibration (SACC) experiments for ISBI 2027; follows isbi2027/ANALYSIS_PLAN_ISBI.md."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ferret-mpl')
import hashlib
import json

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import blast_stats as bs
from blast_explore import atlas_labels
from blast_model import scan_session
from prepare_data import ROOT

OUT = ROOT / 'ferret-study/isbi2027'
FIGS = OUT / 'figures'
ALL = bs.ALL
DIRECTION = np.array([1, -1, 1, 1, 1, 0], np.float32)  # T2, FA, MD, AD, RD, logJ
AMPS = [.5, 1, 1.5, 2, 3]
ALPHAS = [.01, .05, .10]
MODES = ['naive', 'sacc', 'sacc_m']
LABEL = {'naive': 'naive', 'sacc': 'SACC', 'sacc_m': 'SACC+M'}


def check_lock():
    want = (OUT / 'plan.sha256').read_text().split()[0]
    if hashlib.sha256((OUT / 'ANALYSIS_PLAN_ISBI.md').read_bytes()).hexdigest() != want:
        raise SystemExit('ISBI analysis plan changed after locking')


def region_scores(ref, ft, absolute):
    """ref (m, C, R) reference controls, ft (C, R) -> max-over-channel robust z per region (R,)."""
    med = np.median(ref, 0)
    mad = 1.4826 * np.median(np.abs(ref - med), 0)
    floor = np.percentile(mad, 10, axis=1, keepdims=True)
    z = (ft - med) / np.maximum(mad, np.maximum(floor, 1e-8))
    return (np.abs(z) if absolute else z).max(0)


def calibration(F, K, sess, mode, absolute):
    """Calibration scores (|K|, R) and weights for controls K. naive: leave-one-animal-out reference;
    sacc: leave-own-session-out reference, each session carrying equal weight."""
    cal, w = [], []
    for k in K:
        ref = [j for j in K if j != k] if mode == 'naive' else [j for j in K if sess[j] != sess[k]]
        if len(ref) < 2:
            ref = [j for j in K if j != k]
        cal.append(region_scores(F[ref], F[k], absolute))
        w.append(1.0 if mode == 'naive' else 1.0 / sum(sess[j] == sess[k] for j in K))
    return np.array(cal), np.array(w)


def calib_set(t, controls, sess, strat, mode, exclude_session=True):
    """Calibration controls for test animal t. naive/sacc exclude t's session (deployment to a new session);
    sacc_m additionally keeps only controls in t's acquisition stratum (deviation D1)."""
    K = [k for k in controls if (sess[k] != sess[t] if exclude_session else k != t)]
    if mode == 'sacc_m':
        K = [k for k in K if strat[k] == strat[t]]
    return K if len({sess[k] for k in K}) >= 2 else None


def pvalues(st, cal, w):
    """Weighted conformal p-values pooling calibration (animal or session) x region units."""
    vals = cal.ravel()
    ww = np.repeat(w, cal.shape[1])
    o = np.argsort(vals)
    vs, cw = vals[o], np.cumsum(ww[o][::-1])[::-1]
    idx = np.searchsorted(vs, st, side='left')
    above = np.where(idx < len(vs), cw[np.minimum(idx, len(vs) - 1)], 0.)
    return (1 + above) / (1 + ww.sum())


def load():
    rows = json.loads((bs.MD / 'rows.json').read_text())
    keep = [i for i, r in enumerate(rows) if r['cohort'] == 'blast']
    brain = np.load(bs.MD / 'brain_mask.npy')
    lab = atlas_labels(brain)[brain]
    regions = [l for l in np.unique(lab) if l > 0 and (lab == l).sum() >= 200]
    X = np.stack([np.stack([np.load(bs.BP / 'sessions' / rows[i]['id'] / 'channels.npz')[c].astype(np.float32)[brain] for c in ALL])
                  for i in keep])

    def regmean(A):
        return np.stack([A[:, :, lab == l].mean(2) for l in regions], 2)
    feats = {'RAW': regmean(X)}
    for name, var in [('RES-A', 'main'), ('RES-S', 'main_sess')]:
        if (bs.MD / var / 'residuals_seed2.npy').exists():
            R, _ = bs.load_residuals(var)
            feats[name] = regmean(R[keep])
    rws = [rows[i] for i in keep]
    sess = [r.get('scan_session') or scan_session(r) for r in rws]
    from prepare_data import parameters
    strat = [parameters(ROOT / r['session'] / 'subject').get('SUBJECT_position') for r in rws]
    return rws, sess, strat, feats, X, brain, lab, regions


def icc_session(F, rows, sess, rng, nperm=2000):
    cell = np.array([f"{r['time']}-{r['group']}" for r in rows])
    Y = F.reshape(len(F), -1).copy()
    for c in np.unique(cell):
        Y[cell == c] -= Y[cell == c].mean(0)
    Y /= Y.std(0, ddof=1) + 1e-12

    def icc(labels):
        u = np.unique(labels); n = len(labels); k = len(u)
        sizes = np.array([(labels == g).sum() for g in u])
        means = np.stack([Y[labels == g].mean(0) for g in u])
        grand = Y.mean(0)
        msb = (sizes[:, None] * (means - grand) ** 2).sum(0) / (k - 1)
        msw = sum(((Y[labels == g] - means[i]) ** 2).sum(0) for i, g in enumerate(u)) / (n - k)
        n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
        return np.median((msb - msw) / (msb + (n0 - 1) * msw))
    s = np.array(sess)
    obs = icc(s)
    null = []
    for _ in range(nperm):
        p = s.copy()
        for c in np.unique(cell):
            ix = np.nonzero(cell == c)[0]; p[ix] = s[rng.permutation(ix)]
        null.append(icc(p))
    return {'median_icc': float(obs), 'null_mean': float(np.mean(null)), 'p_perm': float((np.sum(np.array(null) >= obs) + 1) / (nperm + 1))}


def new_session_null(F, controls, sess, strat, absolute):
    """Experiment 2: hold out whole sham sessions; compare naive vs SACC (vs SACC+M) on held-out shams."""
    res = {m: {'p': [], 'any_flag_q10': [], 'any_flag_q05': [], 'not_calibratable': 0} for m in MODES}
    for s in sorted({sess[k] for k in controls}):
        test = [k for k in controls if sess[k] == s]
        for m in res:
            K = calib_set(test[0], controls, sess, strat, m)
            if K is None:
                res[m]['not_calibratable'] += len(test); continue
            cal, w = calibration(F, K, sess, m, absolute)
            for t in test:
                p = pvalues(region_scores(F[K], F[t], absolute), cal, w)
                res[m]['p'].extend(p.tolist())
                res[m]['any_flag_q10'].append(bool(bs.bh(p, .10).any()))
                res[m]['any_flag_q05'].append(bool(bs.bh(p, .05).any()))
    out = {}
    for m, d in res.items():
        p = np.array(d['p'])
        out[m] = {f'P(p<={a})': float((p <= a).mean()) for a in ALPHAS}
        out[m].update({'false_alarm_rate_q10': float(np.mean(d['any_flag_q10'])),
                       'false_alarm_rate_q05': float(np.mean(d['any_flag_q05'])), 'n_test_shams': len(d['any_flag_q10']),
                       'n_pvalues': int(len(p)), 'ks_uniform_p': float(stats.kstest(p, 'uniform').pvalue),
                       'not_calibratable': d['not_calibratable']})
        out[m]['_p'] = p
    return out


def power(F, controls, sess, strat, rng, absolute=True):
    """Experiment 3: diffuse single-region shifts injected into held-out shams (RAW)."""
    sigma = F[controls].std(0, ddof=1)  # (C, R)
    R = F.shape[2]
    out = {m: {a: {'hit': [], 'fdp': []} for a in AMPS} for m in MODES}
    cache, Ks = {}, {}
    for t in controls:
        regs = rng.choice(R, 3, replace=False)  # drawn first so all modes share the same injections
        for m in out:
            key = (sess[t], m)
            if key not in cache:
                Ks[key] = calib_set(t, controls, sess, strat, m)
                cache[key] = calibration(F, Ks[key], sess, m, absolute) if Ks[key] else None
        for r in regs:
            for a in AMPS:
                ft = F[t].copy(); ft[:, r] += a * sigma[:, r] * DIRECTION
                for m in out:
                    if cache[(sess[t], m)] is None:
                        continue
                    st = region_scores(F[Ks[(sess[t], m)]], ft, absolute)
                    p = pvalues(st, *cache[(sess[t], m)])
                    flag = bs.bh(p, .10)
                    out[m][a]['hit'].append(bool(flag[r]))
                    out[m][a]['fdp'].append(float(flag[np.arange(R) != r].sum() / max(flag.sum(), 1)))
    return {m: [{'amp': a, 'sensitivity_q10': float(np.mean(v['hit'])), 'fdr_q10': float(np.mean(v['fdp'])),
                 'n': len(v['hit'])} for a, v in d.items()] for m, d in out.items()}


def voxel_comparator(X, controls, lab, regions, rng, brain):
    """Earlier voxel-level pooled BH method on the same kind of diffuse regional shift (RAW, a in 1,2,3)."""
    vol = bs.Vol(brain)
    sig_reg = np.stack([X[controls][:, :, lab == l].mean(2) for l in regions], 2).std(0, ddof=1)
    out = []
    for a in [1, 2, 3]:
        hits = []
        for j, t in enumerate(controls):
            r = rng.integers(len(regions))
            Xs = X[controls].copy()
            m = lab == regions[r]
            Xs[j][:, m] += (a * sig_reg[:, r] * DIRECTION)[:, None]
            sham = list(range(len(controls)))
            S = bs.zscores(Xs, sham, vol, absolute=True)
            pool = np.sort(np.concatenate([S[k] for k in sham if k != j]))
            p = (1 + len(pool) - np.searchsorted(pool, S[j], side='left')) / (len(pool) + 1)
            flag = bs.bh(p, .10)
            hits.append(bool(flag[m].mean() >= .10))
        out.append({'amp': a, 'region_detected_q10': float(np.mean(hits)), 'n': len(hits)})
        print('voxel comparator', out[-1], flush=True)
    return out


def application(F, rows, sess, strat, controls, absolute, mode):
    blast = [i for i, r in enumerate(rows) if r['group'] != 'Sham']
    res = {}
    cache = {}
    for t in blast + controls:
        K = calib_set(t, controls, sess, strat, mode)
        if K is None:
            continue
        key = tuple(K)
        if key not in cache:
            cache[key] = calibration(F, K, sess, mode, absolute)
        p = pvalues(region_scores(F[K], F[t], absolute), *cache[key])
        res[rows[t]['id']] = {'n_flag_q10': int(bs.bh(p, .10).sum()), 'min_p': float(p.min()),
                              'group': rows[t]['group'], 'time': rows[t]['time'], 'p': p.tolist()}
    summ = {}
    for tm in ['3d', '7d']:
        for g in ['Sham', '10psi', '20psi']:
            v = [d for d in res.values() if d['group'] == g and d['time'] == tm]
            if not v:
                continue
            summ[f'{tm}-{g}'] = {'n': len(v), 'share_any_flag': float(np.mean([d['n_flag_q10'] > 0 for d in v])),
                                 'mean_flagged_regions': float(np.mean([d['n_flag_q10'] for d in v]))}
    return res, summ


def main():
    check_lock()
    FIGS.mkdir(exist_ok=True)
    rng = np.random.default_rng(2027)
    rows, sess, strat, feats, X, brain, lab, regions = load()
    if hashlib.sha256((OUT / 'DEVIATIONS.md').read_bytes()).hexdigest() != (OUT / 'deviations.sha256').read_text().split()[0]:
        raise SystemExit('DEVIATIONS.md changed after logging')
    controls = [i for i, r in enumerate(rows) if r['group'] == 'Sham']
    results = {'n_animals': len(rows), 'n_controls': len(controls), 'n_sessions': len(set(sess)),
               'n_control_sessions': len({sess[k] for k in controls}), 'n_regions': len(regions),
               'regions': [int(r) for r in regions], 'features': list(feats)}
    print(json.dumps(results), flush=True)
    results['exp1_session_icc_RAW'] = icc_session(feats['RAW'], rows, sess, rng)
    print('ICC', results['exp1_session_icc_RAW'], flush=True)

    results['exp2_new_session_null'] = {}
    curves = {}
    for name, F in feats.items():
        r = new_session_null(F, controls, sess, strat, absolute=(name == 'RAW'))
        curves[name] = {m: r[m].pop('_p') for m in r}
        results['exp2_new_session_null'][name] = r
        print('EXP2', name, json.dumps(r), flush=True)

    results['exp3_power_RAW'] = power(feats['RAW'], controls, sess, strat, rng)
    results['strata'] = {str(k): int(sum(x == k for x in strat)) for k in set(strat)}
    print('EXP3', json.dumps(results['exp3_power_RAW']), flush=True)
    results['exp3_voxel_comparator_RAW'] = voxel_comparator(X, controls, lab, regions, rng, brain)

    results['exp4_application'] = {}
    for name in [n for n in ['RAW', 'RES-S'] if n in feats]:
        for mode in ['sacc', 'sacc_m']:
            per, summ = application(feats[name], rows, sess, strat, controls, (name == 'RAW'), mode)
            key = f'{name}/{mode}'
            results['exp4_application'][key] = {'summary': summ}
            print('EXP4', key, json.dumps(summ), flush=True)
            (OUT / f'per_animal_{name}_{mode}.json').write_text(json.dumps(per, indent=1) + '\n')
            if name == 'RAW':
                results['exp4_application'][key]['mrs'] = mrs_link(per, rows, regions, lab, brain)
    (OUT / 'results_isbi.json').write_text(json.dumps(results, indent=1, default=float) + '\n')
    np.savez(OUT / 'validity_pvalues.npz', **{f'{n}__{m}': v for n, d in curves.items() for m, v in d.items()})


def mrs_link(per, rows, regions, lab, brain):
    out = []
    for r in rows:
        proc = json.loads((bs.BP / 'sessions' / r['id'] / 'processing.json').read_text())
        m = proc.get('mrs')
        if not m or not (m.get('FWHM_ppm', 9) <= .08 and m.get('SNR', 0) >= 5) or (m.get('NAA+NAAG %SD') or 99) > 20:
            continue
        vm = np.load(bs.BP / 'sessions' / r['id'] / 'channels.npz')['mrs_flip_xy'][brain]
        if vm.sum() < 5:
            continue
        labs = lab[vm]
        best = max(regions, key=lambda l: (labs == l).sum())
        if r['id'] not in per:
            continue
        out.append((-np.log10(per[r['id']]['p'][regions.index(best)]), m['NAA+NAAG/Cr+PCr'], int(best)))
    if len(out) < 8:
        return {'n': len(out)}
    x, y, l = zip(*out)
    rho, p = bs.spearman_perm(np.array(x), np.array(y), np.random.default_rng(5))
    return {'n': len(out), 'rho_neglog10p_vs_NAA': rho, 'p_perm': p, 'regions_used': sorted(set(l))}


def figures(results, curves):
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax = axs[0]
    grid = np.linspace(0, .2, 41)
    for name, ls in [('RAW', '-'), ('RES-S', '--'), ('RES-A', ':')]:
        if name not in curves: continue
        for m, col in [('naive', '#c2410c'), ('sacc', '#1d5f86'), ('sacc_m', '#2a9d8f')]:
            p = curves[name][m]
            ax.plot(grid, [(p <= g).mean() for g in grid], ls, color=col, lw=1.4, label=f'{name} {LABEL[m]}')
    ax.plot(grid, grid, color='0.5', lw=.8)
    ax.set_xlabel('nominal α'); ax.set_ylabel('P(p ≤ α), held-out sessions'); ax.legend(fontsize=6); ax.set_title('(a) Validity', fontsize=9)
    ax = axs[1]
    for m, col in [('naive', '#c2410c'), ('sacc', '#1d5f86'), ('sacc_m', '#2a9d8f')]:
        d = results['exp3_power_RAW'][m]
        ax.plot([x['amp'] for x in d], [x['sensitivity_q10'] for x in d], 'o-', color=col, label=LABEL[m])
    vc = results['exp3_voxel_comparator_RAW']
    ax.plot([x['amp'] for x in vc], [x['region_detected_q10'] for x in vc], 's:', color='0.4', label='voxel BH')
    ax.set_xlabel('regional shift (× between-sham SD)'); ax.set_ylabel('sensitivity (BH q=0.10)'); ax.legend(fontsize=7); ax.set_title('(b) Power', fontsize=9)
    ax = axs[2]
    summ = results['exp4_application']['RAW/sacc_m']['summary']
    keys = [k for k in ['3d-Sham', '3d-10psi', '3d-20psi', '7d-Sham', '7d-10psi', '7d-20psi'] if k in summ]
    ax.bar(range(len(keys)), [summ[k]['share_any_flag'] for k in keys], color=['#1d5f86', '#e0904c', '#c2410c'] * 2)
    ax.set_xticks(range(len(keys)), [k.replace('-', '\n') for k in keys], fontsize=7); ax.set_ylabel('animals with ≥1 flagged region')
    ax.set_title('(c) Blast cohort, SACC+M RAW', fontsize=9)
    fig.tight_layout(); fig.savefig(FIGS / 'fig_results.pdf'); fig.savefig(FIGS / 'fig_results.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    main()
