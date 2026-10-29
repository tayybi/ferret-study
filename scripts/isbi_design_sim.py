"""Experiment 5 (deviation D2): how many control sessions does SACC need? Gaussian-copula simulation fitted to supine shams."""
import json

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

import blast_stats as bs
import isbi_sacc as I

ICC = 0.18
S_GRID = [5, 10, 20, 40, 80]
A_GRID = [0, 1, 2, 3]
REPS = 400
PER_SESSION = 2


def fit(F, controls, strat):
    sup = [k for k in controls if strat[k] == 'SUBJ_POS_Supine']
    Z = F[sup].reshape(len(sup), -1)
    med = np.median(Z, 0); mad = 1.4826 * np.median(np.abs(Z - med), 0)
    Z = (Z - med) / np.maximum(mad, 1e-8)
    corr = LedoitWolf().fit(Z).covariance_
    d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
    pooled = np.sort(Z.ravel())
    return np.linalg.cholesky(corr + 1e-6 * np.eye(len(corr))), pooled, len(sup)


def sample(L, pooled, n_sess, per, rng):
    """Latent g = sqrt(ICC) u_session + sqrt(1-ICC) e_animal, mapped through pooled empirical marginals."""
    D = L.shape[0]
    u = rng.standard_normal((n_sess, D)) @ L.T
    e = rng.standard_normal((n_sess * per, D)) @ L.T
    g = np.sqrt(ICC) * np.repeat(u, per, 0) + np.sqrt(1 - ICC) * e
    q = stats.norm.cdf(g)
    return np.interp(q, np.linspace(0, 1, len(pooled)), pooled)


def main():
    rng = np.random.default_rng(2028)
    rows, sess, strat, feats, *_ = I.load()
    controls = [i for i, r in enumerate(rows) if r['group'] == 'Sham']
    L, pooled, n_fit = fit(feats['RAW'], controls, strat)
    C, R = feats['RAW'].shape[1:]
    sd_marg = pooled.std()
    out = {'n_fit_controls': n_fit, 'icc': ICC, 'm_regions': R, 'q': .10, 'reps': REPS, 'cells': []}
    for S in S_GRID:
        res = {a: {'hit': [], 'fdp': [], 'any': []} for a in A_GRID}
        for rep in range(REPS):
            Y = sample(L, pooled, S + 1, PER_SESSION, rng).reshape(-1, C, R)
            ses = [i // PER_SESSION for i in range(len(Y))]
            K = [i for i in range(len(Y)) if ses[i] < S]
            t = S * PER_SESSION  # first animal of the held-out (new) session
            cal, w = I.calibration(Y, K, ses, 'sacc', True)
            r = rng.integers(R)
            for a in A_GRID:
                ft = Y[t].copy(); ft[:, r] += a * sd_marg * I.DIRECTION
                p = I.pvalues(I.region_scores(Y[K], ft, True), cal, w)
                flag = bs.bh(p, .10)
                res[a]['any'].append(bool(flag.any()))
                res[a]['hit'].append(bool(flag[r]))
                res[a]['fdp'].append(float(flag[np.arange(R) != r].sum() / max(flag.sum(), 1)))
        for a in A_GRID:
            cell = {'S': S, 'amp': a, 'p_min_floor': 1 / (1 + S * R),
                    'sensitivity': float(np.mean(res[a]['hit'])) if a else None,
                    'any_flag_rate': float(np.mean(res[a]['any'])), 'mean_fdp': float(np.mean(res[a]['fdp']))}
            out['cells'].append(cell)
            print(cell, flush=True)
    (I.OUT / 'results_design_sim.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
