"""Cross-fitted 2.5D masked-autoencoder normative model; writes per-animal residual maps for one variant.

Follows blast-pipeline/ANALYSIS_PLAN.md. Never reads group labels except to build folds.
"""
import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion

from prepare_data import ROOT
from full_pipeline import atlas

BP = ROOT / 'ferret-study/blast-pipeline'
ALL = ['T2', 'FA', 'MD', 'AD', 'RD', 'logJ']
VARIANTS = {
    'main': dict(channels=ALL, fiskum=False),
    'fiskum_pool': dict(channels=ALL, fiskum=True),
    't2_only': dict(channels=['T2'], fiskum=False),
    'dti_only': dict(channels=['FA', 'MD', 'AD', 'RD'], fiskum=False),
    'no_logj': dict(channels=['T2', 'FA', 'MD', 'AD', 'RD'], fiskum=False),
    'main_sess': dict(channels=ALL, fiskum=False, session_folds=True),
}
PATCH = 8
PADX = 80


def load_cohort():
    rows = []
    for p in sorted((BP / 'sessions').glob('*/processing.json')):
        info = json.loads(p.read_text())
        if info.get('status') != 'ok':
            continue
        if info.get('template_correlation', 0) < .35:
            info['excluded'] = 'template_correlation<0.35'
            continue
        rows.append(info)
    data = [np.load(BP / 'sessions' / r['id'] / 'channels.npz') for r in rows]
    X = np.stack([np.stack([d[c].astype(np.float32) for c in ALL]) for d in data])
    cov = np.stack([d['coverage'] for d in data])
    tpl, tmask = atlas()
    brain = binary_erosion(tmask.numpy() > .5, iterations=1) & cov.all(0)
    brain[:, :, [0, -1]] = False
    for z in range(brain.shape[2]):
        if brain[:, :, z].sum() < 50:
            brain[:, :, z] = False  # thin edge slices are dropped from the analysis mask, not scored as zero
    return rows, np.nan_to_num(X), brain, tpl.numpy().astype(np.float32)


def scan_session(row):
    """Scan day as the session identifier (blast: scanner subject date; Fiskum: folder date)."""
    import re
    if row['cohort'] == 'blast':
        inv = {r['session']: r for r in json.loads((BP / 'inventory.json').read_text())}
        return inv[row['session']]['date'][:10]
    m = re.search(r'_(20\d{2})(\d{2})(\d{2})_', row['session'])
    return f'{m[1]}-{m[2]}-{m[3]}'


def session_folds(rows, k=4):
    """Assign whole scan sessions to folds, greedily balancing shams then animals, so no fold model
    is ever trained on a sham scanned in the same session as an animal it scores."""
    sess = {}
    for r in rows:
        sess.setdefault(scan_session(r), []).append(r)
    order = sorted(sess, key=lambda s: (-sum(r['group'] == 'Sham' for r in sess[s]), -len(sess[s]), s))
    load = [[0, 0] for _ in range(k)]
    fold = {}
    for s in order:
        f = min(range(k), key=lambda i: (load[i][0], load[i][1], i))
        for r in sess[s]:
            fold[r['id']] = f
        load[f][0] += sum(r['group'] == 'Sham' for r in sess[s]); load[f][1] += len(sess[s])
    return fold


def folds(rows, seed=2026, session=False):
    if session:
        return session_folds(rows)
    rng = np.random.default_rng(seed)
    fold = {}
    shams = [r for r in rows if r['cohort'] == 'blast' and r['group'] == 'Sham']
    for t in ['3d', '7d']:
        ids = [r['id'] for r in shams if r['time'] == t]
        rng.shuffle(ids)
        for i, sid in enumerate(ids):
            fold[sid] = (i + (0 if t == '3d' else 1)) % 4  # offset keeps fold sizes balanced (9+7 -> 4,4,4,4)
    for t in ['3d', '7d']:
        for g in ['10psi', '20psi']:
            ids = sorted(r['id'] for r in rows if r['cohort'] == 'blast' and r['group'] == g and r['time'] == t)
            rng.shuffle(ids)
            for i, sid in enumerate(ids):
                fold[sid] = i % 4
    fis = sorted(r['id'] for r in rows if r['cohort'] != 'blast')
    for i, sid in enumerate(fis):
        fold[sid] = i % 4
    return fold


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class MAE(nn.Module):
    def __init__(self, nch):
        super().__init__()
        cin = 3 * nch + 3
        self.e1, self.e2, self.e3 = Block(cin, 32), Block(32, 64), Block(64, 128)
        self.gamma = nn.Parameter(torch.ones(1, 128, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 128, 1, 1))
        self.d2, self.d1 = Block(128 + 64, 64), Block(64 + 32, 32)
        self.out = nn.Conv2d(32, nch, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2)) * self.gamma + self.beta
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.out(a)


class Slices:
    """Builds 2.5D inputs from standardized volumes (N, C, X, Y, Z) on the padded grid."""

    def __init__(self, X, brain, tpl, mean, sd):
        Xs = (X - mean[None, :, None, None, None]) / sd[None, :, None, None, None]
        Xs = np.clip(Xs, -6, 6) * brain[None, None]
        pad = PADX - X.shape[2]
        self.X = torch.from_numpy(np.pad(Xs, ((0, 0), (0, 0), (0, pad), (0, 0), (0, 0))).astype(np.float32))
        self.brain = torch.from_numpy(np.pad(brain, ((0, pad), (0, 0), (0, 0))))
        t = (tpl - tpl[brain].mean()) / tpl[brain].std()
        self.tpl = torch.from_numpy(np.pad(t * brain, ((0, pad), (0, 0), (0, 0))).astype(np.float32))
        self.zs = [z for z in range(1, X.shape[4] - 1) if brain[:, :, z].any()]
        self.Z = X.shape[4]
        self.nx = X.shape[2]

    def batch(self, idx, zs, masks):
        """idx animals, zs slice indices, masks (B, X, Y) bool hidden -> input, target, loss weight."""
        xs, tg = [], []
        for i, z in zip(idx, zs):
            v = self.X[i, :, :, :, z - 1:z + 2]  # C, X, Y, 3
            xs.append(v.permute(0, 3, 1, 2).reshape(-1, *v.shape[1:3]))
            tg.append(self.X[i, :, :, :, z])
        x, t = torch.stack(xs), torch.stack(tg)
        m = masks[:, None].float()
        bz = torch.stack([self.brain[:, :, z] for z in zs])[:, None].float()
        tp = torch.stack([self.tpl[:, :, z] for z in zs])[:, None]
        zc = torch.tensor([z / self.Z for z in zs], dtype=torch.float32)[:, None, None, None].expand_as(m)
        x = torch.cat([x * (1 - m), m, tp, zc], 1)
        return x, t, m * bz


def random_masks(n, rng, shape=(PADX, 40)):
    g = rng.random((n, shape[0] // PATCH, shape[1] // PATCH)) < .5
    return np.kron(g, np.ones((PATCH, PATCH), bool))


def train(ds, ids, nch_idx, args, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MAE(len(nch_idx)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.steps)
    for step in range(args.steps):
        idx = rng.choice(ids, args.batch)
        zs = rng.choice(ds.zs, args.batch)
        masks = torch.from_numpy(random_masks(args.batch, rng))
        x, t, w = ds.batch(idx, zs, masks)
        if rng.random() < .5:  # left-right flip augmentation
            x, t, w = x.flip(2), t.flip(2), w.flip(2)
        x, t, w = x.to(device), t.to(device), w.to(device)
        loss = ((model(x) - t).abs() * w).sum() / (w.sum() * t.shape[1]).clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model, float(loss)


@torch.no_grad()
def residuals(model, ds, i, nch_idx, device, seed, adapt_steps=0):
    """Mean absolute residual per voxel over 4 masks + complements (each voxel hidden 4 times)."""
    rng = np.random.default_rng(seed)
    base = random_masks(4, rng)
    masks = np.concatenate([base, ~base])
    if adapt_steps:
        model = adapt(model, ds, i, nch_idx, device, seed, adapt_steps)
    R = torch.zeros(len(nch_idx), *ds.X.shape[2:])
    for z in ds.zs:
        mk = torch.from_numpy(masks)
        x, t, _ = ds.batch([i] * len(masks), [z] * len(masks), mk)
        pred = model(x.to(device)).cpu()
        err = (pred - t).abs() * mk[:, None]
        R[:, :, :, z] = err.sum(0) / 4
    return R[:, :ds.nx].numpy()


def adapt(model, ds, i, nch_idx, device, seed, steps):
    """FiLM-only test-time adaptation on the animal's own masked reconstruction (ablation)."""
    import copy
    m = copy.deepcopy(model)
    rng = np.random.default_rng(seed + 7)
    opt = torch.optim.SGD([m.gamma, m.beta], lr=1e-2)
    with torch.enable_grad():
        for _ in range(steps):
            zs = rng.choice(ds.zs, 16)
            x, t, w = ds.batch([i] * 16, zs, torch.from_numpy(random_masks(16, rng)))
            x, t, w = x.to(device), t.to(device), w.to(device)
            loss = ((m(x) - t).abs() * w).sum() / (w.sum() * t.shape[1]).clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


AMPS = [1, 2, 3, 4]
SIGNS = {'T2': 1, 'FA': -1, 'MD': 1, 'AD': 1, 'RD': 1, 'logJ': 0}


def lesions(rows, X, brain, seed=4242):
    """One 2 mm-radius soft sphere per sham; returns lesion profiles (n_sham, X, Y, Z) and between-sham SD."""
    from scipy.ndimage import gaussian_filter
    sham = [i for i, r in enumerate(rows) if r['cohort'] == 'blast' and r['group'] == 'Sham']
    sd = X[sham].std(0)
    inner = binary_erosion(brain, iterations=4)
    cand = np.argwhere(inner)
    rng = np.random.default_rng(seed)
    grid = np.indices(brain.shape).transpose(1, 2, 3, 0)
    prof = []
    for _ in sham:
        c = cand[rng.integers(len(cand))]
        d = np.linalg.norm(grid - c, axis=-1)
        prof.append(np.clip(gaussian_filter((d <= 4).astype(np.float32), .7), 0, 1) * brain)
    return sham, np.stack(prof), sd


def synthetic(rows, X, brain, tpl, fold, nch_idx, args, out):
    """Insert lesions into held-out shams, rerun the held-out fold model; saves residuals and raw lesioned values."""
    sham, prof, sd = lesions(rows, X, brain)
    signs = np.array([SIGNS[ALL[c]] for c in range(len(ALL))], np.float32)
    res = np.zeros((len(args.seeds), len(sham), len(AMPS), len(nch_idx), int(brain.sum())), np.float32)
    raw = np.zeros((len(sham), len(AMPS), len(ALL), int(brain.sum())), np.float32)
    for j, i in enumerate(sham):
        for a, amp in enumerate(AMPS):
            raw[j, a] = (X[i] + amp * signs[:, None, None, None] * sd * prof[j][None])[:, brain]
    for si, seed in enumerate(args.seeds):
        for k in range(4):
            tr = [i for i, r in enumerate(rows) if fold[r['id']] != k and
                  ((r['cohort'] == 'blast' and r['group'] == 'Sham') or (VARIANTS[args.variant]['fiskum'] and r['cohort'] != 'blast'))]
            Xc = X[:, nch_idx]
            vals = Xc[tr][:, :, brain]
            mean, sd_c = vals.mean((0, 2)), vals.std((0, 2)) + 1e-6
            model = MAE(len(nch_idx)).to(args.device)
            model.load_state_dict(torch.load(BP / 'models' / args.variant / f'seed{seed}_fold{k}.pt', map_location=args.device))
            model.eval()
            for j, i in enumerate(sham):
                if fold[rows[i]['id']] != k:
                    continue
                for a, amp in enumerate(AMPS):
                    Xl = Xc.copy()
                    Xl[i] = Xc[i] + amp * signs[nch_idx][:, None, None, None] * sd[nch_idx] * prof[j][None]
                    ds = Slices(Xl[[i]], brain, tpl, mean, sd_c)
                    res[si, j, a] = residuals(model, ds, 0, nch_idx, args.device, 1000 + i)[:, brain]
        print('synthetic seed', seed, 'done', flush=True)
    np.savez_compressed(out / 'synthetic.npz', residuals=res.astype(np.float16), raw=raw.astype(np.float16),
                        lesion=prof[:, brain] > .5, sham_index=np.array(sham), amps=np.array(AMPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='main', choices=VARIANTS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--adapt-steps', type=int, default=0)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = ap.parse_args()
    cfg = VARIANTS[args.variant]
    name = args.variant + (f'_adapt{args.adapt_steps}' if args.adapt_steps else '')
    out = BP / 'models' / name
    out.mkdir(parents=True, exist_ok=True)
    rows, X, brain, tpl = load_cohort()
    fold = folds(rows, session=cfg.get('session_folds', False))
    ids = [r['id'] for r in rows]
    nch_idx = [ALL.index(c) for c in cfg['channels']]
    log = {'variant': name, 'config': cfg, 'args': vars(args), 'n_sessions': len(rows), 'fold': fold, 'runs': []}
    for seed in args.seeds:
        Rall = np.zeros((len(rows), len(nch_idx), int(brain.sum())), np.float32)
        for k in range(4):
            t0 = time.time()
            tr = [i for i, r in enumerate(rows) if fold[r['id']] != k and
                  ((r['cohort'] == 'blast' and r['group'] == 'Sham') or (cfg['fiskum'] and r['cohort'] != 'blast'))]
            # Standardization statistics from this fold's training animals only.
            Xc = X[:, nch_idx]
            vals = Xc[tr][:, :, brain]
            mean, sd = vals.mean((0, 2)), vals.std((0, 2)) + 1e-6
            ds = Slices(Xc, brain, tpl, mean, sd)
            ckpt = BP / 'models' / args.variant / f'seed{seed}_fold{k}.pt'
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            model = MAE(len(nch_idx)).to(args.device)
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=args.device)); last = None
            else:
                model, last = train(ds, tr, nch_idx, args, seed * 10 + k, args.device)
                torch.save(model.state_dict(), ckpt)
            model.eval()
            scored = [i for i, r in enumerate(rows) if fold[r['id']] == k]
            for i in scored:
                R = residuals(model, ds, i, nch_idx, args.device, 1000 + i, args.adapt_steps)
                Rall[i] = R[:, brain]
            log['runs'].append({'seed': seed, 'fold': k, 'train_n': len(tr), 'scored_n': len(scored),
                                'final_train_loss': last, 'seconds': time.time() - t0})
            print(name, 'seed', seed, 'fold', k, 'train', len(tr), 'scored', len(scored), 'loss', last,
                  f'{time.time() - t0:.0f}s', flush=True)
        np.save(out / f'residuals_seed{seed}.npy', Rall.astype(np.float16))
    (out / 'log.json').write_text(json.dumps(log, indent=1) + '\n')
    if args.synthetic:
        synthetic(rows, X, brain, tpl, fold, nch_idx, args, out)
    np.save(BP / 'models' / 'brain_mask.npy', brain)
    (BP / 'models' / 'rows.json').write_text(json.dumps([{**{k: r.get(k) for k in ['id', 'cohort', 'group', 'time', 'animal', 'session']},
                                                          'scan_session': scan_session(r), 'fold_' + name: fold[r['id']]}
                                                         for r in rows], indent=1) + '\n')


if __name__ == '__main__':
    main()
