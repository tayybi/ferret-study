"""Per-session preprocessing for the normative-mapping study: T2 + DTI (+ PRESS voxel) into ferret template space.

Never edits sources. Outputs are compact per-session files under blast-pipeline/sessions/<id>/.
Limitations recorded in each processing.json: no FSL topup/eddy; distortion correction is a
reverse-PE halfway SyN restricted to the phase-encode axis; motion is rigid per volume; no bvec rotation.
"""
import os
os.environ.setdefault('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', '3')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ferret-mpl')
import argparse
import json
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ants
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare_data import ROOT, decode, parameters
from full_pipeline import atlas, normalized, provisional_mask, scan_image, digest

BP = ROOT / 'ferret-study/blast-pipeline'
CHANNELS = ['T2', 'FA', 'MD', 'AD', 'RD', 'logJ']


def template():
    """Atlas template/mask on the 0.5 mm analysis grid, in the scanner axis convention used by full_pipeline."""
    return atlas()


def dwi_volumes(path):
    """Return (reps, diffusion, x, y, z) float32 array, image geometry kwargs, bvals and unit gradient directions."""
    a, p = decode(path / 'pdata/1')
    m = parameters(path / 'method')
    order = p['VisuFGOrderDesc']
    if not re.search(r'FG_SLICE.*FG_DIFFUSION.*FG_CYCLE', order.replace('\n', ' ')):
        raise ValueError('Unexpected DTI frame order: ' + order)
    nslice = int(re.search(r'\((\d+), <FG_SLICE>', order)[1])
    ndiff = int(re.search(r'\((\d+), <FG_DIFFUSION>', order)[1])
    nrep = a.shape[0] // (nslice * ndiff)
    a = a.reshape(nrep, ndiff, nslice, *a.shape[1:]).transpose(0, 1, 4, 3, 2)  # -> rep, diff, x, y, z
    orientation = np.array(p['VisuCoreOrientation'].split(), float).reshape(-1, 3, 3)
    if not np.allclose(orientation, np.eye(3)[None], atol=.001):
        raise ValueError('Non-identity DTI orientation')
    pos = np.array(p['VisuCorePosition'].split(), float).reshape(-1, 3)[:nslice]
    dz = float(np.median(np.diff(pos[:, 2])))
    ext = [float(x) for x in p['VisuCoreExtent'].split()]
    geom = dict(origin=tuple(pos[0]), spacing=(ext[0] / a.shape[2], ext[1] / a.shape[3], dz))
    b = np.array(m['PVM_DwEffBval'].split(), float)
    g = np.array(m['PVM_DwGradVec'].split(), float).reshape(-1, 3)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    g = np.divide(g, n, out=np.zeros_like(g), where=n > 0)
    return a.astype(np.float32), geom, b, g


def img(a, geom):
    return ants.from_numpy(np.ascontiguousarray(a, dtype=np.float32), **geom)


def fit_tensor(signal, bvals, bvecs, mask):
    """Weighted least-squares log-linear tensor fit. signal: (n, x, y, z). Returns FA, MD, AD, RD (um^2/ms)."""
    x, y, z = bvecs.T
    B = np.stack([np.ones_like(bvals), -bvals * x * x, -bvals * y * y, -bvals * z * z,
                  -2 * bvals * x * y, -2 * bvals * x * z, -2 * bvals * y * z], 1)
    S = np.clip(signal[:, mask], 1e-3, None)
    L = np.log(S)
    coef = np.linalg.lstsq(B, L, rcond=None)[0]
    W = np.exp(B @ coef) ** 2  # WLS weights from OLS prediction
    out = np.empty((7, S.shape[1]))
    for i in range(0, S.shape[1], 4096):
        w = W[:, i:i + 4096]
        Bw = B[None] * w.T[:, :, None]
        lhs = np.einsum('vnk,nl->vkl', Bw, B)
        rhs = np.einsum('vnk,nv->vk', Bw, L[:, i:i + 4096])
        out[:, i:i + 4096] = np.linalg.solve(lhs + 1e-9 * np.eye(7), rhs).T
    d = out[1:]
    D = np.stack([np.stack([d[0], d[3], d[4]], -1), np.stack([d[3], d[1], d[5]], -1),
                  np.stack([d[4], d[5], d[2]], -1)], -2)
    ev = np.clip(np.linalg.eigvalsh(D), 0, 4e-3)[:, ::-1]
    md = ev.mean(1)
    fa = np.sqrt(1.5 * ((ev - md[:, None]) ** 2).sum(1) / np.clip((ev ** 2).sum(1), 1e-12, None))
    maps = {}
    for name, v in [('FA', fa), ('MD', md * 1e3), ('AD', ev[:, 0] * 1e3), ('RD', ev[:, 1:].mean(1) * 1e3)]:
        full = np.zeros(mask.shape, np.float32)
        full[mask] = v
        maps[name] = full
    return maps


def voxel_box(t2, position, size, orient, sign):
    """PRESS voxel as a binary image on the T2 grid. sign flips the stored position convention for QC."""
    R = np.array(orient, float).reshape(3, 3)
    c = np.array(position, float) * np.array(sign, float)
    idx = np.indices(t2.shape).reshape(3, -1).T
    pts = np.array(t2.origin) + idx * np.array(t2.spacing)
    local = (pts - c) @ R.T
    inside = np.all(np.abs(local) <= np.array(size) / 2, axis=1)
    return ants.from_numpy(inside.reshape(t2.shape).astype(np.float32), origin=t2.origin, spacing=t2.spacing,
                           direction=t2.direction)


def lcmodel(scan_dir):
    """Pick one LCModel result per scan (v6 preferred, else ouput, else any) and parse csv + table QC."""
    csvs = {p.stem: p for p in scan_dir.glob('*.csv')}
    if not csvs:
        return None
    key = 'v6' if 'v6' in csvs else 'ouput' if 'ouput' in csvs else sorted(csvs)[0]
    lines = csvs[key].read_text(errors='replace').splitlines()
    head = [h.strip() for h in lines[0].split(',')]
    vals = [v.strip() for v in lines[1].split(',')]
    rec = {'variant': key, 'variants_present': sorted(csvs)}
    for h, v in zip(head, vals):
        try:
            rec[h] = float(v)
        except ValueError:
            pass
    table = scan_dir / f'{key}.table'
    if table.exists():
        t = table.read_text(errors='replace')
        m = re.search(r'FWHM\s*=\s*([\d.]+)\s*ppm\s*S/N\s*=\s*([\d.]+)', t)
        if m:
            rec['FWHM_ppm'], rec['SNR'] = float(m[1]), float(m[2])
    return rec


def process(row):
    sid = row['id']
    out = BP / 'sessions' / sid
    if (out / 'channels.npz').exists():
        return sid, 'cached'
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    session = ROOT / row['session']
    info = {'id': sid, **{k: row[k] for k in ['session', 'cohort', 'group', 'time', 'animal']}}
    try:
        tpl, tpl_mask = template()
        # --- T2: decode on the existing 160-grid convention, N4, provisional atlas mask
        t2_path = session / row['scans']['t2'][-1] / 'pdata/1'
        t2, _ = scan_image(t2_path)
        t2 = ants.n4_bias_field_correction(t2, shrink_factor=2)
        mask, volume = provisional_mask(t2, tpl, tpl_mask, out)
        info.update(t2_source=str(t2_path.relative_to(ROOT)), t2_sha256=digest(t2_path / '2dseq'), mask_volume_mm3=volume)
        # --- T2 -> template: invert atlas rigid as initialisation, then affine + SyN on brain-extracted images
        rigid = ants.read_transform(str(out / 'atlas_0GenericAffine.mat'))
        ants.write_transform(rigid.invert(), str(out / 'init_t2_to_tpl.mat'))
        t2n = normalized(t2, mask)
        brain = t2n * ants.morphology(mask, 'dilate', 1)
        areg = ants.registration(tpl * tpl_mask, brain, 'Affine', initial_transform=str(out / 'init_t2_to_tpl.mat'),
                                 outprefix=str(out / 'tplaff_'), aff_metric='mattes', aff_iterations=(200, 100, 50),
                                 aff_shrink_factors=(4, 2, 1), aff_smoothing_sigmas=(2, 1, 0), random_seed=42)
        reg = ants.registration(tpl * tpl_mask, brain, 'SyNOnly', initial_transform=areg['fwdtransforms'][0],
                                outprefix=str(out / 'tpl_'), syn_metric='CC', syn_sampling=2,
                                reg_iterations=(40, 20, 10), grad_step=0.2, random_seed=42)
        t2_tpl = ants.apply_transforms(tpl, t2n, reg['fwdtransforms'], interpolator='linear')
        cover = ants.apply_transforms(tpl, ants.from_numpy(np.ones(t2.shape, np.float32), origin=t2.origin,
                                      spacing=t2.spacing, direction=t2.direction), reg['fwdtransforms'],
                                      interpolator='nearestNeighbor')
        logj = ants.create_jacobian_determinant_image(tpl, reg['fwdtransforms'][0], do_log=True)
        sim = float(np.corrcoef(t2_tpl.numpy()[tpl_mask.numpy() > .5], tpl.numpy()[tpl_mask.numpy() > .5])[0, 1])
        info['template_correlation'] = sim
        info['t_template'] = time.time() - t0
        # --- DTI
        dti_dir = session / row['scans']['dti'][-1]
        dwi, geom, bvals, bvecs = dwi_volumes(dti_dir)
        info['dti_source'] = str(dti_dir.relative_to(ROOT))
        b0i = bvals < 100
        fwd_b0 = img(dwi[:, b0i].mean((0, 1)), geom)
        sdc_field = None
        if row['scans'].get('dti_neg'):
            neg, ngeom, nb, _ = dwi_volumes(session / row['scans']['dti_neg'][-1])
            # neg_DtiEpi stores the reverse-PE image flipped along the phase-encode (y) axis; verified on 4 sessions
            # (fwd/rev b0 correlation ~0.0 as stored vs 0.82-0.84 after flipping y).
            rev_b0 = img(neg[:, nb < 100].mean((0, 1))[:, ::-1, :], ngeom)
            fb, rb = normalized(fwd_b0), normalized(rev_b0)
            before = float(np.corrcoef(fb.numpy().ravel(), rb.numpy().ravel())[0, 1])
            sreg = ants.registration(fb, rb, 'SyNOnly', restrict_transformation=(0, 1, 0), syn_metric='CC',
                                     syn_sampling=2, grad_step=0.1, flow_sigma=4, total_sigma=0,
                                     reg_iterations=(60, 40, 20), outprefix=str(out / 'sdc_'), random_seed=42)
            warp = [t for t in sreg['fwdtransforms'] if t.endswith('Warp.nii.gz')][0]
            u = nib.load(warp)
            half = nib.Nifti1Image(-0.5 * u.get_fdata(dtype=np.float32), u.affine, u.header)
            sdc_field = str(out / 'sdc_half.nii.gz')
            nib.save(half, sdc_field)
            after = float(np.corrcoef(fb.numpy().ravel(), sreg['warpedmovout'].numpy().ravel())[0, 1])
            info['sdc'] = {'fwd_rev_corr_before': before, 'fwd_rev_corr_after_full_warp': after,
                           'max_shift_mm': float(np.abs(u.get_fdata()).max())}
        info['t_sdc'] = time.time() - t0
        # Motion: each volume -> mean of its own b-shell (same contrast); shell means -> b0 mean; then half-way SDC.
        # Transforms exceeding 3 mm / ~6 degrees are treated as registration failures and replaced by identity.
        shells = {'b0': bvals < 100, 'b1000': (bvals >= 500) & (bvals < 1500), 'b2000': bvals >= 1500}
        ref = fwd_b0
        corrected = np.zeros_like(dwi)
        to_b0, fallbacks, motion = {}, 0, []

        def rigid(fixed, moving):
            reg_ = ants.registration(fixed, moving, 'Rigid', aff_metric='mattes', aff_iterations=(100, 50, 20),
                                     aff_shrink_factors=(4, 2, 1), aff_smoothing_sigmas=(2, 1, 0),
                                     random_seed=42, verbose=False)
            par = np.array(ants.read_transform(reg_['fwdtransforms'][0]).parameters)
            trans, rot = float(np.linalg.norm(par[9:12])), float(np.abs(par[:9].reshape(3, 3) - np.eye(3)).max())
            return reg_['fwdtransforms'], trans, rot

        for name, sel in shells.items():
            sref = img(dwi[:, sel].mean((0, 1)), geom)
            if name == 'b0':
                to_b0[name] = []
            else:
                tl_, tr_, ro_ = rigid(ref, sref)
                ok_ = tr_ <= 3 and ro_ <= .1
                to_b0[name] = tl_ if ok_ else []
                fallbacks += not ok_
            for k in np.nonzero(sel)[0]:
                for r in range(dwi.shape[0]):
                    v = img(dwi[r, k], geom)
                    tl_, tr_, ro_ = rigid(sref, v)
                    ok_ = tr_ <= 3 and ro_ <= .1
                    fallbacks += not ok_
                    motion.append(tr_ if ok_ else 0.)
                    tl = ([sdc_field] if sdc_field else []) + to_b0[name] + (tl_ if ok_ else [])
                    corrected[r, k] = ants.apply_transforms(ref, v, tl, interpolator='linear').numpy() if tl else dwi[r, k]
        info['t_motion'] = time.time() - t0
        info['motion'] = {'max_translation_mm': float(np.max(motion)), 'median_translation_mm': float(np.median(motion)),
                          'fallbacks_to_identity': int(fallbacks)}
        sig = corrected.mean(0)
        b0c = img(sig[b0i].mean(0), geom)
        # b0 -> T2 rigid (same physical frame at acquisition; corrects inter-scan motion)
        dmask_img = ants.resample_image_to_target(mask, b0c, interp_type='nearestNeighbor')
        breg = ants.registration(t2n, normalized(b0c), 'Rigid', aff_metric='mattes', outprefix=str(out / 'b0t2_'),
                                 random_seed=42, verbose=False)
        dmask = binary_dilation(dmask_img.numpy() > .5, iterations=2)
        maps = fit_tensor(sig, bvals, bvecs, dmask)
        chans = {'T2': t2_tpl.numpy(), 'logJ': logj.numpy()}
        for name, a in maps.items():
            chans[name] = ants.apply_transforms(tpl, img(a, geom), reg['fwdtransforms'] + breg['fwdtransforms'],
                                                interpolator='linear').numpy()
        dcover = ants.apply_transforms(tpl, img(np.ones(b0c.shape), geom), reg['fwdtransforms'] + breg['fwdtransforms'],
                                       interpolator='nearestNeighbor').numpy()
        coverage = (cover.numpy() > .5) & (dcover > .5)
        # --- PRESS voxel -> template (sign convention chosen later from cohort-wide QC; store both)
        vox = {}
        if row.get('mrs_voxel') and row['scans'].get('mrs'):
            mv = row['mrs_voxel']
            pos = [float(x) for x in mv['PVM_VoxArrPosition'].split()]
            size = [float(x) for x in mv['PVM_VoxArrSize'].split()]
            orient = [float(x) for x in mv['PVM_VoxArrGradOrient'].split()]
            for name, sign in [('as_stored', (1, 1, 1)), ('flip_xy', (-1, -1, 1))]:
                box = voxel_box(t2, pos, size, orient, sign)
                native_in_brain = float((box.numpy() * (mask.numpy() > .5)).sum() / max(box.numpy().sum(), 1))
                vox[name] = {'native_voxels': float(box.numpy().sum()), 'fraction_in_brain': native_in_brain,
                             'tpl': ants.apply_transforms(tpl, box, reg['fwdtransforms'], interpolator='linear').numpy() > .5}
            info['mrs'] = lcmodel(session / row['scans']['mrs'][-1])
            info['mrs_voxel_qc'] = {k: {kk: vv for kk, vv in v.items() if kk != 'tpl'} for k, v in vox.items()}
        np.savez_compressed(out / 'channels.npz', **{k: chans[k].astype(np.float16) for k in CHANNELS},
                            coverage=coverage, **{f'mrs_{k}': v['tpl'] for k, v in vox.items()})
        info['seconds'] = time.time() - t0
        info['status'] = 'ok'
        qc(out, t2n, mask, b0c, maps, t2_tpl.numpy(), tpl.numpy(), chans, sid)
    except Exception as e:  # keep going; failures are reported, never silently imputed
        info['status'] = 'failed'
        info['error'] = repr(e)
        info['trace'] = traceback.format_exc()[-2000:]
    for f in [*out.glob('*InverseWarp.nii.gz'), *out.glob('sdc_*Warp.nii.gz'), out / 'sdc_half.nii.gz']:
        if f.exists(): f.unlink()
    (out / 'processing.json').write_text(json.dumps(info, indent=1, default=float) + '\n')
    return sid, info['status']


def qc(out, t2n, mask, b0c, maps, t2_tpl, tpl, chans, sid):
    fig, ax = plt.subplots(3, 5, figsize=(14, 8.4))
    t = t2n.numpy(); m = mask.numpy() > .5
    for j, z in enumerate(np.linspace(3, t.shape[2] - 4, 5).astype(int)):
        ax[0, j].imshow(t[:, :, z].T, cmap='gray'); ax[0, j].contour(m[:, :, z].T, [.5], colors='orange', linewidths=.6)
        ax[0, j].set_title(f'{sid} T2 z{z}', fontsize=8)
    b = b0c.numpy()
    for j, (name, a) in enumerate([('b0', b), *maps.items()]):
        z = a.shape[2] // 2
        ax[1, j].imshow(a[:, :, z].T, cmap='gray', vmax=np.percentile(a, 99.5)); ax[1, j].set_title(name, fontsize=8)
    for j, z in enumerate(np.linspace(8, tpl.shape[2] - 9, 5).astype(int)):
        ax[2, j].imshow(t2_tpl[:, :, z].T, cmap='gray'); ax[2, j].contour(tpl[:, :, z].T, 4, colors='c', linewidths=.4)
        ax[2, j].set_title(f'template space z{z}', fontsize=8)
    for a in ax.ravel(): a.axis('off')
    fig.tight_layout(); fig.savefig(out / 'qc.png', dpi=70); plt.close(fig)


def rows():
    inv = json.loads((BP / 'inventory.json').read_text())
    result = []
    for r in inv:
        if r['folder_time'] == '30d':
            continue  # excluded: internal labels say 7d; unresolved
        result.append({**r, 'id': f"{r['folder_time']}-{r['group']}-{r['animal']}".replace(' ', '_'),
                       'cohort': 'blast', 'time': r['folder_time']})
    sess = json.loads((ROOT / 'Ferret/visual-overview/session-inventory.json').read_text())
    for s in sess:
        mt = re.fullmatch(r'(BC[FR]\d+)_baseline', s['label'])
        if not mt:
            continue
        d = ROOT / 'Ferret' / s['path']
        scans = {}
        for sc in s['scans']:
            for key, pat in [('t2', r'04_RARE_T2_axial$'), ('dti', r'05_DTI_EPI_2Bval_30dir'), ('dti_neg', r'05b_DTI_EPI_B0_neg')]:
                if re.search(pat, sc['protocol']) and (d / sc['number'] / 'pdata/1/2dseq').exists():
                    scans.setdefault(key, []).append(sc['number'])
        result.append({'id': f'fiskum-{mt[1]}', 'session': str(d.relative_to(ROOT)), 'cohort': 'fiskum_baseline',
                       'group': 'baseline', 'time': None, 'animal': mt[1], 'scans': scans})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    todo = [r for r in rows() if not args.only or r['id'] in args.only]
    print(len(todo), 'sessions', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for sid, status in ex.map(process, todo):
            print(time.strftime('%H:%M:%S'), sid, status, flush=True)


if __name__ == '__main__':
    main()
