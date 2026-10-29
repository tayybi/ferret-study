"""Overview figure (method) for the ISBI paper: data, normative model and session-aware calibration."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ferret-mpl')
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from prepare_data import ROOT

BP = ROOT / 'ferret-study/blast-pipeline'
MD = BP / 'models'
OUT = ROOT / 'ferret-study/isbi2027/paper/figures'
INK, MUTED, LINE, FILL = '#1f2328', '#5b636b', '#9aa3ab', '#f2f4f6'
NAIVE, SACC, SACCM = '#eb6834', '#2a78d6', '#1baf7a'
Z = 43  # coronal slice with the largest analysis-mask area
plt.rcParams.update({'font.size': 6.5, 'font.family': 'serif', 'text.color': INK})


def box(ax, x, y, w, h, text='', fc=FILL, ec=LINE, fs=6.3, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=fc, ec=ec, lw=.7))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, **kw)


def arrow(ax, x0, y0, x1, y1, label='', color=MUTED, dy=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, color=color, lw=.8))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha='center', va='bottom', fontsize=5.6, color=MUTED)


def image(fig, ax, img, x, y, w, h, cmap='gray', vmax=None, title=None):
    """Place an image in data coordinates of the canvas axis."""
    t = ax.transData + fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = t.transform((x, y)), t.transform((x + w, y + h))
    a = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
    cm = plt.get_cmap(cmap).copy(); cm.set_bad('#3a3f45')
    a.imshow(img, cmap=cm, origin='lower', vmin=0, vmax=vmax, interpolation='nearest', aspect='auto')
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(LINE); s.set_linewidth(.5)
    if title:
        a.set_title(title, fontsize=5.6, pad=1.2, color=MUTED)
    return a


def main():
    rows = json.loads((MD / 'rows.json').read_text())
    brain = np.load(MD / 'brain_mask.npy')
    i = next(k for k, r in enumerate(rows) if r['cohort'] == 'blast' and r['group'] == 'Sham'
             and r['animal'] not in ('707761', '709611'))
    ch = np.load(BP / 'sessions' / rows[i]['id'] / 'channels.npz')

    def sl(v):
        return np.where(brain, v.astype(np.float32), 0)[13:63, 1:35, Z].T

    t2, fa, md = sl(ch['T2']), sl(ch['FA']), sl(ch['MD'])
    from blast_explore import atlas_labels
    lab = sl(atlas_labels(brain).astype(np.float32))
    lab = np.ma.masked_where(lab == 0, (lab * 7) % 20)
    R = np.load(MD / 'main' / 'residuals_seed0.npy', mmap_mode='r')[i].astype(np.float32).mean(0)
    rv = np.zeros(brain.shape, np.float32); rv[brain] = R
    res = sl(rv)
    rng = np.random.default_rng(3)
    hide = np.kron(rng.random((t2.shape[0] // 8 + 1, t2.shape[1] // 8 + 1)) < .5, np.ones((8, 8)))[:t2.shape[0], :t2.shape[1]]
    masked = np.where(hide, np.nan, t2)

    fig = plt.figure(figsize=(7.1, 2.45))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 142); ax.set_ylim(0, 49); ax.axis('off')
    for x, t in [(1, '(a) Data and region features'), (45, '(b) Sham-trained normative model'), (91, '(c) Session-aware calibration')]:
        ax.text(x, 47, t, fontsize=7, fontweight='bold', va='top')

    # (a) data
    vt2 = np.percentile(t2[t2 > 0], 99)
    image(fig, ax, t2, 1, 29, 13.4, 9.1, vmax=vt2, title='T2')
    image(fig, ax, fa, 15.3, 29, 13.4, 9.1, vmax=.8, title='FA')
    image(fig, ax, md, 29.6, 29, 13.4, 9.1, vmax=np.percentile(md[md > 0], 99), title='MD')
    a = image(fig, ax, t2, 1, 12.5, 13.4, 9.1, vmax=vt2, title='31 atlas regions')
    a.imshow(lab, cmap='tab20', origin='lower', alpha=.75, interpolation='nearest', aspect='auto')
    ax.text(16, 21.5, '52 ferrets, 21 scan sessions\nsham / 10 psi / 20 psi, 3 d / 7 d', fontsize=5.8, va='center')
    ax.text(16, 14.5, 'template SyN, distortion corr.,\nWLS tensors $\\rightarrow$ 6 channels\n(T2, FA, MD, AD, RD, log$|J|$)', fontsize=5.6, va='center', color=MUTED)
    ax.text(1, 6, 'feature $x_{rc}$ per region $r$, channel $c$:\nmean value (RAW) or mean $|x-\\hat{x}|$ (RES)', fontsize=5.6, color=MUTED)

    # (b) model
    image(fig, ax, masked, 45.5, 24.4, 12, 8.2, vmax=vt2, title='input, 50% hidden')
    arrow(ax, 58, 28.5, 61, 28.5)
    xs, ws, hs = [61.5, 64.3, 67.1, 69.9, 72.7], [2.4] * 5, [14, 10, 6, 10, 14]
    for x, w, h, c in zip(xs, ws, hs, [FILL, FILL, '#e3ecf8', FILL, FILL]):
        box(ax, x, 28.5 - h / 2, w, h, fc=c)
    ax.text(68.3, 37.3, 'U-Net', ha='center', fontsize=5.8)
    ax.text(68.3, 20.3, 'FiLM\nbottleneck', ha='center', va='top', fontsize=5.4, color=MUTED)
    ax.plot([62.7, 73.9], [35.5, 35.5], color=LINE, lw=.5, ls=':')
    arrow(ax, 75.5, 28.5, 78, 28.5)
    image(fig, ax, res, 78.5, 25, 10, 6.8, cmap='magma', vmax=np.percentile(res[res > 0], 99), title='$|x-\\hat{x}|$')
    ax.text(45.5, 13.5, 'trained on sham animals only, 3 seeds\n'
                      'RES-A: folds by animal   RES-S: folds by session\n'
                      '(no fold model sees the scored animal\'s session)', fontsize=5.6, va='top', color=MUTED)

    # (c) calibration: scored control k in session 1; sessions 2 and 3 (prone)
    sess = [('session 1', 3, 'supine'), ('session 2', 2, 'supine'), ('session 3', 2, 'prone')]
    rows_c = [('naive', NAIVE, {0, 1, 2}, 'all other controls'),
              ('SACC', SACC, {1, 2}, 'other sessions, weight $1/n_g$'),
              ('SACC+M', SACCM, {1}, 'other sessions, same position')]
    x0 = [110, 120.5, 130]
    for x, (name, n, pos) in zip(x0, sess):
        ax.text(x + (n * 2.6) / 2 + .3, 41.5, f'{name}\n{pos}', ha='center', fontsize=5.3, color=MUTED, va='center')
    for r, (name, col, used, desc) in enumerate(rows_c):
        y = 33 - r * 9.5
        ax.text(91, y, name, fontsize=6.3, fontweight='bold', color=col, va='center')
        ax.text(91, y - 3.3, desc, fontsize=5.2, color=MUTED, va='center')
        for s, (x, (_, n, pos)) in enumerate(zip(x0, sess)):
            excluded = s not in used
            ax.add_patch(FancyBboxPatch((x, y - 2.2), n * 2.6 + .6, 4.4, boxstyle='round,pad=0,rounding_size=.8',
                                        fc='white' if excluded else '#eef4ec', ec=LINE, lw=.6,
                                        hatch='////' if excluded else None))
            for k in range(n):
                cx = x + 1.6 + k * 2.6
                if s == 0 and k == 0:
                    ax.plot(cx, y, marker='*', ms=6.5, color=INK, zorder=4)
                else:
                    ax.plot(cx, y, marker='o', ms=3.6, mfc=col if not excluded else 'white', mec=col if not excluded else LINE, mew=.7, zorder=4)
    ax.plot(92.2, 4.5, marker='*', ms=6.5, color=INK)
    ax.text(93.6, 4.5, 'scored control $k$', fontsize=5.4, va='center', color=MUTED)
    ax.plot(108.5, 4.5, marker='o', ms=3.6, color=SACC)
    ax.text(109.6, 4.5, 'used to calibrate', fontsize=5.4, va='center', color=MUTED)
    ax.add_patch(Rectangle((126.5, 3.2), 3, 2.6, fc='white', ec=LINE, lw=.6, hatch='////'))
    ax.text(130.3, 4.5, 'excluded', fontsize=5.4, va='center', color=MUTED)
    ax.text(116.5, 0.6, 'conformal p per region $\\rightarrow$ BH ($q$ = 0.10) per animal', fontsize=5.6, ha='center', color=INK)
    for x in (44, 89.5):
        ax.plot([x, x], [2, 46], color='#e3e6e9', lw=.6)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'fig_overview.pdf'); fig.savefig(OUT / 'fig_overview.png', dpi=220)


if __name__ == '__main__':
    main()
