"""Figure 1 for the ISBI paper, built only from saved results files."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ferret-mpl')
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare_data import ROOT

OUT = ROOT / 'ferret-study/isbi2027'
INK, MUTED, GRID = '#1f2328', '#5b636b', '#d8dde2'
STYLE = {'naive': ('#eb6834', 'o', '-', 'naive (animal-level)'),
         'sacc': ('#2a78d6', 's', '--', 'SACC'),
         'sacc_m': ('#1baf7a', '^', '-', 'SACC+M')}
plt.rcParams.update({'font.size': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlesize': 7.5, 'axes.titleweight': 'bold', 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'serif'})


def main():
    r = json.loads((OUT / 'results_isbi.json').read_text())
    d = json.loads((OUT / 'results_design_sim.json').read_text())
    pv = np.load(OUT / 'validity_pvalues.npz')
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.0), gridspec_kw={'wspace': .55})
    for a in ax:
        a.grid(axis='y', color=GRID, lw=.5); a.set_axisbelow(True)

    g = np.linspace(0, .2, 81)
    ax[0].plot(g, g, color=MUTED, lw=.8, ls=':')
    for m, (c, mk, ls, lab) in STYLE.items():
        p = pv[f'RAW__{m}']
        ax[0].plot(g, [(p <= x).mean() for x in g], color=c, ls=ls, lw=1.4)
    ax[0].set(xlabel='nominal α', ylabel='P(p ≤ α)', xlim=(0, .2), ylim=(0, .2))
    ax[0].set_title('(a) Validity', loc='left')

    for m, (c, mk, ls, lab) in STYLE.items():
        x = r['exp3_power_RAW'][m]
        ax[1].plot([e['amp'] for e in x], [e['sensitivity_q10'] for e in x], color=c, marker=mk, ms=3.2, ls=ls, lw=1.2, label=lab)
    vc = r['exp3_voxel_comparator_RAW']
    ax[1].plot([e['amp'] for e in vc], [e['region_detected_q10'] for e in vc], color=MUTED, marker='x', ms=3.2, ls=':', lw=1, label='voxel-level BH')
    ax[1].set(xlabel='regional shift (SD)', ylabel='sensitivity', ylim=(-.01, .3))
    ax[1].set_title('(b) Power', loc='left')
    ax[1].legend(fontsize=5.4, loc='upper left', handlelength=2.2)

    cells = d['cells']
    S = sorted({c['S'] for c in cells})
    sens = [next(c['sensitivity'] for c in cells if c['S'] == s and c['amp'] == 3) for s in S]
    null = [next(c['any_flag_rate'] for c in cells if c['S'] == s and c['amp'] == 0) for s in S]
    ax[2].plot(S, sens, color=STYLE['sacc'][0], marker='s', ms=3.2, lw=1.2, label='sensitivity (3 SD)')
    ax[2].plot(S, null, color=INK, marker='o', ms=3.2, lw=1.2, ls='--', label='false alarms (null)')
    ax[2].axhline(.10, color=MUTED, lw=.8, ls=':'); ax[2].text(5.2, .105, 'q = 0.10', color=MUTED, fontsize=6)
    ax[2].set_xscale('log', base=2); ax[2].set_xticks(S, [str(s) for s in S])
    ax[2].set(xlabel='control sessions S', ylabel='rate', ylim=(0, .2))
    ax[2].set_title('(c) Design', loc='left')
    ax[2].legend(fontsize=5.6, loc='upper left')

    summ = r['exp4_application']['RAW/sacc_m']['summary']
    keys = [k for k in ['3d-Sham', '3d-10psi', '3d-20psi', '7d-Sham', '7d-10psi', '7d-20psi'] if k in summ]
    vals = [summ[k]['share_any_flag'] for k in keys]
    ax[3].bar(range(len(keys)), vals, width=.62, color='#8a96a3', edgecolor='white', lw=1)
    for i, k in enumerate(keys):
        n = summ[k]['n']; f = round(summ[k]['share_any_flag'] * n)
        ax[3].text(i, vals[i] + .01, f'{f}/{n}', ha='center', fontsize=6, color=INK)
    ax[3].set_xticks(range(len(keys)), [k.split('-')[1].replace('psi', '').replace('Sham', 'S') + '\n' + k.split('-')[0] for k in keys], fontsize=5.6)
    ax[3].set(ylabel='share flagged', ylim=(0, .5))
    ax[3].set_title('(d) Blast cohort', loc='left')
    fig.savefig(OUT / 'paper' / 'fig1.pdf', bbox_inches='tight'); fig.savefig(OUT / 'figures' / 'fig1.png', dpi=220, bbox_inches='tight')


if __name__ == '__main__':
    main()
