#!/usr/bin/env python3
"""
fig_weight_tradeoff.py — MOS correlation vs. hallucination sensitivity across WaSP aggregation
weights (final metric).

Reads results/hal_validation.json (evaluation/hal_validation.py) and results/mos_analysis.json
(evaluation/mos_analysis.py). Writes results/figures/fig_weight_tradeoff.png
"""

from common import out_path  # noqa: I001  (sets the backend and sys.path)

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.paths import RESULTS_DIR

HAL   = json.load(open(os.path.join(RESULTS_DIR, 'hal_validation.json')))
MOS   = json.load(open(os.path.join(RESULTS_DIR, 'mos_analysis.json')))
OUT   = out_path('fig_weight_tradeoff.png')

# Ordinal blue ramp (light -> dark), validated for a white surface
RAMP   = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
INK    = '#0b0b0b'
INK2   = '#52514e'
GRID   = '#e4e3df'
DS     = ['TID2013', 'KADID-10K', 'PIPAL']

grid = HAL['corrected']
lhs  = sorted({g['lambda'][2] for g in grid})
lps  = sorted({g['lambda'][1] for g in grid})

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2})
fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=300)
ax.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

for col, lh in zip(RAMP, lhs):
    pts = sorted([g for g in grid if g['lambda'][2] == lh], key=lambda g: g['lambda'][1])
    x = [100 * g['hal_rank_acc']['all'] for g in pts]
    y = [g['mos_srcc_avg'] for g in pts]
    ax.plot(x, y, '-', color=col, linewidth=2, zorder=2)
    ax.plot(x, y, 'o', color=col, markersize=6, markeredgecolor='white', markeredgewidth=1.2,
            zorder=3, label=f'$\\lambda_h$ = {lh:g}')
ax.text(0.53, 0.10, 'Each line: $\\lambda_s$ = 1 and $\\lambda_p$ = 0.25, 0.5, 1, 2, 4\n'
        '(from the right-hand end to the left-hand end)', transform=ax.transAxes,
        fontsize=7.5, color=INK2, ha='center')

# Default weights
d = [g for g in grid if tuple(g['lambda']) == (1.0, 0.5, 0.1)][0]
dx, dy = 100 * d['hal_rank_acc']['all'], d['mos_srcc_avg']
ax.plot(dx, dy, 'o', markersize=13, markerfacecolor='none', markeredgecolor=INK,
        markeredgewidth=1.4, zorder=4)
ax.annotate('default (1, 0.5, 0.1)', (dx, dy), xytext=(-14, -16), textcoords='offset points',
            ha='right', fontsize=8, color=INK,
            arrowprops=dict(arrowstyle='-', color=INK, linewidth=0.8))

# Reference metrics
perc = MOS['corrected']['percept-only']
refs = {
    'perceptual\nbranch only': (100 * HAL['baselines']['percept-only']['all'],
                               np.mean([perc[k]['SRCC'] for k in DS]), 's'),
    'LPIPS': (100 * HAL['baselines']['LPIPS']['all'],
              np.mean([MOS['lpips'][k]['SRCC'] for k in DS]), 'D'),
}
offsets = {'perceptual\nbranch only': (-6, -24), 'LPIPS': (-4, 6)}
aligns  = {'perceptual\nbranch only': 'left', 'LPIPS': 'right'}
for name, (x, y, mk) in refs.items():
    ax.plot(x, y, mk, color=INK2, markersize=6.5, markeredgecolor='white', markeredgewidth=1.0, zorder=5)
    ax.annotate(name, (x, y), xytext=offsets[name], textcoords='offset points', fontsize=8,
                color=INK2, ha=aligns[name])

ax.set_xlabel('Hallucination-ranking accuracy (%)\n(hallucinated texture scored worse than an equal-PSNR control)')
ax.set_ylabel('Mean SRCC with MOS\n(TID2013, KADID-10K, PIPAL)')
ax.legend(loc='lower left', frameon=False, fontsize=8, title='hallucination weight',
          title_fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=300, facecolor='white')
print(f'saved -> {OUT}')
