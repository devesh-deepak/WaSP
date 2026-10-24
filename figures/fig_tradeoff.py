#!/usr/bin/env python3
"""
fig_tradeoff.py — perception-distortion trade-off, PSNR vs WaSP on the four x4 benchmarks
(Fig. 3), from results/benchmarks_x4.json (evaluation/make_tables.py).

  python figures/fig_tradeoff.py        -> results/figures/fig3_tradeoff_combined.png
"""

from common import MODEL_PALETTE, STYLE, savefig  # noqa: I001  (sets the backend and sys.path)

import json
import os

import matplotlib.pyplot as plt

from utils.paths import RESULTS_DIR

DATASETS = ['Set5', 'Set14', 'BSD100', 'Urban100']
MODELS = ['Bicubic', 'IMDN', 'SRResNet-L1', 'SRResNet+WaSP'] + \
         [n for m in ['RCAN', 'EDSR', 'RealESRNet', 'RealESRGAN', 'SwinIR', 'SinSR'] for n in (m, m + '+WaSP')]


def main():
    plt.rcParams.update(STYLE)
    bench = json.load(open(os.path.join(RESULTS_DIR, 'benchmarks_x4.json')))
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.8 * len(DATASETS), 5.0))
    fig.patch.set_alpha(0)
    handles = {}
    for ax, ds in zip(axes, DATASETS):
        ax.set_facecolor('none')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color('#888')
        ax.set_title(ds, fontsize=11, color='black')
        ax.set_xlabel('PSNR ↑ (dB)', fontsize=9, color='black')
        ax.set_ylabel('WaSP Score ↓', fontsize=9, color='black')
        ax.tick_params(colors='black')
        for name in MODELS:
            if name not in bench:
                continue
            res = bench[name][ds]
            col = MODEL_PALETTE.get(name, '#777777')
            is_wasp = '+WaSP' in name
            marker = 'D' if is_wasp else 'o'
            # values rounded to 4 decimals, as in the paper's figure
            ax.scatter(round(res['PSNR'], 4), round(res['WaSP'], 4), s=90 if is_wasp else 65, color=col, marker=marker,
                       edgecolors='black', linewidths=0.6, zorder=5)
            if name not in handles:
                handles[name] = plt.scatter([], [], s=80 if is_wasp else 65, color=col, marker=marker,
                                            edgecolors='black', linewidths=0.5, label=name)
    leg = fig.legend(handles=list(handles.values()), loc='lower center', ncol=min(len(handles), 7),
                     fontsize=8.5, framealpha=0.9, edgecolor='#aaaaaa', bbox_to_anchor=(0.5, -0.14))
    for txt in leg.get_texts():
        txt.set_color('black')
    plt.suptitle('Perceptual–Distortion Tradeoff: PSNR vs WaSP Score',
                 fontsize=13, fontweight='bold', color='black', y=1.02)
    plt.tight_layout(pad=1.0)
    savefig('fig3_tradeoff_combined.png')


if __name__ == '__main__':
    main()
