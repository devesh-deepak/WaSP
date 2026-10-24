#!/usr/bin/env python3
"""
fig_mos_correlation.py — agreement with human MOS of PSNR, SSIM, LPIPS, WaSP-Percept (Branch B)
and WaSP: bar charts (Fig. 4) and radar charts (Fig. 7) for SRCC, PLCC and KRCC.

Reads results/mos_baselines.json (evaluation/mos_baselines.py) and results/mos_analysis.json
(evaluation/mos_analysis.py). WaSP-Percept uses the raw Branch B value and WaSP the final
metric with the default weights.

  python figures/fig_mos_correlation.py   -> results/figures/fig4_*_bars.png, fig7_*_radar.png
"""

from common import STYLE, out_path, savefig  # noqa: I001  (sets the backend and sys.path)

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.paths import RESULTS_DIR

DATASETS = ['TID2013', 'KADID-10K', 'PIPAL']


def load_correlations():
    """{corr: {dataset: {metric: value}}}, values rounded to 4 decimals as in the paper."""
    base = json.load(open(os.path.join(RESULTS_DIR, 'mos_baselines.json')))
    ana = json.load(open(os.path.join(RESULTS_DIR, 'mos_analysis.json')))
    out = {}
    for c in ['SRCC', 'PLCC', 'KRCC']:
        out[c] = {ds: {'PSNR': base[ds]['psnr'][c], 'SSIM': base[ds]['ssim'][c], 'LPIPS': base[ds]['lpips'][c],
                       'WaSP-Percept': ana['branches_raw'][ds]['l_percept'][c],
                       'WaSP': ana['corrected']['default (1.0,0.5,0.1)'][ds][c]}
                  for ds in DATASETS}
        out[c] = {ds: {k: round(v, 4) for k, v in d.items()} for ds, d in out[c].items()}
    return out


def stitch(names, out_name):
    """Place images side by side (transparent background)."""
    ims = [Image.open(out_path(n)).convert('RGBA') for n in names]
    canvas = Image.new('RGBA', (sum(i.width for i in ims), max(i.height for i in ims)), (0, 0, 0, 0))
    x = 0
    for im in ims:
        canvas.alpha_composite(im, (x, 0))
        x += im.width
    canvas.save(out_path(out_name))
    print(f'  saved -> {out_path(out_name)}')


def fig4_correlation_bars(DATA_CORR=None):
    print('[Fig4] MOS correlation bars (3 separate figures) …')

    DATA_CORR = DATA_CORR or load_correlations()
    metrics_order = ['PSNR', 'SSIM', 'LPIPS', 'WaSP-Percept', 'WaSP']
    ds_list       = ['TID2013', 'KADID-10K', 'PIPAL']
    ds_hatches    = ['', '//', 'xx']
    bar_w = 0.22

    metric_colors  = {'PSNR': '#78909C', 'SSIM': '#FF8F00',
                      'LPIPS': '#AD1457', 'WaSP-Percept': '#1565C0', 'WaSP': '#2E7D32'}

    for corr_type, corr_data in DATA_CORR.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#888')
        ax.spines['left'].set_color('#888')

        x = np.arange(len(metrics_order))
        for i, (ds, hatch) in enumerate(zip(ds_list, ds_hatches)):
            vals = [corr_data[ds][m] for m in metrics_order]
            bar_x = x + (i - 1) * bar_w
            bars = ax.bar(bar_x, vals, bar_w,
                          label=ds,
                          color=[metric_colors[m] for m in metrics_order],
                          alpha=0.75, edgecolor='black', linewidth=0.5,
                          hatch=hatch)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.006,
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=7, color='black')

        # Annotation: WaSP-Percept > LPIPS on PIPAL
        pipal_lpips = corr_data['PIPAL']['LPIPS']
        pipal_wasp  = corr_data['PIPAL']['WaSP-Percept']
        if pipal_wasp > pipal_lpips:
            wi = metrics_order.index('WaSP-Percept')
            ax.annotate('WaSP-Percept > LPIPS\non PIPAL ★',
                        xy=(wi + 1 * bar_w, pipal_wasp + 0.035),
                        xytext=(wi + 1 * bar_w, 0.86), ha='center', va='bottom',
                        fontsize=8, color='#E65100', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.3))

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_order, fontsize=10, color='black')
        ax.set_ylabel(f'{corr_type} ↑', fontsize=11, color='black')
        ax.set_ylim(0, 0.98)
        ax.tick_params(colors='black')
        ax.set_title(f'Human MOS Correlation — {corr_type}',
                     fontsize=12, fontweight='bold', color='black')

        # Dataset legend (hatch pattern)
        handles = [mpatches.Patch(facecolor='#aaaaaa', edgecolor='black',
                                  hatch=h, label=ds, alpha=0.8)
                   for ds, h in zip(ds_list, ds_hatches)]
        leg = ax.legend(handles=handles, title='Dataset', title_fontsize=9,
                        fontsize=9, framealpha=0.9, loc='upper left')
        for txt in leg.get_texts():
            txt.set_color('black')
        leg.get_title().set_color('black')

        plt.tight_layout(pad=1.0)
        savefig(f'fig4_{corr_type.lower()}_bars.png')


def fig7_triple_radar():
    print('[Fig7] Triple radar charts (3 separate figures) …')

    metrics = ['PSNR', 'SSIM', 'LPIPS', 'WaSP-Percept (L_B)', 'WaSP (Full)']
    metrics_short = ['PSNR', 'SSIM', 'LPIPS', 'WaSP-\nPercept', 'WaSP\n(Full)']

    data = load_correlations()
    order = ['PSNR', 'SSIM', 'LPIPS', 'WaSP-Percept', 'WaSP']
    RADAR_DATA = {c: {ds: [round(v[m], 3) for m in order] for ds, v in d.items()} for c, d in data.items()}

    n_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    for corr_type, corr_data in RADAR_DATA.items():
        fig, ax = plt.subplots(figsize=(6.5, 6.5),
                               subplot_kw=dict(polar=True))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.set_ylim(0, 0.95)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'],
                           fontsize=8, color='black')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_short, fontsize=10, color='black')
        ax.grid(color='#cccccc', linewidth=0.6, linestyle='-')
        ax.spines['polar'].set_color('#bbbbbb')
        ax.tick_params(colors='black')

        ds_styles = {
            'TID2013':   ('#1565C0', 'o-',  0.15),
            'KADID-10K': ('#AD1457', 's--', 0.15),
            'PIPAL':     ('#E65100', 'D:',  0.20),
        }
        for ds, (col, ls, alpha) in ds_styles.items():
            vals = corr_data[ds] + corr_data[ds][:1]
            ax.plot(angles, vals, ls, lw=2, color=col, ms=5, label=ds)
            ax.fill(angles, vals, alpha=alpha, color=col)

        ax.set_title(f'Metric Correlation Profile — {corr_type}',
                     fontsize=12, fontweight='bold', color='black', pad=20)

        leg = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                        fontsize=9, framealpha=0.9)
        for txt in leg.get_texts():
            txt.set_color('black')

        plt.tight_layout(pad=1.2)
        savefig(f'fig7_{corr_type.lower()}_radar.png')
    stitch([f'fig7_{c.lower()}_radar.png' for c in RADAR_DATA], 'fig7_triple_radar.png')


def main():
    plt.rcParams.update(STYLE)
    fig4_correlation_bars()
    fig7_triple_radar()


if __name__ == '__main__':
    main()
