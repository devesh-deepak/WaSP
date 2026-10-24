#!/usr/bin/env python3
"""
fig_hparam.py — the four hyperparameter-sensitivity panels of the paper (final metric), from
results/hparam_sensitivity.json (evaluation/hparam_sweep.py).

Writes results/figures/fig_hparam_{quantile,levels,wavelet,weights}.png
"""

from common import OUT_DIR, out_path  # noqa: I001  (sets the backend and sys.path)

import json
import os

import matplotlib.pyplot as plt

from utils.paths import RESULTS_DIR

R     = json.load(open(os.path.join(RESULTS_DIR, 'hparam_sensitivity.json')))
OUT   = OUT_DIR
DS    = ['TID2013', 'KADID-10K', 'PIPAL']
STYLE = {'TID2013': ('#2a78d6', 'o'), 'KADID-10K': ('#eb6834', 's'), 'PIPAL': ('#1baf7a', '^')}


def val(v):
    return v['WaSP_SRCC'] if isinstance(v, dict) else v


def panel(name, title, xlabel, keys, xs, default, xticks=None, logx=False):
    fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=150)
    for ds in DS:
        c, m = STYLE[ds]
        ax.plot(xs, [val(R[name][ds][k]) for k in keys], marker=m, color=c, linewidth=1.6,
                markersize=6, label=ds)
    ax.axvline(default, color='#52514e', linestyle='--', linewidth=1)
    if logx:
        ax.set_xscale('log', base=2)
    if xticks is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticks)
    ax.set_ylim(0.3, 0.85)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('WaSP-total SRCC vs MOS')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(out_path(f'fig_hparam_{name}.png'), dpi=150, facecolor='white')
    plt.close(fig)


def main():
    qs = ['0.10', '0.15', '0.20', '0.25', '0.30', '0.40', '0.50']
    panel('quantile', '(a) Quantile threshold $q$ (default 0.25)', '$q$', qs, [float(q) for q in qs], 0.25)
    panel('levels', '(b) DWT levels $L$ (default 3)', '$L$', ['1', '2', '3', '4'], [1, 2, 3, 4], 3,
          xticks=['1', '2', '3', '4'])
    wv = ['haar', 'db2', 'db4', 'sym4', 'bior2.2']
    panel('wavelet', '(c) Wavelet basis (default Haar)', 'wavelet', wv, list(range(len(wv))), 0,
          xticks=['Haar', 'db2', 'db4', 'sym4', 'bior2.2'])
    lps = ['lp=0.25', 'lp=0.5', 'lp=1.0', 'lp=2.0', 'lp=4.0']
    panel('weights', '(d) Perceptual weight $\\lambda_p$ (default 0.5)', '$\\lambda_p$', lps,
          [0.25, 0.5, 1, 2, 4], 0.5, xticks=['0.25', '0.5', '1', '2', '4'], logx=True)
    print(f'saved fig_hparam_*.png -> {OUT}')


if __name__ == '__main__':
    main()
