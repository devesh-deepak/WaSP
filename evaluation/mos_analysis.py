#!/usr/bin/env python3
"""
mos_analysis.py — MOS correlation of WaSP, its branches and its aggregation weights.

Reads results/mos_components_<dataset>.npz (mos_components.py) and, for both versions of the
metric (legacy = as submitted: clamped LL band and strict '<' smooth mask; corrected = the
final metric: LL / 2^J, product-form MS-SSIM and '<=' mask), reports
  * SRCC / PLCC / KRCC of single branches, branch combinations and weight sweeps
    ('percept-only' etc. use the tanh-normalised branches; 'branches_raw' the raw values)
  * the same on PIPAL subsets: GAN-based SR outputs (code 03) and all SR outputs
  * paired percentile bootstrap (B = 2000) of the full metric against the perceptual branch
    alone and against LPIPS, and of the perceptual branch against LPIPS
  * percentile-bootstrap 95% CI of the SRCC of each metric

Correlations are sign-corrected so that positive = agreement with MOS (higher MOS = better on
all three datasets; every WaSP quantity and LPIPS is lower-is-better).

Output: results/mos_analysis.json

  python evaluation/mos_analysis.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from utils.paths import RESULTS_DIR

OUT      = os.path.join(RESULTS_DIR, 'mos_analysis.json')
DATASETS = ['TID2013', 'KADID-10K', 'PIPAL']
SCALES   = {'s': 0.30, 'p': 0.05, 'h': 0.01}     # tanh soft-normalisation constants
DEFAULT  = (1.0, 0.5, 0.1)                       # (lambda_s, lambda_p, lambda_h)
SR_CODES = {'00', '01', '02', '03', '05'}        # PIPAL categories containing SR outputs
B_BOOT   = 2000


def load(name):
    return dict(np.load(os.path.join(RESULTS_DIR, f'mos_components_{name}.npz')))


def normed(d, variant):
    sfx = '_legacy' if variant == 'legacy' else ''
    return (np.tanh(d['l_struct' + sfx] / SCALES['s']),
            np.tanh(d['l_percept'] / SCALES['p']),
            np.tanh(d['l_hal' + sfx] / SCALES['h']))


def composite(n, lam):
    return lam[0] * n[0] + lam[1] * n[1] + lam[2] * n[2]


def corr(score, mos):
    s = -np.asarray(score, dtype=np.float64)
    return {'SRCC': float(stats.spearmanr(s, mos)[0]),
            'PLCC': float(stats.pearsonr(s, mos)[0]),
            'KRCC': float(stats.kendalltau(s, mos)[0])}


def paired_bootstrap(a, b, mos, seed=0):
    """Delta = corr(a) - corr(b) on identical resamples; returns point, 95% CI, p."""
    rng = np.random.default_rng(seed)
    n = len(mos)
    out = {}
    for name, fn in [('SRCC', lambda x, m: stats.spearmanr(-x, m)[0]),
                     ('PLCC', lambda x, m: stats.pearsonr(-x, m)[0])]:
        point = fn(a, mos) - fn(b, mos)
        deltas = np.empty(B_BOOT)
        for i in range(B_BOOT):
            idx = rng.integers(0, n, n)
            deltas[i] = fn(a[idx], mos[idx]) - fn(b[idx], mos[idx])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
        out[name] = {'delta': float(point), 'ci95': [float(lo), float(hi)],
                     'p': float(max(p, 1.0 / B_BOOT))}
    return out


def configs():
    c = {'percept-only': (0, 1, 0), 'struct-only': (1, 0, 0), 'hal-only': (0, 0, 1),
         'struct+percept': (1, 0.5, 0), 'percept+hal': (0, 0.5, 0.1),
         'default (1.0,0.5,0.1)': DEFAULT, 'equal (1,1,1)': (1, 1, 1)}
    for v in [0.25, 1.0, 2.0, 4.0]:
        c[f'lp={v}'] = (1.0, v, 0.1)
    for v in [0.0, 0.5, 2.0, 4.0]:
        c[f'ls={v}'] = (v, 0.5, 0.1)
    for v in [0.0, 0.05, 0.2, 0.5]:
        c[f'lh={v}'] = (1.0, 0.5, v)
    return c


def main():
    data = {k: load(k) for k in DATASETS}
    pipal = data['PIPAL']
    subsets = {'PIPAL-GAN': pipal['codes'] == '03',
               'PIPAL-SR':  np.isin(pipal['codes'], list(SR_CODES))}

    res = {'n': {k: int(len(v['mos'])) for k, v in data.items()}}
    res['n'].update({k: int(m.sum()) for k, m in subsets.items()})

    for variant in ['legacy', 'corrected']:
        r = {}
        for cname, lam in configs().items():
            r[cname] = {}
            for ds, d in data.items():
                r[cname][ds] = corr(composite(normed(d, variant), lam), d['mos'])
            for sname, mask in subsets.items():
                n = [x[mask] for x in normed(pipal, variant)]
                r[cname][sname] = corr(composite(n, lam), pipal['mos'][mask])
        res[variant] = r

    res['branches_raw'] = {ds: {k: corr(d[k], d['mos']) for k in ['l_struct', 'l_percept', 'l_hal']}
                           for ds, d in data.items()}
    res['lpips'] = {ds: corr(d['lpips'], d['mos']) for ds, d in data.items()}
    res['lpips'].update({s: corr(pipal['lpips'][m], pipal['mos'][m]) for s, m in subsets.items()})

    # Paired bootstrap (corrected metric): full and struct+percept vs alternatives
    boot = {}
    for ds, d in data.items():
        n = normed(d, 'corrected')
        full, sp, perc = composite(n, DEFAULT), composite(n, (1, 0.5, 0)), n[1]
        boot[ds] = {'full_vs_percept': paired_bootstrap(full, perc, d['mos']),
                    'full_vs_lpips':   paired_bootstrap(full, d['lpips'], d['mos']),
                    'sp_vs_percept':   paired_bootstrap(sp, perc, d['mos']),
                    'sp_vs_lpips':     paired_bootstrap(sp, d['lpips'], d['mos']),
                    'percept_vs_lpips': paired_bootstrap(d['l_percept'], d['lpips'], d['mos'])}
    for sname, m in subsets.items():
        n = normed(pipal, 'corrected')
        full, perc = composite(n, DEFAULT)[m], n[1][m]
        boot[sname] = {'full_vs_percept': paired_bootstrap(full, perc, pipal['mos'][m]),
                       'full_vs_lpips':   paired_bootstrap(full, pipal['lpips'][m], pipal['mos'][m])}
    res['bootstrap_corrected'] = boot

    # 95% CI of each metric's SRCC (percentile bootstrap) for the supplementary table
    rng = np.random.default_rng(0)
    ci = {}
    for ds, d in data.items():
        n_leg, n_cor = normed(d, 'legacy'), normed(d, 'corrected')
        metrics = {'WaSP': composite(n_cor, DEFAULT), 'WaSP-Struct': d['l_struct'],
                   'WaSP-Hal': d['l_hal'], 'WaSP (submitted)': composite(n_leg, DEFAULT),
                   'WaSP-Struct (submitted)': d['l_struct_legacy'], 'WaSP-Hal (submitted)': d['l_hal_legacy']}
        idx = [rng.integers(0, len(d['mos']), len(d['mos'])) for _ in range(B_BOOT)]
        ci[ds] = {}
        for k, v in metrics.items():
            bs = [stats.spearmanr(-v[i], d['mos'][i])[0] for i in idx]
            ci[ds][k] = [float(stats.spearmanr(-v, d['mos'])[0])] + [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    res['srcc_ci'] = ci

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)

    cols = DATASETS + ['PIPAL-GAN', 'PIPAL-SR']
    for variant in ['legacy', 'corrected']:
        print(f'\n=== {variant} metric — SRCC ===')
        print(f'{"config":24s}' + ''.join(f'{c:>11s}' for c in cols) + f'{"avg3":>8s}')
        for cname, row in res[variant].items():
            avg = np.mean([row[d]['SRCC'] for d in DATASETS])
            print(f'{cname:24s}' + ''.join(f'{row[c]["SRCC"]:11.4f}' for c in cols) + f'{avg:8.4f}')
    avg = np.mean([res['lpips'][d]['SRCC'] for d in DATASETS])
    print(f'{"LPIPS":24s}' + ''.join(f'{res["lpips"][c]["SRCC"]:11.4f}' for c in cols) + f'{avg:8.4f}')
    print('\n=== bootstrap: full (fixed, default) minus X ===')
    for ds, b in boot.items():
        for k, v in b.items():
            print(f'{ds:10s} {k:11s} dSRCC={v["SRCC"]["delta"]:+.4f} {np.round(v["SRCC"]["ci95"],4)} p={v["SRCC"]["p"]:.4f}'
                  f' | dPLCC={v["PLCC"]["delta"]:+.4f} {np.round(v["PLCC"]["ci95"],4)} p={v["PLCC"]["p"]:.4f}')
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
