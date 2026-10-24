#!/usr/bin/env python3
"""
make_tables.py — LaTeX table bodies of the SR benchmark, gain, branch-ablation and loss-control
tables, from the per-image caches of eval_benchmarks.py.

Reads results/bench_x<s>/*.json and writes results/tables/*.tex and a summary of the mean
values per configuration and dataset (results/benchmarks_x<s>.json). WaSP = the final metric
with the default weights (1, 0.5, 0.1); --msssim_form mean gives the arithmetic-mean MS-SSIM
version of Branch A (snapshot tables of Supplementary S7, written as *_meanform.tex).

  python evaluation/make_tables.py --scale 4
  python evaluation/make_tables.py --scale 2
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.paths import RESULTS_DIR
from wasp import score_from_components

DS = ['Set5', 'Set14', 'BSD100', 'Urban100']


def wasp(r, variant='final'):
    """Per-image WaSP from the cached branches: 'final', 'mean' (MS-SSIM form) or 'legacy'."""
    struct = {'final': 'l_struct', 'mean': 'l_struct_mean', 'legacy': 'l_struct_legacy'}[variant]
    hal = 'l_hal_legacy' if variant == 'legacy' else 'l_hal'
    return score_from_components(np.array(r[struct]), np.array(r['l_percept']), np.array(r[hal]))


def load(in_dir, variant='final'):
    out = {}
    for f in sorted(glob.glob(os.path.join(in_dir, '*.json'))):
        d = json.load(open(f))
        out[d['config']] = {ds: {'PSNR': float(np.mean(r['psnr'])), 'SSIM': float(np.mean(r['ssim'])),
                                 'WaSP': float(np.mean(wasp(r, variant))),
                                 'WaSP_legacy': float(np.mean(wasp(r, 'legacy'))),
                                 'L_hal': float(np.mean(r['l_hal'])), 'LPIPS': float(np.mean(r['lpips'])),
                                 'n': len(r['psnr'])}
                            for ds, r in d['per_image'].items()}
    return out


def cells(res, cfg, bold=False):
    if cfg not in res:
        return ' & '.join(['--'] * 12)
    f = (lambda x: f'\\textbf{{{x}}}') if bold else (lambda x: x)
    return ' & '.join(f(f"{res[cfg][d]['PSNR']:.2f}") + ' & ' + f(f"{res[cfg][d]['SSIM']:.4f}") + ' & '
                      + f(f"{res[cfg][d]['WaSP']:.4f}") for d in DS)


def bench_rows(res, models):
    lines = []
    for base in models:
        if base in ('Bicubic', 'IMDN'):
            lines.append(f'{base}\n  & {cells(res, base)} \\\\')
            continue
        b = 'SRResNet-L1' if base == 'SRResNet' else base
        lines.append('\\midrule')
        lines.append(f'{b}\n  & {cells(res, b)} \\\\')
        lines.append(f'{base}+WaSP\n  & {cells(res, base + "+WaSP")} \\\\')
    return '\n'.join(lines)


def gain_rows(res, models):
    lines = []
    for base in models:
        b = 'SRResNet-L1' if base == 'SRResNet' else base
        w = base + '+WaSP'
        if b not in res or w not in res:
            continue
        c = []
        for d in DS:
            dp = res[w][d]['PSNR'] - res[b][d]['PSNR']
            dw = 100 * (res[w][d]['WaSP'] - res[b][d]['WaSP']) / res[b][d]['WaSP']
            c.append((f'${dp:+.2f}$' if abs(dp) >= 0.005 else '$0.00$') + f' & ${dw:+.1f}$')
        lines.append(f'{w:16s} & ' + ' & '.join(c) + ' \\\\')
    return '\n'.join(lines)


def ablation_rows(res, models):
    names = [('', '$\\ell_1$ (pretrained)'), (' abla_la', '$\\ell_1 + \\mathcal{L}_A$'),
             (' abla_lb', '$\\ell_1 + \\mathcal{L}_B$'), (' abla_lc', '$\\ell_1 + \\mathcal{L}_C$'),
             (' abla_lalb', '$\\ell_1 + \\mathcal{L}_A + \\mathcal{L}_B$'), ('+WaSP', '$\\ell_1$ + WaSP (full)')]
    lines = []
    for m in models:
        lines.append('\\midrule')
        lines.append(f'\\multirow{{7}}{{*}}{{\\textbf{{{m}}}}}')
        for suf, label in names:
            cfg = m + suf
            lines.append(f'  & {label}\n  & {cells(res, cfg, bold=(suf == "+WaSP"))} \\\\')
        if m in res and m + '+WaSP' in res:
            c = []
            for d in DS:
                dp = res[m + '+WaSP'][d]['PSNR'] - res[m][d]['PSNR']
                dw = 100 * (res[m + '+WaSP'][d]['WaSP'] - res[m][d]['WaSP']) / res[m][d]['WaSP']
                dps = f'{dp:+.2f}' if abs(dp) >= 0.005 else '0.00'
                c.append(f'\\textit{{${dps}$}} & \\textit{{---}} & \\textit{{${dw:+.1f}\\%$}}')
            lines.append('\\cmidrule(lr){2-14}')
            lines.append('& \\textit{$\\Delta$ vs.\\ $\\ell_1$-only}\n  & ' + '\n  & '.join(c) + ' \\\\')
    return '\n'.join(lines)


def control_rows(res, models, ds='Set5', full=False):
    lines = []
    for m in models:
        base = 'SRResNet-L1' if m == 'SRResNet' else m
        cols = [base] + [f'{m} ctrl_{k}' for k in ['l1', 'lpips', 'dists', 'msssim']] \
               + [m + '+WaSP']
        c = []
        for cfg in cols:
            if cfg not in res:
                c.append('--'); continue
            r = res[cfg][ds]
            c.append(f"{r['PSNR']:.2f}/{r['SSIM']:.4f}/{r['WaSP']:.4f}" if full
                     else f"{r['PSNR']:.2f}/{r['WaSP']:.3f}".replace('/0.', '/.'))
        lines.append(f'{m:10s} & ' + ' & '.join(c) + ' \\\\')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scale', type=int, default=4, choices=[2, 4])
    ap.add_argument('--msssim_form', default='product', choices=['product', 'mean'])
    ap.add_argument('--in_dir', default=None, help='default: results/bench_x<scale>')
    ap.add_argument('--out_dir', default=os.path.join(RESULTS_DIR, 'tables'))
    args = ap.parse_args()
    s = args.scale
    variant = 'final' if args.msssim_form == 'product' else 'mean'
    sfx = '' if variant == 'final' else '_meanform'
    res = load(args.in_dir or os.path.join(RESULTS_DIR, f'bench_x{s}'), variant)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f'benchmarks_x{s}{sfx}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    if s == 4:
        models = ['SRResNet', 'RCAN', 'EDSR', 'RealESRNet', 'RealESRGAN', 'SwinIR', 'SinSR']
        tabs = {'bench_x4': 'Bicubic\n  & ' + cells(res, 'Bicubic') + ' \\\\\nIMDN\n  & ' + cells(res, 'IMDN')
                + ' \\\\\n' + bench_rows(res, models),
                'gain_x4': gain_rows(res, models),
                'ablation_x4': ablation_rows(res, ['RCAN', 'EDSR', 'RealESRNet', 'RealESRGAN', 'SwinIR', 'SinSR']),
                'losscontrol_set5': control_rows(res, models)}
        for ds in DS:
            tabs[f'losscontrol_full_{ds}'] = control_rows(res, models, ds, full=True)
    else:
        models = ['RCAN', 'EDSR', 'RealESRGAN', 'SwinIR']
        tabs = {'bench_x2': 'Bicubic\n  & ' + cells(res, 'Bicubic') + ' \\\\\n' + bench_rows(res, models),
                'gain_x2': gain_rows(res, models),
                'losscontrol_x2_set5': control_rows(res, models)}
        for ds in DS:
            tabs[f'losscontrol_x2_full_{ds}'] = control_rows(res, models, ds, full=True)
    for k, v in tabs.items():
        with open(os.path.join(args.out_dir, f'{k}{sfx}.tex'), 'w') as f:
            f.write(v + '\n')
    print(f'{len(res)} configs; wrote {len(tabs)} tables to {args.out_dir}')


if __name__ == '__main__':
    main()
