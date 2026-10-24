#!/usr/bin/env python3
"""
real_sr_hallucination.py — does WaSP flag hallucination in real SR outputs?

A. PIPAL (all 200 training references): GAN-based SR outputs (category 03, whose
   details PIPAL's authors describe as "wrong and different from the reference")
   against PSNR-oriented SR outputs (category 01) of the same reference. Only pairs
   whose RGB PSNR differs by at most 0.25 dB are kept, so pixel fidelity cannot
   separate them. For every metric we report how often it scores the GAN output as
   worse, and how often human raters (MOS) preferred the PSNR-oriented output.

B. RRDB network interpolation, x4 on Set5/Set14/BSD100/Urban100: weights move from
   RealESRNet (alpha 0, L1-trained) to RealESRGAN (alpha 1, GAN-trained) with the
   same architecture and training data, so alpha controls the amount of GAN-generated
   texture. Reads the per-image cache of eval_benchmarks.py (results/bench_x4/).

Outputs: results/pipal_sr_outputs.npz (cached), results/real_sr_hallucination.json

  python evaluation/real_sr_hallucination.py [--skip-pipal]
"""

import glob
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import stats

from utils.paths import IQA_DATA, RESULTS_DIR
from wasp import WaSPMetric
from wasp import score_from_components as wasp_total

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT   = (os.path.join(IQA_DATA, 'PIPAL', 'PIPAL') if os.path.isdir(os.path.join(IQA_DATA, 'PIPAL', 'PIPAL'))
          else os.path.join(IQA_DATA, 'PIPAL'))
CACHE  = RESULTS_DIR
OUT    = os.path.join(RESULTS_DIR, 'real_sr_hallucination.json')
TOL_DB = 0.25


@torch.no_grad()
def pipal_cache():
    path = os.path.join(CACHE, 'pipal_sr_outputs.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    import lpips
    wasp = WaSPMetric().to(DEVICE).eval()
    lp_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
    dist = {}
    for dd in glob.glob(f'{ROOT}/Distortion_*'):
        for f in os.listdir(dd):
            dist[f] = os.path.join(dd, f)
    rows = []
    for lab in sorted(glob.glob(f'{ROOT}/Train_Label/*.txt')):
        ref = os.path.basename(lab).replace('.txt', '.bmp')
        r = TF.to_tensor(Image.open(f'{ROOT}/Train_Ref/{ref}').convert('RGB')).unsqueeze(0).to(DEVICE)
        r_ll, r_hf = wasp.dwt(r)
        for line in open(lab):
            name, mos = [x.strip() for x in line.split(',')]
            code = name.split('_')[1]
            if code not in ('01', '03'):
                continue
            d = TF.to_tensor(Image.open(dist[name]).convert('RGB')).unsqueeze(0).to(DEVICE)
            if d.shape[-2:] != r.shape[-2:]:
                d = F.interpolate(d, size=r.shape[-2:], mode='bilinear', align_corners=False)
            d_ll, d_hf = wasp.dwt(d)
            rows.append((ref, name, code, float(mos),
                         10 * math.log10(1 / F.mse_loss(d, r).item()),
                         wasp.structural_loss(d_ll, r_ll).item(), wasp.perceptual_loss(d, r).item(),
                         wasp.hallucination_penalty(d_hf, r_hf).item(), lp_fn(d * 2 - 1, r * 2 - 1).item()))
        print(f'  {ref}: {len(rows)} outputs', flush=True)
    cols = list(zip(*rows))
    out = {'ref': np.array(cols[0]), 'name': np.array(cols[1]), 'code': np.array(cols[2]),
           'mos': np.array(cols[3]), 'psnr': np.array(cols[4]), 'l_struct': np.array(cols[5]),
           'l_percept': np.array(cols[6]), 'l_hal': np.array(cols[7]), 'lpips': np.array(cols[8])}
    np.savez(path, **out)
    return out


def pipal_analysis(d, tol=None):
    tol = TOL_DB if tol is None else tol
    metrics = {'WaSP-Hal': d['l_hal'], 'WaSP': wasp_total(d['l_struct'], d['l_percept'], d['l_hal']),
               'WaSP-Percept': d['l_percept'], 'LPIPS': d['lpips'], 'PSNR (lower = worse)': -d['psnr']}
    flags = {k: [] for k in metrics}
    human = []
    n_pairs = 0
    for ref in np.unique(d['ref']):
        g = np.where((d['ref'] == ref) & (d['code'] == '03'))[0]
        p = np.where((d['ref'] == ref) & (d['code'] == '01'))[0]
        for i, j in itertools.product(g, p):
            if abs(d['psnr'][i] - d['psnr'][j]) > tol:
                continue
            n_pairs += 1
            for k, v in metrics.items():
                flags[k].append(1.0 if v[i] > v[j] else (0.5 if v[i] == v[j] else 0.0))
            human.append(1.0 if d['mos'][j] > d['mos'][i] else (0.5 if d['mos'][j] == d['mos'][i] else 0.0))
    res = {'n_pairs': n_pairs, 'n_refs': int(len(np.unique(d['ref']))), 'tolerance_db': tol,
           'gan_flagged_worse': {k: float(np.mean(v)) for k, v in flags.items()},
           'humans_prefer_psnr_oriented': float(np.mean(human)),
           'mean_l_hal': {'GAN (03)': float(d['l_hal'][d['code'] == '03'].mean()),
                          'PSNR-oriented (01)': float(d['l_hal'][d['code'] == '01'].mean())}}
    return res


def interp_analysis():
    files = {0.0: 'RealESRNet', 0.25: 'RRDB_interp_0.25', 0.5: 'RRDB_interp_0.5',
             0.75: 'RRDB_interp_0.75', 1.0: 'RealESRGAN'}
    recs = {a: json.load(open(os.path.join(CACHE, 'bench_x4', f'{n}.json')))['per_image'] for a, n in files.items()}
    alphas = sorted(files)
    res = {'alpha': alphas, 'mean': {}, 'monotone_increase_fraction': {}}
    keys = {'PSNR': 'psnr', 'LPIPS': 'lpips', 'WaSP-Hal': 'l_hal', 'WaSP-Percept': 'l_percept'}
    for ds in ['Set5', 'Set14', 'BSD100', 'Urban100']:
        res['mean'][ds], res['monotone_increase_fraction'][ds] = {}, {}
        for label, k in list(keys.items()) + [('WaSP', None)]:
            if k is None:
                arr = np.array([wasp_total(recs[a][ds]['l_struct'], recs[a][ds]['l_percept'], recs[a][ds]['l_hal'])
                                for a in alphas])
            else:
                arr = np.array([recs[a][ds][k] for a in alphas])          # [alpha, image]
            res['mean'][ds][label] = [float(x) for x in arr.mean(1)]
            rho = [stats.spearmanr(alphas, arr[:, i])[0] for i in range(arr.shape[1])]
            res['monotone_increase_fraction'][ds][label] = float(np.mean(np.array(rho) > 0))
    return res


def main():
    res = {}
    if '--skip-pipal' not in sys.argv:
        d = pipal_cache()
        res['pipal_matched_psnr'] = pipal_analysis(d)
        res['pipal_unmatched'] = pipal_analysis(d, tol=float('inf'))   # all GAN vs PSNR-oriented pairs
    if os.path.exists(os.path.join(CACHE, 'bench_x4', 'RRDB_interp_0.5.json')):
        res['rrdb_interpolation'] = interp_analysis()
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == '__main__':
    main()
