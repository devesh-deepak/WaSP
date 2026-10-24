#!/usr/bin/env python3
"""
mos_baselines.py — baseline IQA metrics on the MOS datasets (the comparison table of the paper).

Same (distorted, reference) pairs as mos_components.py: TID2013 (3,000), KADID-10K (10,125)
and the PIPAL 5,000-pair sample (seed 42); distorted images are resized to the reference size
(bilinear) if they differ.
  PSNR, SSIM   scikit-image on RGB in [0, 1] (PSNR capped at 100 dB)
  LPIPS        lpips package, AlexNet
  DISTS, NLPD, VIF (full-reference), CLIP-IQA, MUSIQ (no-reference): pyiqa defaults
Correlations are sign-corrected (each pyiqa metric's lower_better) so that positive means
agreement with MOS. Per-pair scores are cached, so the correlations can be recomputed.

Output: results/mos_baselines_<dataset>.npz, results/mos_baselines.json

  python evaluation/mos_baselines.py
  python evaluation/mos_baselines.py --datasets TID2013 --metrics psnr ssim lpips dists
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from utils.iqa import IQA_DATASETS, align, img_tensor, iter_pairs
from utils.paths import IQA_DATA, RESULTS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
METRICS = ['psnr', 'ssim', 'lpips', 'dists', 'nlpd', 'vif', 'clipiqa', 'musiq']
PYIQA_FR = {'dists', 'nlpd', 'vif', 'pieapp'}
PYIQA_NR = {'clipiqa', 'musiq'}


def build(names):
    """{name: (fn(d, r) -> float, lower_better)} with d, r [1, 3, H, W] on DEVICE."""
    out = {}
    for n in names:
        if n == 'psnr':
            from skimage.metrics import peak_signal_noise_ratio as sk_psnr

            def psnr(d, r):
                v = sk_psnr(_np(r), _np(d), data_range=1.0)
                return min(v, 100.0) if np.isfinite(v) else 100.0
            out[n] = (psnr, False)
        elif n == 'ssim':
            from skimage.metrics import structural_similarity as sk_ssim
            out[n] = ((lambda d, r: sk_ssim(_np(r), _np(d), data_range=1.0, channel_axis=2)), False)
        elif n == 'lpips':
            import lpips
            net = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
            out[n] = ((lambda d, r, net=net: net(d * 2 - 1, r * 2 - 1).item()), True)
        elif n in PYIQA_FR | PYIQA_NR:
            import pyiqa
            m = pyiqa.create_metric(n, device=DEVICE)
            fn = ((lambda d, r, m=m: m(d, r).item()) if n in PYIQA_FR
                  else (lambda d, r, m=m: m(d).item()))
            out[n] = (fn, bool(m.lower_better))
        else:
            raise ValueError(f'unknown metric {n!r}')
    return out


def _np(t):
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()


def correlations(scores, mos, lower_better):
    s, m = np.asarray(scores, dtype=np.float64), np.asarray(mos, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(m)
    s, m = s[ok], m[ok]
    if lower_better:
        s = -s
    return {'SRCC': float(stats.spearmanr(s, m)[0]), 'PLCC': float(stats.pearsonr(s, m)[0]),
            'KRCC': float(stats.kendalltau(s, m)[0]), 'n': int(len(s))}


@torch.no_grad()
def run(name, root, metrics, out_dir):
    scores = {k: [] for k in metrics}
    mos, t0 = [], time.time()
    for i, (_, dp, rp, m) in enumerate(iter_pairs(name, root)):
        r = img_tensor(rp)
        d = align(img_tensor(dp), r)
        d, r = d.to(DEVICE), r.to(DEVICE)
        for k, (fn, _) in metrics.items():
            scores[k].append(fn(d, r))
        mos.append(m)
        if (i + 1) % 1000 == 0:
            print(f'  [{name}] {i + 1} pairs ({time.time() - t0:.0f}s)', flush=True)
    np.savez(os.path.join(out_dir, f'mos_baselines_{name}.npz'), mos=np.array(mos),
             **{k: np.array(v) for k, v in scores.items()})
    return {k: correlations(v, mos, metrics[k][1]) for k, v in scores.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=list(IQA_DATASETS), choices=IQA_DATASETS)
    ap.add_argument('--metrics', nargs='+', default=METRICS,
                    help=f'any of {METRICS + sorted(PYIQA_FR | PYIQA_NR - set(METRICS))}')
    ap.add_argument('--data', default=IQA_DATA)
    ap.add_argument('--out_dir', default=RESULTS_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    metrics = build(args.metrics)
    out_json = os.path.join(args.out_dir, 'mos_baselines.json')
    res = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for name in args.datasets:
        res.setdefault(name, {}).update(run(name, args.data, metrics, args.out_dir))
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=2)
    print(f'{"metric":10s}' + ''.join(f'{d:>12s}' for d in res) + '   (SRCC)')
    for k in args.metrics:
        print(f'{k:10s}' + ''.join(f'{res[d][k]["SRCC"]:12.4f}' for d in res if k in res[d]))
    print(f'saved -> {out_json}')


if __name__ == '__main__':
    main()
