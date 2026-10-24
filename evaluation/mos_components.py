#!/usr/bin/env python3
"""
mos_components.py — per-pair WaSP branch cache for the MOS datasets.

Stores the raw branches for every (distorted, reference) pair of TID2013, KADID-10K and the
PIPAL 5,000-pair sample, so that any aggregation weights can be scored afterwards without
re-running the networks (mos_analysis.py, hal_validation.py). Per pair:
  l_struct        Branch A of the final metric (LL / 2^J, product-form MS-SSIM)
  l_struct_mean   Branch A with the arithmetic-mean MS-SSIM (earlier revision)
  l_struct_legacy Branch A of the submitted metric (LL band clamped to [0, 1])
  l_percept       Branch B
  l_hal           Branch C with the '<=' smooth mask (final);  l_hal_legacy: strict '<'
  lpips           LPIPS (AlexNet), for reference
  mos             human score (higher = better on all three datasets)
  codes           PIPAL distortion category ('' for TID2013 / KADID-10K)

PIPAL categories (file name A####_CC_##.bmp, CC = category; Gu et al., arXiv:2011.15002):
  00 traditional SR   01 PSNR-oriented SR   02 SR with kernel mismatch   03 GAN-based SR
  04 denoising        05 SR + denoising/compression   06 traditional distortions

Output: results/mos_components_<dataset>.npz

  python evaluation/mos_components.py [--datasets TID2013 KADID-10K PIPAL]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from utils.iqa import IQA_DATASETS, align, img_tensor, iter_pairs
from utils.paths import IQA_DATA, RESULTS_DIR
from wasp import WaSPMetric

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KEYS = ['l_struct_legacy', 'l_struct', 'l_struct_mean', 'l_percept', 'l_hal_legacy', 'l_hal', 'lpips']


@torch.no_grad()
def components(metrics, lpips_fn, d, r):
    wasp, wasp_legacy, wasp_mean = metrics
    sr_ll, sr_hf = wasp.dwt(d)
    hr_ll, hr_hf = wasp.dwt(r)
    return {
        'l_struct_legacy': wasp_legacy.structural_loss(sr_ll, hr_ll).item(),
        'l_struct':        wasp.structural_loss(sr_ll, hr_ll).item(),
        'l_struct_mean':   wasp_mean.structural_loss(sr_ll, hr_ll).item(),
        'l_percept':       wasp.perceptual_loss(d, r).item(),
        'l_hal_legacy':    wasp_legacy.hallucination_penalty(sr_hf, hr_hf).item(),
        'l_hal':           wasp.hallucination_penalty(sr_hf, hr_hf).item(),
        'lpips':           lpips_fn(d * 2 - 1, r * 2 - 1).item(),
    }


def run(name, root, metrics, lpips_fn, out_dir):
    out = {k: [] for k in KEYS}
    mos, names, codes = [], [], []
    t0 = time.time()
    for i, (dk, dp, rp, m) in enumerate(iter_pairs(name, root)):
        r = img_tensor(rp)
        d = align(img_tensor(dp), r)
        for k, v in components(metrics, lpips_fn, d.to(DEVICE), r.to(DEVICE)).items():
            out[k].append(v)
        mos.append(m)
        names.append(dk)
        codes.append(dk.split('_')[1] if name == 'PIPAL' else '')
        if (i + 1) % 1000 == 0:
            print(f'  [{name}] {i + 1} pairs ({time.time() - t0:.0f}s)', flush=True)
    path = os.path.join(out_dir, f'mos_components_{name}.npz')
    np.savez(path, mos=np.array(mos), names=np.array(names), codes=np.array(codes),
             **{k: np.array(v) for k, v in out.items()})
    print(f'  [{name}] saved {len(mos)} pairs -> {path}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=list(IQA_DATASETS), choices=IQA_DATASETS)
    ap.add_argument('--data', default=IQA_DATA)
    ap.add_argument('--out_dir', default=RESULTS_DIR)
    args = ap.parse_args()
    import lpips
    metrics = (WaSPMetric().to(DEVICE).eval(),
               WaSPMetric(legacy=True).to(DEVICE).eval(),
               WaSPMetric(msssim_form='mean').to(DEVICE).eval())
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
    os.makedirs(args.out_dir, exist_ok=True)
    for name in args.datasets:
        run(name, args.data, metrics, lpips_fn, args.out_dir)


if __name__ == '__main__':
    main()
