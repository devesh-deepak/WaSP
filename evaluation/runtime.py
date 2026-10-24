#!/usr/bin/env python3
"""
runtime.py — forward-pass time of WaSP, its branches and LPIPS (one image pair, batch 1,
256x256 and 512x512, fp32, no gradient; median over 300 runs after 50 warm-up runs).

Output: results/runtime.json

  python evaluation/runtime.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lpips
import numpy as np
import torch

from utils.paths import RESULTS_DIR
from wasp import WaSPMetric

DEVICE, WARMUP, N = 'cuda', 50, 300


@torch.no_grad()
def time_ms(f):
    for _ in range(WARMUP):
        f()
    t = []
    for _ in range(N):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        f()
        torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t))


def main():
    wasp = WaSPMetric().to(DEVICE).eval()
    lp = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
    res = {'device': torch.cuda.get_device_name(0), 'ms_per_image': {}}
    for s in (256, 512):
        x, y = torch.rand(1, 3, s, s, device=DEVICE), torch.rand(1, 3, s, s, device=DEVICE)

        def branches_ac():
            sr_ll, sr_hf = wasp.dwt(x); hr_ll, hr_hf = wasp.dwt(y)
            wasp.structural_loss(sr_ll, hr_ll); wasp.hallucination_penalty(sr_hf, hr_hf)
        res['ms_per_image'][f'{s}x{s}'] = {
            'WaSP (full)': time_ms(lambda: wasp(x, y)),
            'WaSP: Branch B (perceptual)': time_ms(lambda: wasp.perceptual_loss(x, y)),
            'WaSP: Branches A+C (wavelet)': time_ms(branches_ac),
            'LPIPS (alex)': time_ms(lambda: lp(x * 2 - 1, y * 2 - 1)),
        }
        print(s, {k: round(v, 2) for k, v in res['ms_per_image'][f'{s}x{s}'].items()}, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS_DIR, 'runtime.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
