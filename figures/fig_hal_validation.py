#!/usr/bin/env python3
"""
fig_hal_validation.py — qualitative figure for the controlled hallucination test.

One reference from the test set of evaluation/hal_validation.py; each distortion is matched
to 31 dB PSNR with the same generators and bisection. Bottom row: the finest-level
smooth-region mask and the masked excess-energy map ReLU(E_sr - E_hr) * M of the
corrected metric, with the image's L_C value.

  python figures/fig_hal_validation.py [image index, default 34 = BSD100 img_021]
Writes results/figures/fig_hal_validation.png
"""

from common import out_path  # noqa: I001  (sets the backend and sys.path)

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import evaluation.hal_validation as H
from wasp import WaSPMetric

OUT = out_path('fig_hal_validation.png')
SHOW = ['hallucinate', 'noise', 'sharpen', 'ringing', 'blur']


@torch.no_grad()
def main(idx):
    wasp = WaSPMetric().to(H.DEVICE).eval()
    files, refs = H.load_refs()
    y = refs[idx].to(H.DEVICE)
    donor = F.interpolate(refs[(idx + 1) % len(refs)].to(H.DEVICE), size=y.shape[-2:],
                          mode='bicubic', align_corners=False).clamp(0, 1)
    tex = donor - H.gblur(donor, 1.0)
    tex = tex / tex.pow(2).mean().sqrt()
    m_smooth, _ = H.region_masks(y)
    fam = H.make_family(y, tex, m_smooth, seed=idx)
    _, hr_hf = wasp.dwt(y)
    e = lambda b: torch.sqrt(b['LH'] ** 2 + b['HL'] ** 2 + b['HH'] ** 2 + 1e-8).mean(1, keepdim=True)
    hr_e = e(hr_hf[0])
    mask = (hr_e <= torch.quantile(hr_e.flatten(), 0.25)).float()

    imgs, maps, vals = [], [], []
    for cond in SHOW:
        fn, hi = fam[cond]
        d = H.match_psnr(y, fn, hi, 31.0)
        _, sr_hf = wasp.dwt(d)
        exc = F.relu(e(sr_hf[0]) - hr_e) * mask
        imgs.append(d[0].permute(1, 2, 0).cpu().numpy())
        maps.append(exc[0, 0].cpu().numpy())
        vals.append(wasp.hallucination_penalty(sr_hf, hr_hf).item())
    vmax = np.percentile(np.concatenate([m.flatten() for m in maps]), 99.5)

    h, w = y.shape[-2:]
    fig, ax = plt.subplots(2, 6, figsize=(13.6, 2 * 2.27 * h / w + 1.2), dpi=200)
    ax[0, 0].imshow(y[0].permute(1, 2, 0).cpu().numpy()); ax[0, 0].set_title('reference', fontsize=12)
    ax[1, 0].imshow(mask[0, 0].cpu().numpy(), cmap='gray'); ax[1, 0].set_title('smooth-region mask', fontsize=11)
    for j, cond in enumerate(SHOW, start=1):
        ax[0, j].imshow(imgs[j - 1]); ax[0, j].set_title(cond + ' (31 dB)', fontsize=12)
        ax[1, j].imshow(maps[j - 1], cmap='hot', vmin=0, vmax=vmax)
        ax[1, j].set_title(f'$\\mathcal{{L}}_C$ excess energy\n$\\mathcal{{L}}_C$={vals[j-1]:.4f}', fontsize=11)
    for a in ax.flat:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, facecolor='white')
    print(f'{os.path.basename(files[idx])}: ' + ', '.join(f'{c}={v:.4f}' for c, v in zip(SHOW, vals)))
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 34)
