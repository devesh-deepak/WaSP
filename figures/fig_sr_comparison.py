#!/usr/bin/env python3
"""
fig_sr_comparison.py — qualitative x4 comparison before and after WaSP fine-tuning (Fig. 1):
HR, five pretrained models and their +WaSP versions, with PSNR / SSIM / WaSP and a zoomed crop.

  python figures/fig_sr_comparison.py                      # Set5 image 3 (the paper's figure)
  python figures/fig_sr_comparison.py --dataset Urban100 --index 2
"""

from common import DEVICE, STYLE, infer, overlay, savefig, t2np  # noqa: I001  (sets the backend and sys.path)

import argparse

import matplotlib.pyplot as plt
import numpy as np

from evaluation.eval_benchmarks import registry
from utils.data import BENCHMARKS, benchmark_pairs, load_benchmark_pair
from utils.metrics import calculate_psnr, calculate_ssim
from wasp import WaSPMetric

SHOWCASE = ['RCAN', 'EDSR', 'RealESRGAN', 'SwinIR', 'SinSR']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='Set5', choices=BENCHMARKS)
    ap.add_argument('--index', type=int, default=2, help='image index (sorted file names)')
    args = ap.parse_args()
    plt.rcParams.update(STYLE)

    wasp = WaSPMetric().to(DEVICE).eval()
    reg = registry(4)
    hp, lp = benchmark_pairs(args.dataset, 4)[args.index]
    lr_t, hr_t = load_benchmark_pair(hp, lp, 4, DEVICE)
    hr_np = t2np(hr_t)
    _, _, H, W = hr_t.shape
    cy, cx = H // 3, W // 3
    ch, cw = min(96, H // 3), min(96, W // 3)

    col_keys = ['HR'] + SHOWCASE + [m + '+WaSP' for m in SHOWCASE]
    n_cols = len(col_keys)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 6.0))
    fig.patch.set_alpha(0)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.05, wspace=0.04)

    for ci, key in enumerate(col_keys):
        ax_top, ax_bot = axes[0, ci], axes[1, ci]
        for ax in (ax_top, ax_bot):
            ax.axis('off')
            ax.set_facecolor('none')
        if key == 'HR':
            img = hr_np
            ax_top.set_title('HR Reference', fontsize=9, color='black', pad=3, fontweight='bold')
        else:
            model = reg[key][0]()
            sr = infer(model, lr_t, hr_t)
            del model
            img = t2np(sr)
            p = min(calculate_psnr(sr, hr_t).item(), 100.0)
            s = calculate_ssim(sr, hr_t).item()
            w = wasp(sr, hr_t)[0].item()
            ax_top.set_title(key, fontsize=9, color='black', pad=3,
                             fontweight='bold' if '+WaSP' in key else 'normal')
            overlay(ax_top, f'PSNR {p:.2f} dB\nSSIM {s:.4f}\nWaSP {w:.4f}', 'tr')
            print(f'  {key:16s} PSNR {p:.2f}  SSIM {s:.4f}  WaSP {w:.4f}', flush=True)
        ax_top.imshow(np.clip(img, 0, 1))
        ax_top.add_patch(plt.Rectangle((cx, cy), cw, ch, lw=1.8, edgecolor='#FF6F00', facecolor='none'))
        ax_bot.imshow(np.clip(img[cy:cy + ch, cx:cx + cw], 0, 1), interpolation='nearest')

    sep_x = (len(SHOWCASE) + 1) / n_cols
    fig.add_artist(plt.Line2D([sep_x, sep_x], [0.02, 0.98], transform=fig.transFigure,
                              color='#FF6F00', lw=1.4, linestyle='--'))
    fig.text((1 + len(SHOWCASE) / 2) / n_cols, 0.97, 'Before WaSP Fine-tuning',
             ha='center', fontsize=9, fontweight='bold', color='black')
    fig.text((1 + 1.5 * len(SHOWCASE)) / n_cols, 0.97, 'After WaSP Fine-tuning',
             ha='center', fontsize=9, fontweight='bold', color='black')
    plt.suptitle(f'Qualitative SR Comparison — {args.dataset} x4 (Image #{args.index + 1})',
                 fontsize=11, fontweight='bold', color='black', y=1.01)
    savefig(f'fig1_sr_comparison_{args.dataset.lower()}.png')


if __name__ == '__main__':
    main()
