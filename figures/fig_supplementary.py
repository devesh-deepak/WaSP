#!/usr/bin/env python3
"""
fig_supplementary.py — supplementary qualitative figures built around SinSR (x4):

  S1  loss-control strip    one SinSR fine-tune per auxiliary loss (Set5 image 3)
  S2  branch-ablation strip L_A / L_B / L_C / L_A+L_B / full WaSP (Set5 image 3)
  S3  metric disagreement   Urban100 images where PSNR prefers bicubic but WaSP prefers SinSR

SinSR samples prior noise, so the seed is pinned before every forward pass.

  python figures/fig_supplementary.py [--figs S1 S2 S3]
Writes results/figures/figS*.png and results/metric_disagreement.json
"""

from common import DEVICE, infer, overlay, savefig, t2np  # noqa: I001  (sets the backend and sys.path)

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import build_model, checkpoint_path, load_checkpoint
from utils.data import benchmark_pairs, to_tensor
from utils.metrics import calculate_psnr, calculate_ssim
from utils.paths import RESULTS_DIR
from wasp import WaSPMetric


def sinsr(variant=None):
    """Pretrained SinSR, or a released fine-tuned variant ('wasp', 'ctrl_l1', 'abla_la', ...)."""
    m = build_model('sinsr', 4, DEVICE)
    return load_checkpoint(m, checkpoint_path('sinsr', 4, variant)) if variant else m


def pair(ds, idx):
    """(HR, LR) tensors; the HR is not cropped (the SR output is resized to it if needed)."""
    hp, lp = benchmark_pairs(ds, 4)[idx]
    return to_tensor(hp).to(DEVICE), to_tensor(lp).to(DEVICE)


# ══════════════════════════════════════════════════════════════════════════
# Shared strip renderer for S1 / S2
# ══════════════════════════════════════════════════════════════════════════
def variant_strip(variants, ds, idx, title, out_name, crop_frac=(0.36, 0.30)):
    """variants: list of (label, checkpoint variant or None). Full image + zoomed crop."""
    wasp = WaSPMetric().to(DEVICE).eval()

    hr_t, lr_t = pair(ds, idx)
    H, W = hr_t.shape[-2:]
    cy, cx = int(H * crop_frac[0]), int(W * crop_frac[1])
    ch = cw = min(96, H // 3, W // 3)

    cols = [('HR Reference', t2np(hr_t), None)]
    for label, variant in variants:
        m = sinsr(variant)
        sr = infer(m, lr_t, hr_t)
        with torch.no_grad():
            w = wasp(sr, hr_t)[0].item()
        cols.append((label, t2np(sr), (calculate_psnr(sr, hr_t).item(), calculate_ssim(sr, hr_t).item(), w)))
        del m
        torch.cuda.empty_cache()

    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(2.45 * n, 5.4))
    plt.subplots_adjust(wspace=0.03, hspace=0.04)
    for i, (label, img, sc) in enumerate(cols):
        at, ab = axes[0, i], axes[1, i]
        at.imshow(img.clip(0, 1)); at.axis('off')
        at.set_title(label, fontsize=9, pad=3,
                     fontweight='bold' if 'WaSP' in label or i == 0 else 'normal')
        at.add_patch(plt.Rectangle((cx, cy), cw, ch, lw=1.6,
                                   edgecolor='#FF6F00', facecolor='none'))
        if sc:
            overlay(at, f'PSNR {sc[0]:.2f}\nSSIM {sc[1]:.4f}\nWaSP {sc[2]:.4f}', 'tr', fontsize=7.5, alpha=0.55, pad=1.4)
        ab.imshow(img[cy:cy + ch, cx:cx + cw].clip(0, 1), interpolation='nearest'); ab.axis('off')
    plt.suptitle(title, fontsize=11, fontweight='bold', y=0.99)
    savefig(out_name, dpi=200)


# ══════════════════════════════════════════════════════════════════════════
# S3 – where PSNR and WaSP disagree
# ══════════════════════════════════════════════════════════════════════════
def fig_s3_disagreement(ds='Urban100', n_show=3, max_scan=100):
    """Find images where bicubic beats SinSR on PSNR but LOSES on WaSP."""
    print(f'[S3] scanning {ds} for PSNR/WaSP rank inversions ...')
    wasp = WaSPMetric().to(DEVICE).eval()
    model = sinsr(None)                   # pretrained SinSR

    hrs, lrs = map(list, zip(*benchmark_pairs(ds, 4)[:max_scan]))
    cands = []
    for i, (hp, lp) in enumerate(zip(hrs, lrs)):
        hr, lr = to_tensor(hp).to(DEVICE), to_tensor(lp).to(DEVICE)
        bic = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        if bic.shape[-2:] != hr.shape[-2:]:
            bic = F.interpolate(bic, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        sr = infer(model, lr, hr)
        with torch.no_grad():
            wb, ws = wasp(bic, hr)[0].item(), wasp(sr, hr)[0].item()
        pb, ps = calculate_psnr(bic, hr).item(), calculate_psnr(sr, hr).item()
        # PSNR prefers bicubic, WaSP prefers SinSR -> the metrics disagree
        if pb > ps and ws < wb:
            cands.append({'i': i, 'psnr_b': pb, 'psnr_s': ps, 'wasp_b': wb,
                          'wasp_s': ws, 'gap': (pb - ps) + 40 * (wb - ws)})
        del hr, lr, bic, sr
    cands.sort(key=lambda c: -c['gap'])
    print(f'  {len(cands)}/{len(hrs)} images show a rank inversion')
    if not cands:
        print('  none found — skipping S3'); return
    sel = cands[:n_show]

    fig, axes = plt.subplots(len(sel), 3, figsize=(9.6, 3.35 * len(sel)))
    if len(sel) == 1: axes = axes[np.newaxis, :]
    plt.subplots_adjust(wspace=0.03, hspace=0.12)
    for r, c in enumerate(sel):
        hr, lr = to_tensor(hrs[c['i']]).to(DEVICE), to_tensor(lrs[c['i']]).to(DEVICE)
        bic = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        if bic.shape[-2:] != hr.shape[-2:]:
            bic = F.interpolate(bic, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        sr = infer(model, lr, hr)
        H, W = hr.shape[-2:]
        cy, cx = H // 3, W // 3
        s = min(180, H // 3, W // 3)
        for k, (lab, img, txt) in enumerate([
                ('HR Reference', hr, None),
                ('Bicubic $\\times$4', bic,
                 f"PSNR {c['psnr_b']:.2f} (better)\nWaSP {c['wasp_b']:.4f} (worse)"),
                ('SinSR', sr,
                 f"PSNR {c['psnr_s']:.2f} (worse)\nWaSP {c['wasp_s']:.4f} (better)")]):
            ax = axes[r, k]
            ax.imshow(t2np(img).clip(0, 1)[cy:cy + s, cx:cx + s]); ax.axis('off')
            if r == 0:
                ax.set_title(lab, fontsize=10, fontweight='bold', pad=4)
            if txt: overlay(ax, txt, 'tr', fontsize=7.5, alpha=0.55, pad=1.4)
        axes[r, 0].text(-0.04, 0.5, f"{ds} #{c['i']+1}", transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=9, fontweight='bold')
        del hr, lr, bic, sr
    plt.suptitle('Where PSNR and WaSP Disagree: PSNR prefers bicubic, WaSP prefers SinSR',
                 fontsize=11.5, fontweight='bold', y=0.995)
    savefig('figS3_metric_disagreement.png', dpi=200)
    with open(os.path.join(RESULTS_DIR, 'metric_disagreement.json'), 'w') as f:
        json.dump({'dataset': ds, 'n_scanned': len(hrs), 'n_inversions': len(cands),
                   'shown': sel}, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figs', nargs='+', default=['S1', 'S2', 'S3'], choices=['S1', 'S2', 'S3'])
    args = ap.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': 'black',
                         'axes.titlecolor': 'black', 'figure.facecolor': 'none', 'savefig.transparent': True})
    if 'S1' in args.figs:
        print('[S1] loss-control strip ...')
        variant_strip([('SinSR (base)', None), ('+$\\ell_1$', 'ctrl_l1'), ('+LPIPS', 'ctrl_lpips'),
                       ('+DISTS', 'ctrl_dists'), ('+MS-SSIM', 'ctrl_msssim'), ('+WaSP', 'wasp')],
                      'Set5', 2, 'SinSR Fine-tuning: Auxiliary-Loss Control Study (Set5 $\\times$4)',
                      'figS1_sinsr_loss_controls.png')
    if 'S2' in args.figs:
        print('[S2] branch-ablation strip ...')
        variant_strip([('SinSR (base)', None), ('+$\\mathcal{L}_A$', 'abla_la'), ('+$\\mathcal{L}_B$', 'abla_lb'),
                       ('+$\\mathcal{L}_C$', 'abla_lc'), ('+$\\mathcal{L}_A$+$\\mathcal{L}_B$', 'abla_lalb'),
                       ('+WaSP (full)', 'wasp')],
                      'Set5', 2, 'SinSR Fine-tuning: WaSP Branch Ablation (Set5 $\\times$4)',
                      'figS2_sinsr_branch_ablation.png')
    if 'S3' in args.figs:
        fig_s3_disagreement()


if __name__ == '__main__':
    main()
