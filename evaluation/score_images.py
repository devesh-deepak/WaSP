#!/usr/bin/env python3
"""
score_images.py — WaSP (and Y-channel PSNR / SSIM) of a folder of SR images against references.

Images are paired by file name; an SR image whose size differs from its reference is resized to
it (bilinear). Writes one CSV row per image and prints the means.

  python evaluation/score_images.py --sr results/my_model/Set5 --hr datasets/sr/Set5/HR
  python evaluation/score_images.py --sr SR_DIR --hr HR_DIR --out scores.csv --crop_border 4
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from utils.data import to_tensor
from utils.iqa import align
from utils.metrics import calculate_psnr, calculate_ssim
from wasp import WaSPMetric

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sr', required=True, help='folder of super-resolved images')
    ap.add_argument('--hr', required=True, help='folder of reference images (same file names)')
    ap.add_argument('--out', default=None, help='CSV file (default: <sr>/wasp_scores.csv)')
    ap.add_argument('--crop_border', type=int, default=4, help='border excluded from PSNR')
    ap.add_argument('--legacy', action='store_true', help='the originally submitted metric')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    names = sorted(f for f in os.listdir(args.sr) if f.lower().endswith(EXTS))
    missing = [f for f in names if not os.path.exists(os.path.join(args.hr, f))]
    if missing or not names:
        sys.exit(f'{len(names)} SR images; no reference with the same name for: {missing[:5]}')

    wasp = WaSPMetric(legacy=args.legacy).to(device).eval()
    rows = []
    with torch.no_grad():
        for f in names:
            hr = to_tensor(os.path.join(args.hr, f)).to(device)
            sr = align(to_tensor(os.path.join(args.sr, f)).to(device), hr).clamp(0, 1)
            score, l_s, l_p, l_h = wasp(sr, hr)
            rows.append({'image': f, 'wasp': score.item(), 'l_struct': l_s.item(), 'l_percept': l_p.item(),
                         'l_hal': l_h.item(), 'psnr_y': calculate_psnr(sr, hr, args.crop_border).item(),
                         'ssim_y': calculate_ssim(sr, hr).item()})

    out = args.out or os.path.join(args.sr, 'wasp_scores.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'image'}
    print(f'{len(rows)} images  ' + '  '.join(f'{k}={v:.4f}' for k, v in means.items()))
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
