#!/usr/bin/env python3
"""
fig_spatial_maps.py — spatial decomposition of WaSP, computed from the actual branches.

For Set5 image 3 (x4, official LR) and a few SR outputs, it shows
  * Branch A map: 1 - local SSIM on the normalised coarsest LL band (LL_J / 2^J), first
    MS-SSIM scale, 11x11 Gaussian window (sigma 1.5), upsampled to image size
  * Branch B map: per-pixel weighted L1 distance between channel-normalised AlexNet
    features (the five taps and weights of Branch B), upsampled to image size
  * Branch C map: masked excess energy ReLU(E_sr - E_hr) * M ('<=' mask, q = 0.25),
    averaged over the three levels after upsampling to image size
Each map column shares one colour scale across rows; overlays give the component values
(L_A, L_B, L_C) and the total WaSP score of the final metric.

  python figures/fig_spatial_maps.py        -> results/figures/fig2_spatial_maps.png
"""

from common import DEVICE, infer, out_path  # noqa: I001  (sets the backend and sys.path)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from evaluation.eval_benchmarks import registry
from utils.data import benchmark_pairs
from wasp import WaSPMetric
from wasp.metric import gaussian_kernel

ROWS = ['SwinIR+WaSP', 'RealESRGAN', 'RealESRGAN+WaSP', 'SinSR', 'SinSR+WaSP']
IDX  = 2


def ssim_map(a, b, win=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    k = gaussian_kernel(win, sigma, a.shape[1], a.device)
    f = lambda x: F.conv2d(x, k, padding=win // 2, groups=a.shape[1])
    mu_a, mu_b = f(a), f(b)
    sa, sb, sab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return s.mean(1, keepdim=True)


@torch.no_grad()
def maps(wasp, sr, hr):
    size = hr.shape[-2:]
    up = lambda m, mode='bilinear': F.interpolate(m, size=size, mode=mode,
                                                  align_corners=False if mode == 'bilinear' else None)
    sr_ll, sr_hf = wasp.dwt(sr)
    hr_ll, hr_hf = wasp.dwt(hr)
    scale = 2 ** wasp.dwt.levels
    a = up(1 - ssim_map(sr_ll / scale, hr_ll / scale), 'nearest')
    fs, fh = wasp.feat_extractor(sr), wasp.feat_extractor(hr)
    b = sum(w * up((x - y).abs().mean(1, keepdim=True)) for w, x, y in zip(wasp.feat_weights, fs, fh))
    e = lambda d: torch.sqrt(d['LH'] ** 2 + d['HL'] ** 2 + d['HH'] ** 2 + 1e-8).mean(1, keepdim=True)
    c = 0
    for s_b, h_b in zip(sr_hf, hr_hf):
        es, eh = e(s_b), e(h_b)
        m = (eh <= torch.quantile(eh.flatten(), 0.25)).float()
        c = c + up(F.relu(es - eh) * m, 'nearest')
    c = c / len(hr_hf)
    score, l_a, l_b, l_c = wasp(sr, hr)
    return [x[0, 0].cpu().numpy() for x in (a, b, c)], [v.item() for v in (l_a, l_b, l_c, score)]


def overlay(ax, txt, loc='tl'):
    """Small white label in a translucent black box inside the panel."""
    x, y, ha = (0.03, 0.97, 'left') if loc == 'tl' else (0.97, 0.97, 'right')
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=10.5, color='white', fontweight='bold',
            va='top', ha=ha, bbox=dict(facecolor='black', alpha=0.45, pad=1.5, edgecolor='none'))


def plot(hr_np, rows, vmax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    n = len(rows)
    head = 0.32                                     # inches reserved for column headers
    W, H = 19.0, 3.42 * n + head
    fig, ax = plt.subplots(n, 5, figsize=(W, H), dpi=220)
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0.003, right=0.985, top=1 - head / H, bottom=0.003, wspace=0.14, hspace=0.03)
    heads = ['HR Reference', 'SR Output', r'$\mathcal{L}_{\mathrm{struct}}$ map',
             r'$\mathcal{L}_{\mathrm{percept}}$ map', r'$\mathcal{L}_{\mathrm{hal}}$ map']
    labels = ['WaSP-A', 'WaSP-B', 'WaSP-C']
    for i, (name, sr_np, m, v) in enumerate(rows):
        for j in range(5):
            a = ax[i, j]
            a.set_facecolor('none')
            cax = make_axes_locatable(a).append_axes('right', size='4.5%', pad=0.03)
            if j == 0:
                a.imshow(hr_np)
            elif j == 1:
                a.imshow(sr_np)
                overlay(a, f'{name}\nWaSP={v[3]:.4f}', 'tl')
            else:
                k = j - 2
                im = a.imshow(m[k], cmap='inferno', vmin=0, vmax=vmax[k])
                overlay(a, f'{labels[k]}={v[k]:.4f}', 'tr')
                cb = fig.colorbar(im, cax=cax)
                cb.locator = matplotlib.ticker.MaxNLocator(nbins=4)
                cb.update_ticks()
                cb.ax.tick_params(labelsize=7, length=2, pad=1)
                cb.outline.set_linewidth(0.4)
            if j < 2:
                cax.set_visible(False)       # keeps every image the same size
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if i == 0:
                a.set_title(heads[j], fontsize=13, pad=4)
    fig.savefig(out_path('fig2_spatial_maps.png'), dpi=220, transparent=True)
    plt.close(fig)


def main():
    wasp = WaSPMetric().to(DEVICE).eval()
    reg = registry(4)
    hp, lp = benchmark_pairs('Set5', 4)[IDX]
    hr = TF.to_tensor(Image.open(hp).convert('RGB')).unsqueeze(0).to(DEVICE)
    lr = TF.to_tensor(Image.open(lp).convert('RGB')).unsqueeze(0).to(DEVICE)
    rows = []
    for name in ROWS:
        model = reg[name][0]()
        sr = infer(model, lr, hr)
        m, v = maps(wasp, sr, hr)
        rows.append((name, sr[0].permute(1, 2, 0).cpu().numpy(), m, v))
        print(f'  {name}: L_A={v[0]:.4f} L_B={v[1]:.4f} L_C={v[2]:.4f} WaSP={v[3]:.4f}', flush=True)
        del model
        torch.cuda.empty_cache()
    vmax = [np.percentile(np.concatenate([r[2][k].ravel() for r in rows]), 99.5) for k in range(3)]
    plot(hr[0].permute(1, 2, 0).cpu().numpy(), rows, vmax)
    print(f"saved -> {out_path('fig2_spatial_maps.png')}")


if __name__ == '__main__':
    main()
