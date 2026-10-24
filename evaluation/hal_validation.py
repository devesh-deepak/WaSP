#!/usr/bin/env python3
"""
hal_validation.py — controlled validation of the hallucination branch and the trade-off
between hallucination sensitivity and MOS correlation across WaSP weights.

Stage 1 (cached): a controlled, matched-PSNR distortion set built from 44 clean
references (all 14 Set14 HR images + the first 30 BSD100 HR images). For every
reference and every target PSNR (31 and 34 dB) it creates

  hallucinate   natural high-frequency texture (high-pass of another reference)
                added only inside the reference's smooth regions
  noise_smooth  white Gaussian noise added only inside the smooth regions
  noise         white Gaussian noise over the whole image
  sharpen       unsharp masking (Gaussian, sigma 1.5)
  ringing       blend towards an ideal low-pass image (Gibbs oscillation at edges)
  edge_shift    sub-pixel translation (bilinear)
  blur          Gaussian blur

Each distortion strength is set by bisection so that RGB PSNR equals the target.
Smooth regions = pixels whose 7x7 local luminance std is below its 30th percentile;
textured regions = above its 70th percentile (both softened, Gaussian sigma 1.5).
A region test adds the same texture, at the same amplitude, either to the smooth or
to the textured region. Per image, the raw WaSP components are cached for the
legacy (submitted) and corrected metric, together with LPIPS.

Stage 2 reports
  * mean L_hal per distortion and target PSNR (corrected and legacy)
  * the smooth/textured region ratio of L_hal
  * hallucination-ranking accuracy over a (lambda_p, lambda_h) grid with lambda_s = 1:
    fraction of (image, PSNR level, control) triples in which the composite scores the
    hallucinated image as worse than the control (controls: noise, sharpen, ringing,
    edge_shift, blur; ties count 0.5), next to the mean SRCC with MOS
    (from mos_components.py)

Outputs: results/hal_controlled.npz (stage 1), results/hal_validation.json (stage 2)

  python evaluation/hal_validation.py             # builds the cache if it is missing
  python evaluation/hal_validation.py --rebuild
"""

import glob
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

from utils.paths import RESULTS_DIR, SR_DATA
from wasp import WaSPMetric

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA     = SR_DATA
NPZ      = os.path.join(RESULTS_DIR, 'hal_controlled.npz')
OUT_JSON = os.path.join(RESULTS_DIR, 'hal_validation.json')
TARGETS  = [31.0, 34.0]
CONDS    = ['hallucinate', 'noise_smooth', 'noise', 'sharpen', 'ringing', 'edge_shift', 'blur']
CONTROLS = ['noise', 'sharpen', 'ringing', 'edge_shift', 'blur']
REGION_AMP = 0.05                     # texture amplitude (RMS) for the region test
SCALES   = (0.30, 0.05, 0.01)
LP_GRID  = [0.25, 0.5, 1.0, 2.0, 4.0]
LH_GRID  = [0.0, 0.05, 0.1, 0.2, 0.5]
DEFAULT  = (1.0, 0.5, 0.1)
KEYS     = ['l_struct_legacy', 'l_struct', 'l_struct_mean', 'l_percept', 'l_hal_legacy', 'l_hal', 'lpips', 'psnr']


# ── Distortions ──────────────────────────────────────────────────────────────

def gblur(x, sigma):
    if sigma <= 1e-3:
        return x
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(x, [k, k], [sigma, sigma])


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def region_masks(y):
    lum = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3])
    pad = F.pad(lum, (3, 3, 3, 3), mode='reflect')
    mu  = F.avg_pool2d(pad, 7, stride=1)
    var = F.avg_pool2d(pad ** 2, 7, stride=1) - mu ** 2
    std = var.clamp_min(0).sqrt().flatten()
    smooth   = (std < torch.quantile(std, 0.30)).float().view_as(lum)
    textured = (std > torch.quantile(std, 0.70)).float().view_as(lum)
    return gblur(smooth, 1.5).clamp(0, 1), gblur(textured, 1.5).clamp(0, 1)


def ideal_lowpass(y, frac=0.30):
    H, W = y.shape[-2:]
    fy = torch.fft.fftfreq(H, device=y.device).abs().view(H, 1)
    fx = torch.fft.fftfreq(W, device=y.device).abs().view(1, W)
    mask = ((fy ** 2 + fx ** 2).sqrt() <= frac * 0.5).float()
    return torch.fft.ifft2(torch.fft.fft2(y) * mask).real


def shift(y, d):
    _, _, H, W = y.shape
    theta = torch.tensor([[1, 0, 2 * d / W], [0, 1, 2 * d / H]], dtype=y.dtype,
                         device=y.device).unsqueeze(0)
    grid = F.affine_grid(theta, y.shape, align_corners=False)
    return F.grid_sample(y, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def make_family(y, tex, m, seed):
    """Return {condition: (fn(strength) -> image, max_strength)}."""
    # CUDA random streams depend on the device (number of SMs), so on another GPU the two
    # noise conditions get a different realisation; results/hal_controlled.npz is the paper's.
    g = torch.Generator(device=y.device).manual_seed(seed)
    noise = torch.randn(y.shape, generator=g, device=y.device)
    lp = ideal_lowpass(y)
    hp = y - gblur(y, 1.5)
    c = lambda x: x.clamp(0, 1)
    return {
        'hallucinate':  (lambda a: c(y + a * tex * m),   2.0),
        'noise_smooth': (lambda a: c(y + a * noise * m), 2.0),
        'noise':        (lambda a: c(y + a * noise),     1.0),
        'sharpen':      (lambda a: c(y + a * hp),        40.0),
        'ringing':      (lambda a: c(y + a * (lp - y)),  4.0),
        'edge_shift':   (lambda a: c(shift(y, a)),       4.0),
        'blur':         (lambda a: c(gblur(y, a)),       8.0),
    }


def match_psnr(y, fn, hi, target, iters=40):
    if psnr(fn(hi), y) > target:
        return None                       # target not reachable within range
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if psnr(fn(mid), y) > target:
            lo = mid
        else:
            hi = mid
    return fn(hi)


# ── Stage 1 ──────────────────────────────────────────────────────────────────

def load_refs():
    files  = sorted(glob.glob(f'{DATA}/Set14/image_SRF_4/*_HR.png'))
    files += sorted(glob.glob(f'{DATA}/BSD100/image_SRF_4/*_HR.png'))[:30]
    return files, [TF.to_tensor(Image.open(f).convert('RGB')).unsqueeze(0) for f in files]


@torch.no_grad()
def components(metrics, lp_fn, d, y, hr_ll, hr_hf):
    wasp, wasp_legacy, wasp_mean = metrics
    sr_ll, sr_hf = wasp.dwt(d)
    return {'l_struct_legacy': wasp_legacy.structural_loss(sr_ll, hr_ll).item(),
            'l_struct':        wasp.structural_loss(sr_ll, hr_ll).item(),
            'l_struct_mean':   wasp_mean.structural_loss(sr_ll, hr_ll).item(),
            'l_percept':       wasp.perceptual_loss(d, y).item(),
            'l_hal_legacy':    wasp_legacy.hallucination_penalty(sr_hf, hr_hf).item(),
            'l_hal':           wasp.hallucination_penalty(sr_hf, hr_hf).item(),
            'lpips':           lp_fn(d * 2 - 1, y * 2 - 1).item(),
            'psnr':            psnr(d, y)}


@torch.no_grad()
def build_cache():
    import lpips
    wasp = WaSPMetric().to(DEVICE).eval()
    metrics = (wasp, WaSPMetric(legacy=True).to(DEVICE).eval(),
               WaSPMetric(msssim_form='mean').to(DEVICE).eval())
    lp_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
    files, refs = load_refs()
    arr = {k: np.full((len(refs), len(TARGETS), len(CONDS)), np.nan) for k in KEYS}
    reg = {k: np.full((len(refs), 2), np.nan) for k in KEYS}      # [smooth, textured]
    for i, y in enumerate(refs):
        y = y.to(DEVICE)
        donor = refs[(i + 1) % len(refs)].to(DEVICE)
        donor = F.interpolate(donor, size=y.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
        tex = donor - gblur(donor, 1.0)
        tex = tex / tex.pow(2).mean().sqrt()
        m_smooth, m_tex = region_masks(y)
        fam = make_family(y, tex, m_smooth, seed=i)
        hr_ll, hr_hf = wasp.dwt(y)
        for t, target in enumerate(TARGETS):
            for c, cond in enumerate(CONDS):
                fn, hi = fam[cond]
                d = match_psnr(y, fn, hi, target)
                if d is None:
                    continue
                for k, v in components(metrics, lp_fn, d, y, hr_ll, hr_hf).items():
                    arr[k][i, t, c] = v
        for r, m in enumerate([m_smooth, m_tex]):
            d = (y + REGION_AMP * tex * m).clamp(0, 1)
            for k, v in components(metrics, lp_fn, d, y, hr_ll, hr_hf).items():
                reg[k][i, r] = v
        print(f'  [{i+1}/{len(refs)}] {os.path.basename(files[i])}', flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(NPZ, files=np.array([os.path.relpath(f, DATA) for f in files]), targets=np.array(TARGETS),
             conds=np.array(CONDS),
             **arr, **{f'region_{k}': v for k, v in reg.items()})
    print(f'saved -> {NPZ}')


# ── Stage 2 ──────────────────────────────────────────────────────────────────

def composite(d, lam, variant, prefix=''):
    sfx = '_legacy' if variant == 'legacy' else ''
    return (lam[0] * np.tanh(d[prefix + 'l_struct' + sfx] / SCALES[0])
            + lam[1] * np.tanh(d[prefix + 'l_percept'] / SCALES[1])
            + lam[2] * np.tanh(d[prefix + 'l_hal' + sfx] / SCALES[2]))


def ranking_accuracy(score, conds):
    """score: [N, T, C]. Fraction of triples where hallucinate is scored worse (higher)."""
    h = score[:, :, conds.index('hallucinate')]
    out = {}
    for ctrl in CONTROLS:
        s = score[:, :, conds.index(ctrl)]
        ok = np.isfinite(h) & np.isfinite(s)
        out[ctrl] = float(((h > s) + 0.5 * (h == s))[ok].mean())
    out['all'] = float(np.mean([out[c] for c in CONTROLS]))
    return out


def mos_srcc(lam, variant):
    vals = []
    for ds in ['TID2013', 'KADID-10K', 'PIPAL']:
        m = dict(np.load(os.path.join(RESULTS_DIR, f'mos_components_{ds}.npz')))
        vals.append(stats.spearmanr(-composite(m, lam, variant), m['mos'])[0])
    return float(np.mean(vals)), [float(v) for v in vals]


def analyse():
    d = dict(np.load(NPZ))
    conds = [str(c) for c in d['conds']]
    res = {'n_images': int(d['l_hal'].shape[0]), 'targets': TARGETS,
           'reachable': {c: int(np.isfinite(d['l_hal'][:, :, k]).sum()) for k, c in enumerate(conds)}}
    res['l_hal_by_condition'] = {
        v: {f'{t:g}dB': {c: float(np.nanmean(d['l_hal' + v][:, ti, ci])) for ci, c in enumerate(conds)}
            for ti, t in enumerate(TARGETS)} for v in ['', '_legacy']}
    res['region_test'] = {v or '_corrected': {
        'smooth': float(np.mean(d['region_l_hal' + v][:, 0])),
        'textured': float(np.mean(d['region_l_hal' + v][:, 1])),
        'ratio': float(np.mean(d['region_l_hal' + v][:, 0]) / np.mean(d['region_l_hal' + v][:, 1]))}
        for v in ['', '_legacy']}
    res['baselines'] = {
        'LPIPS':        ranking_accuracy(d['lpips'], conds),
        'percept-only': ranking_accuracy(np.tanh(d['l_percept'] / SCALES[1]), conds),
        'hal-only':     ranking_accuracy(np.tanh(d['l_hal'] / SCALES[2]), conds),
    }
    for variant in ['legacy', 'corrected']:
        grid = []
        for lp in LP_GRID:
            for lh in LH_GRID:
                lam = (1.0, lp, lh)
                avg, per = mos_srcc(lam, variant)
                grid.append({'lambda': lam, 'mos_srcc_avg': avg, 'mos_srcc': per,
                             'hal_rank_acc': ranking_accuracy(composite(d, lam, variant), conds)})
        res[variant] = grid
        print(f'\n=== {variant} metric: lambda_s = 1 ===')
        print(f'{"lp":>6s}{"lh":>6s}{"MOS avg":>9s}{"halAcc":>8s}  ' + ' '.join(f'{c[:7]:>7s}' for c in CONTROLS))
        for g in grid:
            a = g['hal_rank_acc']
            tag = '  <- default' if tuple(g['lambda']) == DEFAULT else ''
            print(f'{g["lambda"][1]:6.2f}{g["lambda"][2]:6.2f}{g["mos_srcc_avg"]:9.4f}{a["all"]:8.3f}  '
                  + ' '.join(f'{a[c]:7.3f}' for c in CONTROLS) + tag)
    print('\nbaselines:', json.dumps({k: round(v['all'], 3) for k, v in res['baselines'].items()}))
    print('L_hal by condition (corrected):', json.dumps(res['l_hal_by_condition']['']))
    print('region test:', json.dumps(res['region_test']))
    print('reachable:', res['reachable'])
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'saved -> {OUT_JSON}')


if __name__ == '__main__':
    if not os.path.exists(NPZ) or '--rebuild' in sys.argv:
        build_cache()
    analyse()
