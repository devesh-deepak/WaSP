#!/usr/bin/env python3
"""
patch_scale_analysis.py — Branch A scales, and patch-based training vs full-image evaluation.

The structural branch computes SSIM at up to three scales of the normalised coarsest
LL band (LL_3 / 8), and a scale is used only when both dimensions are >= 11 (the SSIM
window). A 128 x 128 training patch gives a 16 x 16 band, so training sees one scale,
while full benchmark images give two or three. This script quantifies the consequences
with the corrected metric:

  mos    For TID2013, KADID-10K and the PIPAL 5,000-pair sample: SRCC with MOS of
         Branch A and of the full score computed (i) on the full image, (ii) averaged
         over non-overlapping 128 x 128 patches (the training regime), and (iii) on the
         full image with the standard product form of MS-SSIM (Wang et al., 2003)
         instead of the weighted arithmetic mean (the form of an earlier revision).
  bench  For the x4 outputs of the 16 models of the benchmark table on Set5, Set14,
         BSD100 and Urban100: agreement between full-image and patch-averaged scores
         (per image and for the model ranking).

Patch scores average the per-patch composite tanh(L_A/.3) + .5 tanh(L_B/.05) + .1 tanh(L_C/.01).

Outputs: results/patch_scale_<part>.npz, results/patch_scale.json
  python evaluation/patch_scale_analysis.py mos
  python evaluation/patch_scale_analysis.py bench
  python evaluation/patch_scale_analysis.py report
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from utils.paths import RESULTS_DIR
from wasp import WaSPMetric
from wasp.metric import gaussian_kernel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE  = RESULTS_DIR
OUT    = os.path.join(RESULTS_DIR, 'patch_scale.json')
P      = 128
W5     = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
SC     = (0.30, 0.05, 0.01)
BENCH_CONFIGS = ['Bicubic', 'IMDN', 'SRResNet-L1', 'SRResNet+WaSP', 'RCAN', 'RCAN+WaSP',
                 'EDSR', 'EDSR+WaSP', 'RealESRNet', 'RealESRNet+WaSP', 'RealESRGAN',
                 'RealESRGAN+WaSP', 'SwinIR', 'SwinIR+WaSP', 'SinSR', 'SinSR+WaSP']


# ── Branch A variants (per sample) ───────────────────────────────────────────

def _maps(x, y, win=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2, eps=1e-8):
    """SSIM map exactly as in wasp.metric.ssim_cs (including its eps in the denominator)
    and the contrast-structure map used by the standard product form."""
    k = gaussian_kernel(win, sigma, x.shape[1], x.device)
    f = lambda t: F.conv2d(t, k, padding=win // 2, groups=x.shape[1])
    mx, my = f(x), f(y)
    sx, sy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    ssim = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2) + eps)
    cs = (2 * sxy + C2) / (sx + sy + C2)
    return ssim, cs


def n_scales(h, w, levels=3, win=11):
    n = 0
    for _ in range(levels):
        if min(h, w) < win:
            break
        n += 1
        h, w = h // 2, w // 2
    return n


def struct_per_sample(a, b, form='mean', levels=3):
    """1 - multi-scale SSIM per sample. form='mean': weighted arithmetic mean of the full
    SSIM (earlier revision); form='product': standard MS-SSIM (the final metric)."""
    m = n_scales(*a.shape[-2:], levels)
    if m == 0:
        return torch.zeros(a.shape[0], device=a.device)
    w = torch.tensor(W5[:levels][:m], device=a.device)
    w = w / w.sum()
    ssim, cs = [], []
    for i in range(m):
        sm, c = _maps(a, b)
        ssim.append(sm.mean(dim=(1, 2, 3)))
        cs.append(c.mean(dim=(1, 2, 3)))
        if i < m - 1:
            a, b = F.avg_pool2d(a, 2, 2), F.avg_pool2d(b, 2, 2)
    if form == 'mean':
        val = sum(wi * s for wi, s in zip(w, ssim))
    else:
        val = torch.ones_like(ssim[0])
        for i in range(m - 1):
            val = val * cs[i].clamp_min(0) ** w[i]
        val = val * ssim[-1].clamp_min(0) ** w[-1]
    return 1 - val


def percept_per_sample(wasp, x, y):
    fx, fy = wasp.feat_extractor(x), wasp.feat_extractor(y)
    return sum(wl * (a - b).abs().mean(dim=(1, 2, 3)) for wl, a, b in zip(wasp.feat_weights, fx, fy))


def hal_per_sample(sr_hf, hr_hf, q=0.25):
    e = lambda d: torch.sqrt(d['LH'] ** 2 + d['HL'] ** 2 + d['HH'] ** 2 + 1e-8).mean(1, keepdim=True)
    tot = 0
    for s, h in zip(sr_hf, hr_hf):
        es, eh = e(s), e(h)
        thr = torch.quantile(eh.flatten(1), q, dim=1).view(-1, 1, 1, 1)
        tot = tot + (F.relu(es - eh) * (eh <= thr).float()).mean(dim=(1, 2, 3))
    return tot / len(hr_hf)


def composite(s, p, h):
    return torch.tanh(s / SC[0]) + 0.5 * torch.tanh(p / SC[1]) + 0.1 * torch.tanh(h / SC[2])


def tiles(x):
    _, c, H, W = x.shape
    nh, nw = H // P, W // P
    if nh == 0 or nw == 0:
        return None
    x = x[:, :, :nh * P, :nw * P].unfold(2, P, P).unfold(3, P, P)      # 1,C,nh,nw,P,P
    return x.permute(0, 2, 3, 1, 4, 5).reshape(nh * nw, c, P, P)


@torch.no_grad()
def scores(wasp, sr, hr):
    """Full-image (mean and product form) and patch-averaged scores for one pair."""
    out = {}
    s_ll, s_hf = wasp.dwt(sr)
    h_ll, h_hf = wasp.dwt(hr)
    a, b = s_ll / 8, h_ll / 8
    out['scales_full'] = n_scales(*a.shape[-2:])
    out['struct_full'] = struct_per_sample(a, b, 'mean').item()
    out['struct_full_product'] = struct_per_sample(a, b, 'product').item()
    p = percept_per_sample(wasp, sr, hr)
    hh = hal_per_sample(s_hf, h_hf)
    out['wasp_full'] = composite(torch.tensor(out['struct_full']), p.cpu(), hh.cpu()).item()
    out['wasp_full_product'] = composite(torch.tensor(out['struct_full_product']), p.cpu(), hh.cpu()).item()
    ts, th = tiles(sr), tiles(hr)
    if ts is None:
        out.update(struct_patch=np.nan, wasp_patch=np.nan, n_patches=0)
        return out
    ps_ll, ps_hf = wasp.dwt(ts)
    ph_ll, ph_hf = wasp.dwt(th)
    sp = struct_per_sample(ps_ll / 8, ph_ll / 8, 'mean')
    pp = percept_per_sample(wasp, ts, th)
    hp = hal_per_sample(ps_hf, ph_hf)
    out['struct_patch'] = sp.mean().item()
    out['wasp_patch'] = composite(sp, pp, hp).mean().item()
    out['n_patches'] = ts.shape[0]
    return out


def save(part, rec, extra):
    keys = list(rec[0].keys())
    np.savez(os.path.join(CACHE, f'patch_scale_{part}.npz'),
             **{k: np.array([r[k] for r in rec]) for k in keys}, **extra)


# ── Parts ────────────────────────────────────────────────────────────────────

def run_mos():
    from utils.iqa import IQA_DATASETS, align, img_tensor, iter_pairs
    wasp = WaSPMetric().to(DEVICE).eval()
    for name in IQA_DATASETS:
        rec, mos, t0 = [], [], time.time()
        for i, (_, dp, rp, m) in enumerate(iter_pairs(name)):
            r = img_tensor(rp)
            d = align(img_tensor(dp), r)
            rec.append(scores(wasp, d.to(DEVICE), r.to(DEVICE)))
            mos.append(m)
            if (i + 1) % 2000 == 0:
                print(f'  [{name}] {i + 1} pairs ({time.time() - t0:.0f}s)', flush=True)
        save(f'mos_{name}', rec, {'mos': np.array(mos)})
        print(f'  [{name}] saved {len(rec)} pairs', flush=True)


def run_bench():
    from evaluation.eval_benchmarks import registry
    from models import super_resolve
    from utils.data import BENCHMARKS, benchmark_pairs, load_benchmark_pair
    wasp = WaSPMetric().to(DEVICE).eval()
    reg = registry(4)
    rec = []
    for cfg in BENCH_CONFIGS:
        model = reg[cfg][0]()
        for ds in BENCHMARKS:
            torch.manual_seed(42)
            for hp, lp in benchmark_pairs(ds, 4):
                lr, hr = load_benchmark_pair(hp, lp, 4, DEVICE)
                with torch.no_grad():
                    sr = super_resolve(model, lr, hr.shape[-2:])
                r = scores(wasp, sr, hr)
                r.update(config=cfg, dataset=ds)
                rec.append(r)
        print(f'  [{cfg}] done', flush=True)
        del model
        torch.cuda.empty_cache()
    keys = [k for k in rec[0] if k not in ('config', 'dataset')]
    np.savez(os.path.join(CACHE, 'patch_scale_bench.npz'),
             config=np.array([r['config'] for r in rec]), dataset=np.array([r['dataset'] for r in rec]),
             **{k: np.array([r[k] for r in rec]) for k in keys})


def report():
    res = {'scales': {}, 'mos': {}, 'bench': {}}
    sr = lambda x, m: float(stats.spearmanr(-x, m)[0])
    for ds in ['TID2013', 'KADID-10K', 'PIPAL']:
        d = dict(np.load(os.path.join(CACHE, f'patch_scale_mos_{ds}.npz')))
        ok = np.isfinite(d['struct_patch'])
        res['scales'][ds] = {str(k): int((d['scales_full'] == k).sum()) for k in np.unique(d['scales_full'])}
        res['mos'][ds] = {
            'struct_full': sr(d['struct_full'], d['mos']),
            'struct_patch': sr(d['struct_patch'][ok], d['mos'][ok]),
            'struct_full_product': sr(d['struct_full_product'], d['mos']),
            'wasp_full': sr(d['wasp_full'], d['mos']),
            'wasp_patch': sr(d['wasp_patch'][ok], d['mos'][ok]),
            'wasp_full_product': sr(d['wasp_full_product'], d['mos']),
            'n': int(len(d['mos'])), 'n_patch': int(ok.sum()),
            'patches_per_image': float(np.mean(d['n_patches'][ok])),
            'struct_ratio_patch_over_full': float(np.mean(d['struct_patch'][ok]) / np.mean(d['struct_full_product'][ok]))}
    b = dict(np.load(os.path.join(CACHE, 'patch_scale_bench.npz')))
    for ds in ['Set5', 'Set14', 'BSD100', 'Urban100']:
        m = b['dataset'] == ds
        res['scales'][ds] = {str(k): int((b['scales_full'][m & (b['config'] == 'Bicubic')] == k).sum())
                             for k in np.unique(b['scales_full'][m])}
        ok = m & np.isfinite(b['wasp_patch'])
        cfgs = list(dict.fromkeys(b['config'][m]))
        # full-image scores use the final metric (standard product form)
        wf, sf = b['wasp_full_product'], b['struct_full_product']
        full_mean = [wf[ok & (b['config'] == c)].mean() for c in cfgs]
        patch_mean = [b['wasp_patch'][ok & (b['config'] == c)].mean() for c in cfgs]
        res['bench'][ds] = {
            'n_scored': int(ok.sum()), 'n_total': int(m.sum()),
            'srcc_per_image_wasp': float(stats.spearmanr(wf[ok], b['wasp_patch'][ok])[0]),
            'srcc_per_image_struct': float(stats.spearmanr(sf[ok], b['struct_patch'][ok])[0]),
            'mean_abs_rel_diff_wasp': float(np.mean(np.abs(b['wasp_patch'][ok] - wf[ok]) / wf[ok])),
            'struct_ratio_patch_over_full': float(np.mean(b['struct_patch'][ok]) / np.mean(sf[ok])),
            'model_rank_kendall': float(stats.kendalltau(full_mean, patch_mean)[0]),
            'model_rank_spearman': float(stats.spearmanr(full_mean, patch_mean)[0])}
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'report'
    {'mos': run_mos, 'bench': run_bench, 'report': report}[part]()
