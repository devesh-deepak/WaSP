#!/usr/bin/env python3
"""
hparam_sweep.py — sensitivity of the (corrected) WaSP metric to its hyperparameters.

On the same image pairs as mos_components.py (TID2013, KADID-10K, PIPAL 5,000
sample, seed 42) it recomputes the components for
  * the hallucination quantile q in {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}
  * the number of Haar levels L in {1, 2, 3, 4}  (Branch A on LL_L / 2^L)
  * the wavelet basis in {haar, db2, db4, sym4, bior2.2}  (pywt, periodization)
  * the Branch B backbone (AlexNet vs VGG19)
holding everything else at the defaults (q = 0.25, L = 3, Haar, AlexNet, weights
(1, 0.5, 0.1)). The '<=' smooth mask and LL / 2^L normalisation are used throughout.

Outputs: results/hparam_components_<dataset>.npz (cached), results/hparam_sensitivity.json

  python evaluation/hparam_sweep.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from scipy import stats

from utils.iqa import IQA_DATASETS, align, img_tensor, iter_pairs
from utils.paths import RESULTS_DIR
from wasp import MultiLevelDWT, WaSPMetric, ms_ssim_loss

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE    = RESULTS_DIR
OUT_JSON = os.path.join(RESULTS_DIR, 'hparam_sensitivity.json')
QS       = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
LEVELS   = [1, 2, 3, 4]
WAVELETS = ['db2', 'db4', 'sym4', 'bior2.2']          # haar = the project's own DWT
SCALES   = (0.30, 0.05, 0.01)
DEFAULT  = (1.0, 0.5, 0.1)


def hal_penalty(sr_hf, hr_hf, q=0.25):
    """Branch C with the corrected '<=' mask and a configurable quantile."""
    total = 0.0
    for sb, hb in zip(sr_hf, hr_hf):
        he = torch.sqrt(hb['LH'] ** 2 + hb['HL'] ** 2 + hb['HH'] ** 2 + 1e-8).mean(1, keepdim=True)
        se = torch.sqrt(sb['LH'] ** 2 + sb['HL'] ** 2 + sb['HH'] ** 2 + 1e-8).mean(1, keepdim=True)
        thr = torch.quantile(he.view(he.shape[0], -1), q, dim=1).view(-1, 1, 1, 1)
        total = total + (F.relu(se - he) * (he <= thr).float()).mean()
    return (total / len(hr_hf)).item()


def haar_levels(dwt, x, n):
    lls, hfs, ll = [], [], x
    for _ in range(n):
        ll, lh, hl, hh = dwt.haar_step(ll)
        lls.append(ll)
        hfs.append({'LH': lh, 'HL': hl, 'HH': hh})
    return lls, hfs


def pywt_dwt(x, wavelet, level=3):
    c = pywt.wavedec2(x[0].cpu().numpy(), wavelet, mode='periodization', level=level, axes=(-2, -1))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(0).to(DEVICE)
    hf = [{'LH': t(h), 'HL': t(v), 'HH': t(dd)} for (h, v, dd) in c[1:]][::-1]   # finest first
    return t(c[0]), hf


@torch.no_grad()
def pair_components(dwt, alex, vgg, d, r):
    out = {}
    sl, sh = haar_levels(dwt, d, 4)
    hl, hh = haar_levels(dwt, r, 4)
    for L in LEVELS:
        out[f'struct_L{L}'] = ms_ssim_loss(sl[L - 1] / 2 ** L, hl[L - 1] / 2 ** L, levels=3).item()
        out[f'hal_L{L}'] = hal_penalty(sh[:L], hh[:L])
    for q in QS:
        out[f'hal_q{q:.2f}'] = hal_penalty(sh[:3], hh[:3], q)
    for w in WAVELETS:
        sa, sf = pywt_dwt(d, w)
        ha, hf = pywt_dwt(r, w)
        out[f'struct_{w}'] = ms_ssim_loss(sa / 8, ha / 8, levels=3).item()
        out[f'hal_{w}'] = hal_penalty(sf, hf)
    out['percept_alexnet'] = alex.perceptual_loss(d, r).item()
    out['percept_vgg19'] = vgg.perceptual_loss(d, r).item()
    return out


def build(name, dwt, alex, vgg):
    path = os.path.join(CACHE, f'hparam_components_{name}.npz')
    if os.path.exists(path):
        return dict(np.load(path))
    rec, mos, t0 = {}, [], time.time()
    for i, (_, dp, rp, m) in enumerate(iter_pairs(name)):
        r = img_tensor(rp)
        d = align(img_tensor(dp), r)
        for k, v in pair_components(dwt, alex, vgg, d.to(DEVICE), r.to(DEVICE)).items():
            rec.setdefault(k, []).append(v)
        mos.append(m)
        if (i + 1) % 1000 == 0:
            print(f'  [{name}] {i + 1} pairs ({time.time() - t0:.0f}s)', flush=True)
    out = {k: np.array(v) for k, v in rec.items()}
    out['mos'] = np.array(mos)
    np.savez(path, **out)
    return out


def srcc(score, mos):
    return float(stats.spearmanr(-score, mos)[0])


def total(s, p, h, lam=DEFAULT):
    return (lam[0] * np.tanh(s / SCALES[0]) + lam[1] * np.tanh(p / SCALES[1])
            + lam[2] * np.tanh(h / SCALES[2]))


def main():
    dwt = MultiLevelDWT(levels=4).to(DEVICE)
    alex = WaSPMetric(backbone='alexnet').to(DEVICE).eval()
    vgg = WaSPMetric(backbone='vgg19').to(DEVICE).eval()
    data = {}
    for name in IQA_DATASETS:
        data[name] = build(name, dwt, alex, vgg)
        print(f'  [{name}] ready ({len(data[name]["mos"])} pairs)', flush=True)

    res = {'quantile': {}, 'levels': {}, 'wavelet': {}, 'backbone': {}, 'weights': {}}
    for ds, d in data.items():
        P, S3, H3 = d['percept_alexnet'], d['struct_L3'], d['hal_L3']
        res['quantile'][ds] = {f'{q:.2f}': {'Lhal_SRCC': srcc(d[f'hal_q{q:.2f}'], d['mos']),
                                            'WaSP_SRCC': srcc(total(S3, P, d[f'hal_q{q:.2f}']), d['mos'])}
                               for q in QS}
        res['levels'][ds] = {str(L): {'Lstruct_SRCC': srcc(d[f'struct_L{L}'], d['mos']),
                                      'Lhal_SRCC': srcc(d[f'hal_L{L}'], d['mos']),
                                      'WaSP_SRCC': srcc(total(d[f'struct_L{L}'], P, d[f'hal_L{L}']), d['mos'])}
                             for L in LEVELS}
        wv = {'haar': (S3, H3)}
        wv.update({w: (d[f'struct_{w}'], d[f'hal_{w}']) for w in WAVELETS})
        res['wavelet'][ds] = {w: {'Lstruct_SRCC': srcc(s, d['mos']), 'Lhal_SRCC': srcc(h, d['mos']),
                                  'WaSP_SRCC': srcc(total(s, P, h), d['mos'])} for w, (s, h) in wv.items()}
        res['backbone'][ds] = {b: {'percept_SRCC': srcc(d[f'percept_{b}'], d['mos']),
                                   'WaSP_SRCC': srcc(total(S3, d[f'percept_{b}'], H3), d['mos'])}
                               for b in ['alexnet', 'vgg19']}
        res['weights'][ds] = {f'lp={lp}': srcc(total(S3, P, H3, (1.0, lp, 0.1)), d['mos'])
                              for lp in [0.25, 0.5, 1.0, 2.0, 4.0]}

    def max_delta(block):
        return {ds: max(v.values()) - min(v.values())
                for ds, v in {ds: {k: (x['WaSP_SRCC'] if isinstance(x, dict) else x)
                                   for k, x in cfg.items()} for ds, cfg in block.items()}.items()}
    res['max_delta'] = {k: max_delta(res[k]) for k in ['quantile', 'levels', 'wavelet', 'weights', 'backbone']}
    with open(OUT_JSON, 'w') as f:
        json.dump(res, f, indent=2)
    for k in ['quantile', 'levels', 'wavelet', 'backbone', 'weights']:
        print(f'\n== {k}')
        for ds in data:
            row = {c: round(v['WaSP_SRCC'] if isinstance(v, dict) else v, 4) for c, v in res[k][ds].items()}
            print(f'  {ds:10s} {row}')
    print('\nmax delta:', json.dumps({k: {ds: round(v, 3) for ds, v in m.items()} for k, m in res['max_delta'].items()}))
    print(f'saved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
