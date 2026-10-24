#!/usr/bin/env python3
"""
eval_benchmarks.py — one protocol for every SR benchmark table of the paper.

Every configuration (pretrained models, +WaSP, branch ablations, loss controls, RRDB network
interpolation) is evaluated the same way:
  * HR = image_SRF_<s>/*HR.png cropped to a multiple of s; LR = the official LR images
  * PSNR / SSIM on the Y channel with a 4-pixel border (utils/metrics.py)
  * SR resized to the HR size (bilinear) only if a model's output size differs
  * torch.manual_seed(42) at the start of every dataset (SinSR samples noise)
Per image it stores the raw WaSP branches of the final metric (l_struct, l_percept, l_hal), of
the submitted version (l_struct_legacy, l_hal_legacy) and Branch A with the arithmetic-mean
MS-SSIM (l_struct_mean), plus LPIPS, so any weighting can be recomputed without inference.

Output: <out_dir>/<config>.json, default results/bench_x<s>/ (existing files are skipped).

  python evaluation/eval_benchmarks.py --scale 4                    # all x4 configurations
  python evaluation/eval_benchmarks.py --scale 4 --group base main
  python evaluation/eval_benchmarks.py --scale 2 --only RCAN RCAN+WaSP
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import DISPLAY, build_model, checkpoint_path, load_checkpoint, super_resolve
from utils.data import BENCHMARKS, benchmark_pairs, load_benchmark_pair
from utils.metrics import calculate_psnr, calculate_ssim
from utils.paths import RESULTS_DIR, SR_DATA
from wasp import WaSPMetric

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KEYS = ['psnr', 'ssim', 'l_struct_legacy', 'l_struct', 'l_struct_mean', 'l_percept',
        'l_hal_legacy', 'l_hal', 'lpips']
ABLATIONS = ['la', 'lb', 'lc', 'lalb']
CONTROLS = ['l1', 'lpips', 'dists', 'msssim']


def cache_name(config):
    return config.replace(' ', '_').replace('+', 'plus') + '.json'


def interp_rrdb(alpha):
    """Network interpolation RealESRNet (alpha = 0) -> RealESRGAN (alpha = 1), x4."""
    def make():
        a, b = build_model('realesrnet', 4, 'cpu'), build_model('realesrgan', 4, 'cpu')
        sa, sb = a.state_dict(), b.state_dict()
        a.load_state_dict({k: (1 - alpha) * sa[k] + alpha * sb[k] for k in sa}, strict=True)
        return a.to(DEVICE).eval()
    return make


def registry(s, ckpt_dir=None):
    """{config name: (model factory, group)} of every configuration at scale s."""
    def pretrained(name):
        return lambda: build_model(name, s, DEVICE, ckpt_dir=ckpt_dir)

    def finetuned(name, variant):
        return lambda: load_checkpoint(build_model(name, s, DEVICE, ckpt_dir=ckpt_dir),
                                       checkpoint_path(name, s, variant, ckpt_dir))

    if s == 4:
        tuned = ['srresnet', 'rcan', 'edsr', 'realesrnet', 'realesrgan', 'swinir', 'sinsr']
        reg = {'Bicubic': (pretrained('bicubic'), 'base'), 'IMDN': (pretrained('imdn'), 'base')}
    else:
        tuned = ['rcan', 'edsr', 'realesrgan', 'swinir']
        reg = {'Bicubic': (pretrained('bicubic'), 'base')}
    for name in tuned:
        disp = DISPLAY[name]
        reg['SRResNet-L1' if name == 'srresnet' else disp] = (pretrained(name), 'base')
        reg[f'{disp}+WaSP'] = (finetuned(name, 'wasp'), 'main')
        if s == 4 and name != 'srresnet':
            for k in ABLATIONS:
                reg[f'{disp} abla_{k}'] = (finetuned(name, f'abla_{k}'), 'abla')
        for k in CONTROLS:
            reg[f'{disp} ctrl_{k}'] = (finetuned(name, f'ctrl_{k}'), 'ctrl')
    if s == 4:
        for a in [0.25, 0.5, 0.75]:
            reg[f'RRDB interp {a}'] = (interp_rrdb(a), 'interp')
    return reg


@torch.no_grad()
def run_config(name, make, s, metrics, lpips_fn, out_path, data_root):
    wasp, wasp_legacy, wasp_mean = metrics
    model = make()
    rec = {}
    for ds in BENCHMARKS:
        torch.manual_seed(42)
        pairs = benchmark_pairs(ds, s, data_root)
        r = {k: [] for k in KEYS}
        for hp, lp in pairs:
            lr, hr = load_benchmark_pair(hp, lp, s, DEVICE)
            sr = super_resolve(model, lr, hr.shape[-2:])
            sr_ll, sr_hf = wasp.dwt(sr)
            hr_ll, hr_hf = wasp.dwt(hr)
            r['psnr'].append(calculate_psnr(sr, hr).item())
            r['ssim'].append(calculate_ssim(sr, hr).item())
            r['l_struct_legacy'].append(wasp_legacy.structural_loss(sr_ll, hr_ll).item())
            r['l_struct'].append(wasp.structural_loss(sr_ll, hr_ll).item())
            r['l_struct_mean'].append(wasp_mean.structural_loss(sr_ll, hr_ll).item())
            r['l_percept'].append(wasp.perceptual_loss(sr, hr).item())
            r['l_hal_legacy'].append(wasp_legacy.hallucination_penalty(sr_hf, hr_hf).item())
            r['l_hal'].append(wasp.hallucination_penalty(sr_hf, hr_hf).item())
            r['lpips'].append(lpips_fn(sr * 2 - 1, hr * 2 - 1).item())
        r['files'] = [os.path.basename(hp) for hp, _ in pairs]
        rec[ds] = r
    with open(out_path, 'w') as f:
        json.dump({'config': name, 'scale': s, 'per_image': rec}, f)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scale', type=int, default=4, choices=[2, 4])
    ap.add_argument('--group', nargs='*', default=None, help='any of: base main abla ctrl interp')
    ap.add_argument('--only', nargs='*', default=None, help='configuration names')
    ap.add_argument('--data', default=SR_DATA, help='root of Set5/Set14/BSD100/Urban100')
    ap.add_argument('--ckpt_dir', default=None, help='released checkpoints (default: checkpoints/)')
    ap.add_argument('--out_dir', default=None, help='default: results/bench_x<scale>')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--list', action='store_true', help='print the configurations and exit')
    args = ap.parse_args()

    reg = registry(args.scale, args.ckpt_dir)
    if args.list:
        for name, (_, group) in reg.items():
            print(f'{group:7s} {name}')
        return

    import lpips
    metrics = (WaSPMetric().to(DEVICE).eval(),
               WaSPMetric(legacy=True).to(DEVICE).eval(),
               WaSPMetric(msssim_form='mean').to(DEVICE).eval())
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, f'bench_x{args.scale}')
    os.makedirs(out_dir, exist_ok=True)
    for name, (make, group) in reg.items():
        if args.only and name not in args.only:
            continue
        if args.group and group not in args.group:
            continue
        out = os.path.join(out_dir, cache_name(name))
        if os.path.exists(out) and not args.overwrite:
            continue
        t0 = time.time()
        print(f'[{name}]', flush=True)
        try:
            run_config(name, make, args.scale, metrics, lpips_fn, out, args.data)
            print(f'  done in {time.time() - t0:.0f}s', flush=True)
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {str(e)[:300]}', flush=True)


if __name__ == '__main__':
    main()
