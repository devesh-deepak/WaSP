#!/usr/bin/env python3
"""
training_cost.py — full training-iteration time and GPU memory of each fine-tuning loss.

The forward-pass runtime (runtime.py) does not show the training overhead of WaSP. This script
times the complete training iteration of training/finetune.py
(zero_grad, SR forward, clamp, loss, backward, gradient clipping, Adam step) for every model and
for each Phase-2 loss (l1, wasp, lpips, dists, msssim), with the recipe of the paper: fp32,
128x128 HR crops, batch 8 (SwinIR 4), Adam, the same loss weights.

Timing: per-iteration wall-clock time with CUDA synchronisation, after warm-up; the losses are
interleaved in several rounds and the median over all timed iterations is reported. CUDA events
split each iteration into SR forward, loss forward, backward and optimiser step.
Memory: peak allocated GPU memory (torch.cuda.max_memory_allocated) over a few iterations, with
only the loss module being measured instantiated, so it includes the model, its gradients, the
Adam state, the activations and the loss network.
Data loading is excluded: the batches are pre-loaded DIV2K crops, identical for every loss.

  python evaluation/training_cost.py              # all models, x4 and x2 (one process each)
  python evaluation/training_cost.py --one rcan 4 (internal: one model per process)

Output: results/training_cost.json
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from training.losses import build_phase2_loss
from utils.paths import DIV2K_TRAIN_HR, RESULTS_DIR
from wasp import WaSPMetric

OUT    = os.path.join(RESULTS_DIR, 'training_cost.json')
HR_DIR = DIV2K_TRAIN_HR
DEVICE = 'cuda'
LOSSES = ['l1', 'wasp', 'lpips', 'dists', 'msssim']
CROP   = 128
LAM    = dict(l1=1.0, struct=0.1, percept=1.0, hal=0.05, scale=0.01)    # training/finetune.py defaults
RUNS   = {4: [('srresnet', 8), ('rcan', 8), ('edsr', 8), ('realesrgan', 8), ('swinir', 4), ('sinsr', 8)],
          2: [('rcan', 8), ('edsr', 8), ('realesrgan', 8), ('swinir', 4)]}
WARMUP, ITERS, ROUNDS, MEM_ITERS = 20, 100, 3, 5


def load_model(name, scale):
    """Same starting point and trainable parameters as training/finetune.py."""
    return build_model(name, scale, DEVICE, trainable=True)


def build_loss(name):
    """Phase-2 criterion of training/finetune.py, as a function (sr, hr) -> loss."""
    wasp = WaSPMetric().to(DEVICE).eval() if name == 'wasp' else None
    crit = build_phase2_loss(name, wasp, DEVICE, weights=(LAM['struct'], LAM['percept'], LAM['hal']),
                             alpha=LAM['scale'], lambda_l1=LAM['l1'])
    return (lambda sr, hr: crit(sr, hr)[0]), [crit]


def batches(scale, bs, n=8, seed=0):
    """n fixed (LR, HR) batches of DIV2K crops, LR by PIL bicubic as in training."""
    from PIL import Image
    import torchvision.transforms.functional as TF
    rng = random.Random(seed)
    files = sorted(glob.glob(f'{HR_DIR}/*.png'))
    out = []
    for _ in range(n):
        lrs, hrs = [], []
        for _ in range(bs):
            im = Image.open(rng.choice(files)).convert('RGB')
            x, y = rng.randrange(im.width - CROP), rng.randrange(im.height - CROP)
            hr = im.crop((x, y, x + CROP, y + CROP))
            lr = hr.resize((CROP // scale, CROP // scale), Image.BICUBIC)
            hrs.append(TF.to_tensor(hr)); lrs.append(TF.to_tensor(lr))
        out.append((torch.stack(lrs).to(DEVICE), torch.stack(hrs).to(DEVICE)))
    return out


def make_step(model, opt, params):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def step(loss_fn, lr, hr, timed=False):
        if timed: ev[0].record()
        opt.zero_grad(set_to_none=True)
        out = model(lr)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, 'reconstruction'):
            out = out.reconstruction
        sr = out.clamp(0, 1)
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = F.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=False).clamp(0, 1)
        if timed: ev[1].record()
        loss = loss_fn(sr, hr)
        if timed: ev[2].record()
        loss.backward()
        if timed: ev[3].record()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        if timed: ev[4].record()
        return ev
    return step


def run_one(name, scale, bs):
    torch.manual_seed(0)
    model = load_model(name, scale)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-5, betas=(0.9, 0.999))
    step = make_step(model, opt, params)
    data = batches(scale, bs)
    res = {'model': name, 'scale': scale, 'batch': bs, 'crop': CROP,
           'trainable_params': int(sum(p.numel() for p in params)), 'losses': {}}

    # Memory: one loss module alive at a time.
    for ln in LOSSES:
        fn, mods = build_loss(ln)
        for i in range(2):
            step(fn, *data[i])
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for i in range(MEM_ITERS):
            step(fn, *data[i % len(data)])
        torch.cuda.synchronize()
        res['losses'][ln] = {'peak_alloc_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                             'peak_reserved_gb': torch.cuda.max_memory_reserved() / 2 ** 30}
        del fn, mods
        torch.cuda.empty_cache()

    # Time: all loss modules alive, losses interleaved over rounds.
    fns = {ln: build_loss(ln)[0] for ln in LOSSES}
    times = {ln: [] for ln in LOSSES}
    parts = {ln: [] for ln in LOSSES}
    for ln in LOSSES:
        for i in range(WARMUP):
            step(fns[ln], *data[i % len(data)])
    for r in range(ROUNDS):
        for ln in (LOSSES if r % 2 == 0 else LOSSES[::-1]):
            for i in range(ITERS):
                lr, hr = data[i % len(data)]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ev = step(fns[ln], lr, hr, timed=True)
                torch.cuda.synchronize(); times[ln].append((time.perf_counter() - t0) * 1e3)
                parts[ln].append([ev[k].elapsed_time(ev[k + 1]) for k in range(4)])
    for ln in LOSSES:
        t = np.array(times[ln]); p = np.median(np.array(parts[ln]), 0)
        res['losses'][ln].update({'iter_ms_median': float(np.median(t)), 'iter_ms_p25': float(np.percentile(t, 25)),
                                  'iter_ms_p75': float(np.percentile(t, 75)), 'n_iters': int(len(t)),
                                  'sr_forward_ms': float(p[0]), 'loss_forward_ms': float(p[1]),
                                  'backward_ms': float(p[2]), 'optimizer_ms': float(p[3])})
    return res


def flops():
    """FLOPs of the WaSP and LPIPS forward passes for one 128x128 pair (torch FlopCounterMode)."""
    from torch.utils.flop_counter import FlopCounterMode
    import lpips
    out = {}
    x, y = torch.rand(1, 3, CROP, CROP, device=DEVICE), torch.rand(1, 3, CROP, CROP, device=DEVICE)
    wasp = WaSPMetric().to(DEVICE).eval()
    lp = lpips.LPIPS(net='alex', verbose=False).to(DEVICE).eval()
    for k, f in [('wasp', lambda: wasp(x, y)), ('wasp_branch_b', lambda: wasp.perceptual_loss(x, y)),
                 ('lpips', lambda: lp(x * 2 - 1, y * 2 - 1))]:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            f()
        out[k + '_gflops'] = fc.get_total_flops() / 1e9
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--one', nargs=2)
    a = ap.parse_args()
    if a.one:
        name, scale = a.one[0], int(a.one[1])
        bs = dict(RUNS[scale])[name]
        print('RESULT ' + json.dumps(run_one(name, scale, bs)), flush=True)
        return
    res = {'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'settings': {'crop': CROP, 'precision': 'fp32', 'warmup': WARMUP, 'iters_per_round': ITERS,
                        'rounds': ROUNDS, 'mem_iters': MEM_ITERS, 'loss_weights': LAM},
           'flops_128': flops(), 'runs': []}
    for scale, runs in RUNS.items():
        for name, _ in runs:
            t0 = time.time()
            p = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--one', name, str(scale)],
                               capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
            if not line:
                print(f'  {name} x{scale} FAILED\n{p.stderr[-2000:]}', flush=True)
                continue
            r = json.loads(line[0][7:]); res['runs'].append(r)
            L = r['losses']
            print(f"  {name:10s} x{scale}  l1 {L['l1']['iter_ms_median']:7.1f} ms  wasp {L['wasp']['iter_ms_median']:7.1f} ms "
                  f"(+{100 * (L['wasp']['iter_ms_median'] / L['l1']['iter_ms_median'] - 1):5.1f}%)  "
                  f"mem {L['l1']['peak_alloc_gb']:.2f} -> {L['wasp']['peak_alloc_gb']:.2f} GB  ({time.time() - t0:.0f}s)", flush=True)
            os.makedirs(RESULTS_DIR, exist_ok=True)
            json.dump(res, open(OUT, 'w'), indent=1)
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
