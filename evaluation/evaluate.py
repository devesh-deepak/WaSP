#!/usr/bin/env python3
"""
evaluate.py — Evaluate SR models on standard benchmarks using PSNR, SSIM, and WaSP.

Benchmarks: Set5 | Set14 | BSD100 | Urban100
Output:     evaluation/results.json  (human-readable JSON table)

Usage:
  python evaluation/evaluate.py \\
      --model_path checkpoints/rcan_x4_wasp/rcan_wasp_best.pth \\
      --model rcan \\
      --scale 4 \\
      --data_dir /path/to/benchmarks

Expected data_dir layout:
  data_dir/
    Set5/HR/
    Set5/LR_x4/
    Set14/HR/
    Set14/LR_x4/
    BSD100/HR/
    BSD100/LR_x4/
    Urban100/HR/
    Urban100/LR_x4/

If LR images are absent, they are generated on-the-fly via bicubic downsampling.
"""

import os, sys, glob, json, argparse, logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'wasp'))
sys.path.insert(0, os.path.join(_ROOT, 'utils'))
sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from wasp_metric import WaSPMetric
from metrics import calculate_psnr, calculate_ssim

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='WaSP benchmark evaluation')
parser.add_argument('--model',      required=True,
                    choices=['rcan', 'edsr', 'realesrnet', 'realesrgan', 'swinir', 'bicubic'],
                    help='SR model architecture.')
parser.add_argument('--model_path', type=str, default='',
                    help='Path to model weights (.pth). Not required for --model bicubic.')
parser.add_argument('--scale',      type=int, default=4,
                    help='SR upscaling factor (default: 4).')
parser.add_argument('--data_dir',   required=True,
                    help='Root directory containing benchmark subfolders.')
parser.add_argument('--datasets',   nargs='+',
                    default=['Set5', 'Set14', 'BSD100', 'Urban100'],
                    help='Which benchmark datasets to evaluate (default: all four).')
parser.add_argument('--out',        type=str, default='evaluation/results.json',
                    help='Output JSON path (default: evaluation/results.json).')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(name, scale, model_path, device):
    if name == 'bicubic':
        return None
    if name == 'rcan':
        from models.rcan_model import load_rcan
        model = load_rcan(scale=scale, device=device)
    elif name == 'edsr':
        from models.edsr_model import load_edsr
        model = load_edsr(scale=scale, device=device)
    elif name == 'realesrnet':
        from models.esrgan_model import load_realesrnet
        model = load_realesrnet(scale=scale, device=device)
    elif name == 'realesrgan':
        from models.esrgan_model import load_realesrgan
        model = load_realesrgan(scale=scale, device=device)
    elif name == 'swinir':
        from models.swinir_model import load_swinir
        model = load_swinir(scale=scale, device=device)
    else:
        raise ValueError(f'Unknown model: {name}')

    if model_path and os.path.exists(model_path):
        sd = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'model_state' in sd:
            sd = sd['model_state']
        model.load_state_dict(sd, strict=False)
        log.info(f'Loaded weights: {model_path}')
    return model.to(device).eval()


@torch.no_grad()
def infer(model, lr_t, scale):
    if model is None:
        return F.interpolate(lr_t, scale_factor=scale,
                             mode='bicubic', align_corners=False).clamp(0, 1)
    out = model(lr_t)
    if isinstance(out, (tuple, list)): out = out[0]
    if hasattr(out, 'reconstruction'):  out = out.reconstruction
    return out.clamp(0, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_dataset(ds_name, model, wasp, scale, data_dir):
    hr_dir = os.path.join(data_dir, ds_name, 'HR')
    lr_dir = os.path.join(data_dir, ds_name, f'LR_x{scale}')

    hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.png')) +
                      glob.glob(os.path.join(hr_dir, '*.jpg')))
    if not hr_files:
        log.warning(f'[{ds_name}] No HR images found in {hr_dir}. Skipping.')
        return {}

    psnrs, ssims, wasps = [], [], []
    for idx, hr_path in enumerate(hr_files):
        hr_pil = Image.open(hr_path).convert('RGB')
        w, h   = hr_pil.size
        w, h   = (w // scale) * scale, (h // scale) * scale
        hr_pil = hr_pil.crop((0, 0, w, h))

        bn      = os.path.splitext(os.path.basename(hr_path))[0]
        lr_path = None
        if os.path.isdir(lr_dir):
            cands = (glob.glob(os.path.join(lr_dir, f'{bn}*.png')) +
                     glob.glob(os.path.join(lr_dir, f'{bn}*.jpg')))
            if cands: lr_path = cands[0]

        lr_pil = (Image.open(lr_path).convert('RGB') if lr_path
                  else hr_pil.resize((w // scale, h // scale), Image.BICUBIC))

        hr_t = TF.to_tensor(hr_pil).unsqueeze(0).to(DEVICE)
        lr_t = TF.to_tensor(lr_pil).unsqueeze(0).to(DEVICE)

        sr_t = infer(model, lr_t, scale)
        if sr_t.shape[-2:] != hr_t.shape[-2:]:
            sr_t = F.interpolate(sr_t, size=hr_t.shape[-2:],
                                 mode='bilinear', align_corners=False)

        psnrs.append(calculate_psnr(sr_t, hr_t).item())
        ssims.append(calculate_ssim(sr_t, hr_t).item())
        w_out, *_ = wasp(sr_t, hr_t)
        wasps.append(w_out.item())

    result = {
        'PSNR': round(sum(psnrs) / len(psnrs), 4),
        'SSIM': round(sum(ssims) / len(ssims), 4),
        'WaSP': round(sum(wasps) / len(wasps), 4),
        'n':    len(psnrs),
    }
    log.info(f'  {ds_name:<12}  PSNR={result["PSNR"]:.2f} dB  '
             f'SSIM={result["SSIM"]:.4f}  WaSP={result["WaSP"]:.4f}  '
             f'({result["n"]} images)')
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info('=' * 60)
    log.info(f'  WaSP Evaluation  |  Model: {args.model.upper()} ×{args.scale}')
    log.info(f'  Device: {DEVICE}  |  Datasets: {args.datasets}')
    log.info('=' * 60)

    model = load_model(args.model, args.scale, args.model_path, DEVICE)
    wasp  = WaSPMetric(backbone='alexnet').to(DEVICE).eval()
    for p in wasp.parameters(): p.requires_grad_(False)
    log.info('[WaSP] Metric loaded.')

    all_results = {}
    for ds in args.datasets:
        all_results[ds] = eval_dataset(ds, model, wasp, args.scale, args.data_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({args.model: all_results}, f, indent=2)
    log.info(f'\nResults saved → {args.out}')


if __name__ == '__main__':
    main()
