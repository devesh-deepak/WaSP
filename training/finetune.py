#!/usr/bin/env python3
"""
finetune.py — Fine-tune any pretrained SR model with the WaSP perceptual loss.

Two-phase training strategy:
  Phase 1 (warm-up):   L1 only  — adapts pretrained weights to the training data.
  Phase 2 (fine-tune): L1 + WaSP — perceptual refinement with hallucination penalty.

Supported models: rcan | edsr | realesrnet | realesrgan | swinir

Usage:
  python training/finetune.py \\
      --model rcan \\
      --hr_dir /path/to/DIV2K/DIV2K_train_HR \\
      --scale 4 \\
      --warmup_epochs 20 \\
      --finetune_epochs 200

All arguments and defaults are documented below.
"""

import os, sys, time, logging, argparse, glob, random

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'wasp'))
sys.path.insert(0, os.path.join(_ROOT, 'utils'))
sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from wasp_metric import WaSPMetric
from metrics import calculate_psnr, calculate_ssim

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='WaSP fine-tuning for SR models')
parser.add_argument('--model', required=True,
                    choices=['rcan', 'edsr', 'realesrnet', 'realesrgan', 'swinir'],
                    help='SR model architecture to fine-tune.')
parser.add_argument('--scale',           type=int,   default=4,
                    help='SR upscaling factor (default: 4).')
parser.add_argument('--hr_dir',          required=True,
                    help='Path to the HR training image directory (e.g., DIV2K_train_HR).')
parser.add_argument('--pretrained',      type=str,   default='',
                    help='Path to pretrained model weights (.pth). If empty, '
                         'loads official weights via the model loader.')
parser.add_argument('--out_dir',         type=str,   default='checkpoints',
                    help='Directory to save fine-tuned checkpoints.')
# Training schedule
parser.add_argument('--warmup_epochs',   type=int,   default=20,
                    help='Number of L1-only warm-up epochs (default: 20).')
parser.add_argument('--finetune_epochs', type=int,   default=200,
                    help='Number of L1+WaSP fine-tuning epochs (default: 200).')
parser.add_argument('--lr',              type=float, default=5e-5,
                    help='Initial learning rate (default: 5e-5).')
parser.add_argument('--lr_decay_step',   type=int,   default=50,
                    help='StepLR decay period in epochs (default: 50).')
parser.add_argument('--lr_decay_gamma',  type=float, default=0.5,
                    help='StepLR decay factor (default: 0.5).')
# Data
parser.add_argument('--crop',            type=int,   default=128,
                    help='HR patch crop size (default: 128).')
parser.add_argument('--batch',           type=int,   default=8,
                    help='Batch size (default: 8).')
parser.add_argument('--workers',         type=int,   default=4,
                    help='DataLoader worker threads (default: 4).')
parser.add_argument('--val_size',        type=int,   default=20,
                    help='Number of images held out for validation (default: 20).')
# WaSP loss weights
parser.add_argument('--lambda_l1',       type=float, default=1.0,
                    help='Weight for the L1 pixel loss (default: 1.0).')
parser.add_argument('--lambda_struct',   type=float, default=0.1,
                    help='Weight for WaSP structural branch λ_s (default: 0.1).')
parser.add_argument('--lambda_percept',  type=float, default=1.0,
                    help='Weight for WaSP perceptual branch λ_p (default: 1.0).')
parser.add_argument('--lambda_hal',      type=float, default=0.05,
                    help='Weight for WaSP hallucination branch λ_h (default: 0.05).')
parser.add_argument('--wasp_scale',      type=float, default=0.01,
                    help='Global WaSP loss multiplier α in L_total = L1 + α·WaSP '
                         '(default: 0.01).')
# Logging
parser.add_argument('--save_freq',       type=int,   default=25,
                    help='Save a checkpoint every N epochs (default: 25).')
parser.add_argument('--print_freq',      type=int,   default=50,
                    help='Print a training log line every N batches (default: 50).')
args = parser.parse_args()

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
TOTAL_EPOCHS = args.warmup_epochs + args.finetune_epochs
CKPT_DIR     = os.path.join(args.out_dir, f'{args.model}_x{args.scale}_wasp')
os.makedirs(CKPT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────
class _PatchDataset(Dataset):
    """Random HR patch → bicubic LR downscale dataset."""
    def __init__(self, files, crop, scale, augment=True):
        self.files, self.crop, self.scale, self.augment = files, crop, scale, augment

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert('RGB')
        w, h = hr.size
        w, h = (w // self.scale) * self.scale, (h // self.scale) * self.scale
        hr = hr.crop((0, 0, w, h))
        c = self.crop
        if w > c and h > c:
            x, y = random.randint(0, w - c), random.randint(0, h - c)
            hr = hr.crop((x, y, x + c, y + c))
        lc = hr.size[0] // self.scale
        lr = hr.resize((lc, lc), Image.BICUBIC)
        hr_t, lr_t = TF.to_tensor(hr), TF.to_tensor(lr)
        if self.augment:
            if random.random() > 0.5: hr_t, lr_t = torch.flip(hr_t,[2]), torch.flip(lr_t,[2])
            if random.random() > 0.5: hr_t, lr_t = torch.flip(hr_t,[1]), torch.flip(lr_t,[1])
        return lr_t, hr_t


class _ValDataset(Dataset):
    def __init__(self, files, scale):
        self.files, self.scale = files, scale
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert('RGB')
        w, h = hr.size
        w, h = (w // self.scale) * self.scale, (h // self.scale) * self.scale
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        return TF.to_tensor(lr), TF.to_tensor(hr)


def _get_loaders():
    exts = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(args.hr_dir, ext))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f'No images found in: {args.hr_dir}')
    val_files   = files[-args.val_size:]
    train_files = files[:-args.val_size]
    log.info(f'[Data] train={len(train_files)}  val={len(val_files)}')
    train_ds = _PatchDataset(train_files, args.crop, args.scale, augment=True)
    val_ds   = _ValDataset(val_files, args.scale)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    return train_loader, val_loader


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_model(name, scale, device, pretrained_path=''):
    """
    Load a supported SR architecture. Official pretrained weights are fetched
    automatically by each model loader; pass --pretrained to override.
    """
    log.info(f'[Model] Loading {name.upper()} ×{scale} …')
    if name == 'rcan':
        from models.rcan_model import load_rcan       # noqa: F401  (user supplies)
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
        from models.swinir_model import load_swinir_trainable
        model = load_swinir_trainable(scale=scale, device=device)
    else:
        raise ValueError(f'Unknown model: {name}')

    if pretrained_path and os.path.exists(pretrained_path):
        sd = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'model_state' in sd:
            sd = sd['model_state']
        model.load_state_dict(sd, strict=False)
        log.info(f'[Model] Loaded custom weights from {pretrained_path}')

    model.to(device).train()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'[Model] {name.upper()} ready — {n:,} trainable params')
    return model


def _infer(model, lr_t):
    out = model(lr_t)
    if isinstance(out, (tuple, list)): out = out[0]
    if hasattr(out, 'reconstruction'):  out = out.reconstruction
    return out.clamp(0, 1)


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def _validate(model, loader, wasp, limit=20):
    model.eval()
    psnr_sum = ssim_sum = wasp_sum = 0.0
    count = 0
    for i, (lr, hr) in enumerate(loader):
        if i >= limit: break
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        sr = _infer(model, lr)
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = F.interpolate(sr, size=hr.shape[-2:], mode='bilinear', align_corners=False)
        psnr_sum += calculate_psnr(sr, hr).item()
        ssim_sum += calculate_ssim(sr, hr).item()
        w, *_ = wasp(sr, hr)
        wasp_sum += w.item()
        count += 1
    model.train()
    n = max(count, 1)
    return psnr_sum / n, ssim_sum / n, wasp_sum / n


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    log.info('=' * 68)
    log.info(f'  WaSP Fine-tuning: {args.model.upper()} ×{args.scale}  |  Device: {DEVICE}')
    log.info(f'  Warm-up: {args.warmup_epochs} epochs (L1)  |  Fine-tune: {args.finetune_epochs} epochs (L1+WaSP)')
    log.info(f'  LR={args.lr}  Batch={args.batch}  Crop={args.crop}')
    log.info(f'  λ_l1={args.lambda_l1}  λ_s={args.lambda_struct}  λ_p={args.lambda_percept}  '
             f'λ_h={args.lambda_hal}  α={args.wasp_scale}')
    log.info(f'  Checkpoints → {CKPT_DIR}')
    log.info('=' * 68)

    model    = _load_model(args.model, args.scale, DEVICE, args.pretrained)
    wasp     = WaSPMetric(backbone='alexnet').to(DEVICE).eval()
    for p in wasp.parameters(): p.requires_grad_(False)
    log.info('[WaSP] Metric ready (frozen).')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.lr_decay_step,
                                          gamma=args.lr_decay_gamma)
    l1_loss = nn.L1Loss()
    train_loader, val_loader = _get_loaders()

    best_wasp = float('inf')

    for epoch in range(1, TOTAL_EPOCHS + 1):
        in_wasp_phase = epoch > args.warmup_epochs
        phase = 'WaSP' if in_wasp_phase else 'L1  '
        e_loss = e_l1 = e_struct = e_percept = e_hal = 0.0
        t0 = time.time()

        for bi, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            sr = _infer(model, lr)
            if sr.shape[-2:] != hr.shape[-2:]:
                sr = F.interpolate(sr, size=hr.shape[-2:],
                                   mode='bilinear', align_corners=False).clamp(0, 1)

            l_pix = l1_loss(sr, hr)
            if in_wasp_phase:
                _, l_s, l_p, l_h = wasp(sr, hr)
                wasp_component = (args.lambda_struct  * l_s +
                                  args.lambda_percept * l_p +
                                  args.lambda_hal     * l_h)
                loss = args.lambda_l1 * l_pix + args.wasp_scale * wasp_component
            else:
                loss = l_s = l_p = l_h = l_pix

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            e_loss    += loss.item()
            e_l1      += l_pix.item()
            e_struct  += l_s.item() if isinstance(l_s, torch.Tensor) else 0.0
            e_percept += l_p.item() if isinstance(l_p, torch.Tensor) else 0.0
            e_hal     += l_h.item() if isinstance(l_h, torch.Tensor) else 0.0

        scheduler.step()
        nb = max(len(train_loader), 1)
        val_psnr, val_ssim, val_wasp = _validate(model, val_loader, wasp)

        log.info(
            f'[{phase}] Epoch [{epoch:4d}/{TOTAL_EPOCHS}]  '
            f'Loss:{e_loss/nb:.4f}  L1:{e_l1/nb:.4f}  '
            f'Str:{e_struct/nb:.4f}  Per:{e_percept/nb:.4f}  Hal:{e_hal/nb:.4f}  | '
            f'PSNR:{val_psnr:.2f}dB  SSIM:{val_ssim:.4f}  WaSP:{val_wasp:.4f}  '
            f'LR:{scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s'
        )

        if epoch % args.save_freq == 0:
            ckpt = os.path.join(CKPT_DIR, f'{args.model}_wasp_ep{epoch:04d}.pth')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'psnr': val_psnr, 'wasp': val_wasp}, ckpt)

        if val_wasp < best_wasp:
            best_wasp = val_wasp
            best_path = os.path.join(CKPT_DIR, f'{args.model}_wasp_best.pth')
            torch.save(model.state_dict(), best_path)
            log.info(f'  *** Best WaSP: {best_wasp:.4f} → {best_path}')

    final_path = os.path.join(CKPT_DIR, f'{args.model}_wasp_final.pth')
    torch.save(model.state_dict(), final_path)
    log.info(f'Training complete.  Best WaSP={best_wasp:.4f}')
    log.info(f'Final checkpoint → {final_path}')


if __name__ == '__main__':
    train()
