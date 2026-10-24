#!/usr/bin/env python3
"""
finetune.py — two-phase fine-tuning of a pretrained SR model with WaSP (or a control loss).

  Phase 1 (--warmup_epochs):   L1 only, to adapt the pretrained weights to DIV2K bicubic patches
  Phase 2 (--finetune_epochs): L1 + alpha * auxiliary loss (--loss)

Data: DIV2K training HR images (--hr_dir); random 128x128 HR crops with flips, LR made on the
fly by PIL bicubic downsampling. The last --val_size images are held out for validation.
best.pth is the checkpoint with the lowest validation WaSP over all epochs (the selection rule
of the paper, which can pick a Phase-1 epoch); best_phase2.pth is the lowest over Phase 2 only.

  python training/finetune.py --model rcan --scale 4
  python training/finetune.py --model swinir --scale 4 --lr 2e-5 --batch 4
  python training/finetune.py --model edsr --scale 4 --loss lpips --run_name edsr_x4_ctrl_lpips

Outputs in <out_root>/<run_name>/: best.pth, best_phase2.pth, final.pth, ep<NNNN>.pth, train.log,
tensorboard/.
See scripts/ for the exact commands behind every checkpoint of the paper.
"""

import argparse
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import FINETUNE_MODELS, build_model, super_resolve
from training.losses import LOSSES, build_phase2_loss
from utils.data import PatchDataset, ValDataset, list_images
from utils.metrics import calculate_psnr, calculate_ssim
from utils.paths import DIV2K_TRAIN_HR, REPO
from wasp import WaSPMetric

# WaSP versions: 'final' = the metric of the paper; 'mean' = arithmetic-mean MS-SSIM (the x2
# runs); 'legacy' = the originally submitted metric (all x4 runs of the paper).
WASP_VARIANTS = {'final': {}, 'mean': {'msssim_form': 'mean'}, 'legacy': {'legacy': True}}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=FINETUNE_MODELS)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--loss', default='wasp', choices=LOSSES,
                   help='Phase-2 auxiliary loss; "l1" = no auxiliary term (pure-L1 control)')
    p.add_argument('--wasp_variant', default='final', choices=list(WASP_VARIANTS),
                   help='WaSP version used in the loss and for checkpoint selection')
    p.add_argument('--hr_dir', default=DIV2K_TRAIN_HR)
    p.add_argument('--crop', type=int, default=128, help='HR patch size')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--warmup_epochs', type=int, default=20, help='Phase 1 (L1-only) epochs')
    p.add_argument('--finetune_epochs', type=int, default=200, help='Phase 2 epochs')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--lr_decay_step', type=int, default=50)
    p.add_argument('--lr_decay_gamma', type=float, default=0.5)
    p.add_argument('--lambda_l1', type=float, default=1.0)
    p.add_argument('--lambda_struct', type=float, default=0.1, help="lambda_s' of the training loss")
    p.add_argument('--lambda_percept', type=float, default=1.0, help="lambda_p' of the training loss")
    p.add_argument('--lambda_hal', type=float, default=0.05, help="lambda_h' of the training loss")
    p.add_argument('--wasp_scale', type=float, default=0.01, help='global scale alpha of the auxiliary loss')
    p.add_argument('--val_size', type=int, default=20)
    p.add_argument('--save_freq', type=int, default=25)
    p.add_argument('--print_freq', type=int, default=50)
    p.add_argument('--run_name', default=None, help='default: <model>_x<scale>_<loss>')
    p.add_argument('--out_root', default=os.path.join(REPO, 'runs'))
    p.add_argument('--resume_from', default='', help='checkpoint to load before training')
    p.add_argument('--start_epoch', type=int, default=1, help='first epoch when resuming')
    p.add_argument('--seed', type=int, default=None, help='seed Python and torch (the paper runs were unseeded)')
    p.add_argument('--ckpt_dir', default=None, help='released checkpoints (SRResNet starting point)')
    return p.parse_args()


def get_loaders(args, log):
    files = list_images(args.hr_dir)
    if not files:
        raise FileNotFoundError(f'No images in {args.hr_dir}')
    val_files, train_files = files[-args.val_size:], files[:-args.val_size]
    log.info(f'[Data] train={len(train_files)}  val={len(val_files)}')
    train_loader = DataLoader(PatchDataset(train_files, args.crop, args.scale, augment=True),
                              batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(ValDataset(val_files, args.scale), batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)
    return train_loader, val_loader


@torch.no_grad()
def validate(model, loader, wasp, device, limit=20):
    model.eval()
    psnr_tot = ssim_tot = wasp_tot = 0.0
    count = 0
    for i, (lr, hr) in enumerate(loader):
        if i >= limit:
            break
        lr, hr = lr.to(device), hr.to(device)
        sr = super_resolve(model, lr, hr.shape[-2:])
        psnr_tot += calculate_psnr(sr, hr).item()
        ssim_tot += calculate_ssim(sr, hr).item()
        wasp_tot += wasp(sr, hr)[0].item()
        count += 1
    model.train()
    if hasattr(model, 'vqgan'):
        model.vqgan.eval()
    n = max(count, 1)
    return psnr_tot / n, ssim_tot / n, wasp_tot / n


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    run_name = args.run_name or f'{args.model}_x{args.scale}_{args.loss}'
    run_dir = os.path.join(args.out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_path, final_path = os.path.join(run_dir, 'best.pth'), os.path.join(run_dir, 'final.pth')
    best_p2_path = os.path.join(run_dir, 'best_phase2.pth')
    total_epochs = args.warmup_epochs + args.finetune_epochs

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                        handlers=[logging.FileHandler(os.path.join(run_dir, 'train.log')),
                                  logging.StreamHandler()])
    log = logging.getLogger('finetune')
    writer = SummaryWriter(os.path.join(run_dir, 'tensorboard'))

    log.info('=' * 70)
    log.info(f' Fine-tuning {args.model} x{args.scale}  ({run_dir})')
    log.info(f' Warmup {args.warmup_epochs} epochs (L1), then {args.finetune_epochs} epochs '
             f'({"L1 only" if args.loss == "l1" else "L1 + " + args.loss.upper()})')
    log.info(f' LR {args.lr}  batch {args.batch}  crop {args.crop}  WaSP variant {args.wasp_variant}')
    log.info(f' lambda_l1={args.lambda_l1}  lambda_struct={args.lambda_struct}  '
             f'lambda_percept={args.lambda_percept}  lambda_hal={args.lambda_hal}  alpha={args.wasp_scale}')
    log.info('=' * 70)

    model = build_model(args.model, args.scale, device, trainable=True, ckpt_dir=args.ckpt_dir)
    log.info(f'[Model] {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    if args.resume_from:
        sd = torch.load(args.resume_from, map_location='cpu', weights_only=True)
        model.load_state_dict(sd.get('model_state', sd), strict=False)
        log.info(f'[Resume] weights from {args.resume_from}')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step,
                                          gamma=args.lr_decay_gamma)

    wasp = WaSPMetric(**WASP_VARIANTS[args.wasp_variant]).to(device).eval()
    l1_only = nn.L1Loss()
    phase2_loss = build_phase2_loss(args.loss, wasp, device,
                                    weights=(args.lambda_struct, args.lambda_percept, args.lambda_hal),
                                    alpha=args.wasp_scale, lambda_l1=args.lambda_l1)
    train_loader, val_loader = get_loaders(args, log)

    best_wasp, best_wasp_p2, best_psnr, global_step = float('inf'), float('inf'), 0.0, 0
    start = max(1, args.start_epoch)
    for _ in range((start - 1) // args.lr_decay_step):
        scheduler.step()

    for epoch in range(start, total_epochs + 1):
        model.train()
        if hasattr(model, 'vqgan'):
            model.vqgan.eval()
        phase2 = epoch > args.warmup_epochs
        tag = args.loss.upper()[:4].ljust(4) if phase2 else 'L1  '
        e_loss = e_l1 = e_struct = e_percept = e_hal = 0.0
        t0 = time.time()

        for bi, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad(set_to_none=True)
            sr = super_resolve(model, lr, hr.shape[-2:])
            if phase2:
                loss, l_pix, l_s, l_p, l_h = phase2_loss(sr, hr)
            else:
                loss = l_pix = l1_only(sr, hr)
                l_s = l_p = l_h = torch.tensor(0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            e_loss += loss.item()
            e_l1 += l_pix.item()
            e_struct += l_s.item()
            e_percept += l_p.item()
            e_hal += l_h.item()
            global_step += 1
            if (bi + 1) % args.print_freq == 0:
                writer.add_scalar('Step/Loss', loss.item(), global_step)

        scheduler.step()
        nb = max(len(train_loader), 1)
        val_psnr, val_ssim, val_wasp = validate(model, val_loader, wasp, device)
        lr_now = scheduler.get_last_lr()[0]
        log.info(f'[{tag}] Epoch [{epoch:4d}/{total_epochs}] Loss:{e_loss/nb:.4f} L1:{e_l1/nb:.4f} '
                 f'Str:{e_struct/nb:.4f} Per:{e_percept/nb:.4f} Hal:{e_hal/nb:.4f} | '
                 f'PSNR:{val_psnr:.2f}dB SSIM:{val_ssim:.4f} WaSP:{val_wasp:.4f} | '
                 f'LR:{lr_now:.2e} {time.time() - t0:.0f}s')
        writer.add_scalars('Val', {'PSNR': val_psnr, 'SSIM': val_ssim, 'WaSP': val_wasp}, epoch)
        writer.add_scalar('LR', lr_now, epoch)

        if epoch % args.save_freq == 0:
            path = os.path.join(run_dir, f'ep{epoch:04d}.pth')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'psnr': val_psnr, 'wasp': val_wasp}, path)
        if val_wasp < best_wasp:
            best_wasp = val_wasp
            torch.save(model.state_dict(), best_path)
            log.info(f'  best validation WaSP {best_wasp:.4f} (epoch {epoch}) -> {best_path}')
        if phase2 and val_wasp < best_wasp_p2:
            best_wasp_p2 = val_wasp
            torch.save(model.state_dict(), best_p2_path)
            log.info(f'  best Phase-2 validation WaSP {best_wasp_p2:.4f} (epoch {epoch}) -> {best_p2_path}')
        best_psnr = max(best_psnr, val_psnr)

    torch.save(model.state_dict(), final_path)
    log.info(f'Done. Best WaSP={best_wasp:.4f}  best PSNR={best_psnr:.2f} dB  final -> {final_path}')
    writer.close()


if __name__ == '__main__':
    main()
