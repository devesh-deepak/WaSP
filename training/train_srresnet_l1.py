#!/usr/bin/env python3
"""
train_srresnet_l1.py — trains the SRResNet-L1 baseline from scratch (x4).

SRResNet has no official PyTorch checkpoint, so the paper's SRResNet starting point is this
model: L1 loss, Adam (lr 2e-4), StepLR (x0.5 every 200 epochs), 1000 epochs, batch 16, random
96x96 HR crops of the DIV2K training set with horizontal / vertical flips, LR by PIL bicubic.
The final checkpoint is used (checkpoints/x4/srresnet_l1.pth).

  python training/train_srresnet_l1.py --out runs/srresnet_l1
"""

import argparse
import glob
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.srresnet import SRResNet
from utils.metrics import calculate_psnr, calculate_ssim
from utils.paths import DIV2K_TRAIN_HR, DIV2K_VALID_HR, REPO


class SRDataset(Dataset):
    """Random HR crop (training) or full image (validation); LR = PIL bicubic of the HR."""

    def __init__(self, hr_dir, crop_size=None, scale=4, augment=True, limit=None):
        self.files = sorted(glob.glob(os.path.join(hr_dir, '*.png')))[:limit]
        if not self.files:
            raise ValueError(f'No images in {hr_dir}')
        self.crop_size, self.scale, self.augment = crop_size, scale, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert('RGB')
        if self.crop_size is not None:
            w, h = hr.size
            if w >= self.crop_size and h >= self.crop_size:
                i = random.randint(0, h - self.crop_size)
                j = random.randint(0, w - self.crop_size)
                hr = TF.crop(hr, i, j, self.crop_size, self.crop_size)
        if self.augment:
            if random.random() > 0.5:
                hr = TF.hflip(hr)
            if random.random() > 0.5:
                hr = TF.vflip(hr)
        lr = hr.resize((hr.size[0] // self.scale, hr.size[1] // self.scale), Image.BICUBIC)
        return TF.to_tensor(lr), TF.to_tensor(hr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hr_dir', default=DIV2K_TRAIN_HR)
    p.add_argument('--val_dir', default=DIV2K_VALID_HR, help='first 20 images, for monitoring only')
    p.add_argument('--out', default=os.path.join(REPO, 'runs', 'srresnet_l1'))
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--crop', type=int, default=96)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--lr_decay_step', type=int, default=200)
    p.add_argument('--lr_decay_gamma', type=float, default=0.5)
    p.add_argument('--save_freq', type=int, default=100)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.out, 'train.log')),
                                  logging.StreamHandler()])
    writer = SummaryWriter(os.path.join(args.out, 'tensorboard'))

    model = SRResNet(upscale_factor=args.scale).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step,
                                          gamma=args.lr_decay_gamma)
    train_loader = DataLoader(SRDataset(args.hr_dir, args.crop, args.scale, augment=True),
                              batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(SRDataset(args.val_dir, None, args.scale, augment=False, limit=20),
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            loss = criterion(model(lr), hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        psnr = ssim = 0.0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr).clamp(0, 1)
                psnr += calculate_psnr(sr, hr).item()
                ssim += calculate_ssim(sr, hr).item()
        psnr, ssim = psnr / len(val_loader), ssim / len(val_loader)
        logging.info(f'Epoch [{epoch}/{args.epochs}] Loss: {epoch_loss / len(train_loader):.5f} | '
                     f'PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}')
        writer.add_scalar('Loss/train_epoch', epoch_loss / len(train_loader), epoch)
        writer.add_scalar('Metrics/PSNR', psnr, epoch)
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.out, f'ep{epoch:04d}.pth'))

    torch.save(model.state_dict(), os.path.join(args.out, 'final.pth'))
    logging.info(f'Final model -> {os.path.join(args.out, "final.pth")}')
    writer.close()


if __name__ == '__main__':
    main()
