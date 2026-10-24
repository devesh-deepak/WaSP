"""SR data: DIV2K training patches (LR made on the fly) and the benchmark test pairs."""

import glob
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

BENCHMARKS = ('Set5', 'Set14', 'BSD100', 'Urban100')


def list_images(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.png')))
    return files or sorted(glob.glob(os.path.join(folder, '*.jpg')))


def to_tensor(path):
    """[1, 3, H, W] float tensor in [0, 1]."""
    return TF.to_tensor(Image.open(path).convert('RGB')).unsqueeze(0)


# ── Training ────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """Random HR crop (+ horizontal / vertical flips); LR = PIL bicubic downsampling."""

    def __init__(self, hr_files, crop, scale, augment=True):
        self.files = hr_files
        self.crop = crop
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert('RGB')
        w, h = hr.size
        w = (w // self.scale) * self.scale
        h = (h // self.scale) * self.scale
        hr = hr.crop((0, 0, w, h))

        c = self.crop
        if w > c and h > c:
            x = random.randint(0, w - c)
            y = random.randint(0, h - c)
            hr = hr.crop((x, y, x + c, y + c))

        lc = hr.size[0] // self.scale
        lr = hr.resize((lc, lc), Image.BICUBIC)

        hr_t = TF.to_tensor(hr)
        lr_t = TF.to_tensor(lr)
        if self.augment:
            if random.random() > 0.5:
                hr_t = torch.flip(hr_t, [2])
                lr_t = torch.flip(lr_t, [2])
            if random.random() > 0.5:
                hr_t = torch.flip(hr_t, [1])
                lr_t = torch.flip(lr_t, [1])
        return lr_t, hr_t


class ValDataset(Dataset):
    """Full images, HR cropped to a multiple of the scale, LR = PIL bicubic downsampling."""

    def __init__(self, hr_files, scale):
        self.files = hr_files
        self.scale = scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hr = Image.open(self.files[idx]).convert('RGB')
        w, h = hr.size
        w = (w // self.scale) * self.scale
        h = (h // self.scale) * self.scale
        hr = hr.crop((0, 0, w, h))
        lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        return TF.to_tensor(lr), TF.to_tensor(hr)


# ── Benchmarks ──────────────────────────────────────────────────────────────

def benchmark_pairs(dataset, scale, root=None):
    """(HR path, LR path) pairs of a benchmark with the official LR images.

    Layout (the SelfExSR release of Huang et al., 2015):
        <root>/<dataset>/image_SRF_<s>/*_HR.png and *_LR.png
    or with the LR images in their own folder:
        <root>/<dataset>/LR_SRF_<s>/*.png
    """
    from utils.paths import SR_DATA
    base = os.path.join(root or SR_DATA, dataset)
    hrs = sorted(glob.glob(os.path.join(base, f'image_SRF_{scale}', '*HR.png')))
    lr_dir = os.path.join(base, f'LR_SRF_{scale}')
    lrs = (sorted(glob.glob(os.path.join(lr_dir, '*.png'))) if os.path.isdir(lr_dir)
           else sorted(glob.glob(os.path.join(base, f'image_SRF_{scale}', '*LR.png'))))
    if not hrs or len(hrs) != len(lrs):
        raise FileNotFoundError(f'{base}: found {len(hrs)} HR and {len(lrs)} LR images for x{scale}')
    return list(zip(hrs, lrs))


def load_benchmark_pair(hr_path, lr_path, scale, device='cpu'):
    """(LR, HR) tensors [1, 3, H, W]; HR is cropped to a multiple of the scale."""
    hr = Image.open(hr_path).convert('RGB')
    w, h = hr.size
    hr = TF.to_tensor(hr.crop((0, 0, w // scale * scale, h // scale * scale))).unsqueeze(0)
    return to_tensor(lr_path).to(device), hr.to(device)
