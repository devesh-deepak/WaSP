"""SR model zoo: the pretrained models of the paper and loaders for the released checkpoints.

    model = build_model('rcan', scale=4, device='cuda')              # official weights
    load_checkpoint(model, checkpoint_path('rcan', 4, 'wasp'))        # RCAN+WaSP
    sr = super_resolve(model, lr, size=hr.shape[-2:])
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Models fine-tuned in the paper and the scales with official pretrained weights
FINETUNE_MODELS = ('srresnet', 'rcan', 'edsr', 'realesrnet', 'realesrgan', 'swinir', 'sinsr')
SCALES = {'bicubic': (2, 4), 'srresnet': (4,), 'rcan': (2, 4), 'edsr': (2, 4), 'realesrnet': (4,),
          'realesrgan': (2, 4), 'swinir': (2, 4), 'sinsr': (4,), 'imdn': (4,)}
DISPLAY = {'bicubic': 'Bicubic', 'srresnet': 'SRResNet', 'rcan': 'RCAN', 'edsr': 'EDSR',
           'realesrnet': 'RealESRNet', 'realesrgan': 'RealESRGAN', 'swinir': 'SwinIR',
           'sinsr': 'SinSR', 'imdn': 'IMDN'}


class Bicubic(nn.Module):
    """Bicubic upsampling (torch, align_corners=False), as a model."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)


def checkpoint_path(name, scale, variant, ckpt_dir=None):
    """Released checkpoint <ckpt_dir>/x<scale>/<name>_<variant>.pth.

    variant: 'wasp' (+WaSP), 'ctrl_<l1|lpips|dists|msssim>' (loss controls),
             'abla_<la|lb|lc|lalb>' (branch ablations), or 'l1' (SRResNet-L1 baseline).
    """
    from utils.paths import CKPT_DIR
    return os.path.join(ckpt_dir or CKPT_DIR, f'x{scale}', f'{name}_{variant}.pth')


def build_model(name, scale=4, device='cuda', trainable=False, ckpt_dir=None, cache_dir=None):
    """The pretrained starting point of a model.

    Official weights are downloaded to cache_dir (default: utils.paths.PRETRAINED_DIR).
    SRResNet has no official checkpoint: its starting point is the L1-trained SRResNet of
    training/train_srresnet_l1.py, read from <ckpt_dir>/x4/srresnet_l1.pth.

    trainable=True puts the model in train mode with the parameters that are fine-tuned in
    the paper unfrozen (all of them, except the frozen VQGAN of SinSR); otherwise the model is
    in eval mode with every parameter frozen.
    """
    if scale not in SCALES.get(name, ()):
        raise ValueError(f'{name} x{scale} is not supported (scales: {SCALES.get(name)})')
    if name == 'bicubic':
        model = Bicubic(scale)
    elif name == 'srresnet':
        from .srresnet import SRResNet
        model = SRResNet(upscale_factor=scale)
        sd = torch.load(checkpoint_path('srresnet', scale, 'l1', ckpt_dir), map_location='cpu',
                        weights_only=True)
        model.load_state_dict(sd)
    elif name == 'rcan':
        from .rcan import load_rcan
        model = load_rcan(scale, device, cache_dir)
    elif name == 'edsr':
        from .edsr import load_edsr
        model = load_edsr(scale, device, cache_dir)
    elif name == 'realesrnet':
        from .rrdbnet import load_realesrnet
        model = load_realesrnet(scale, device, cache_dir)
    elif name == 'realesrgan':
        from .rrdbnet import load_realesrgan
        model = load_realesrgan(scale, device, cache_dir)
    elif name == 'swinir':
        from .swinir import load_swinir
        model = load_swinir(scale, device, cache_dir)
    elif name == 'sinsr':
        from .sinsr import load_sinsr
        model = load_sinsr(scale, device, cache_dir)
    elif name == 'imdn':
        from .imdn import load_imdn
        model = load_imdn(scale, device, cache_dir)
    else:
        raise ValueError(f'unknown model {name!r}')

    model = model.to(device)
    if trainable:
        model.train()
        if name == 'sinsr':
            # Only the UNet is fine-tuned; the VQGAN stays frozen, as in the official recipe
            for p in model.unet.parameters():
                p.requires_grad_(True)
            model.vqgan.eval()
        else:
            for p in model.parameters():
                p.requires_grad_(True)
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def load_checkpoint(model, path):
    """Load a fine-tuned checkpoint into a model from build_model().

    Accepts plain state dicts and training checkpoints ({'model_state': ...}). SinSR
    checkpoints may hold the UNet only; the frozen VQGAN then keeps its official weights.
    """
    sd = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state' in sd:
        sd = sd['model_state']
    result = model.load_state_dict(sd, strict=False)
    optional = ('vqgan.',) if hasattr(model, 'vqgan') else ()
    missing = [k for k in result.missing_keys if not k.startswith(optional)]
    if missing or result.unexpected_keys:
        raise RuntimeError(f'{path}: {len(missing)} missing keys {missing[:3]}, '
                           f'{len(result.unexpected_keys)} unexpected {result.unexpected_keys[:3]}')
    return model


def super_resolve(model, lr, size=None):
    """model(lr) clamped to [0, 1]; resized (bilinear) to `size` if the output differs from it."""
    sr = model(lr)
    if isinstance(sr, (tuple, list)):
        sr = sr[0]
    if hasattr(sr, 'reconstruction'):
        sr = sr.reconstruction
    sr = sr.clamp(0, 1)
    if size is not None and tuple(sr.shape[-2:]) != tuple(size):
        sr = F.interpolate(sr, size=tuple(size), mode='bilinear', align_corners=False).clamp(0, 1)
    return sr
