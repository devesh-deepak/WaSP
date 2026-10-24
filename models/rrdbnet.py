# RRDBNet generator (ESRGAN, Wang et al., ECCVW 2018) with the official Real-ESRGAN weights
# (Wang et al., ICCVW 2021): RealESRNet_x4plus (L1-trained) and RealESRGAN_x{2,4}plus (GAN).
# Module names match xinntao's checkpoints so they load with strict=True.
# Input/Output: [B, 3, H, W] in [0, 1].

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._download import download


# ── Residual Dense Block — conv naming matches xinntao checkpoints ─────────────

class ResidualDenseBlock(nn.Module):
    """
    5-layer dense block with residual scaling (beta=0.2).
    Layer names: conv1..conv5 to match xinntao/Real-ESRGAN checkpoint keys.
    """
    def __init__(self, in_ch=64, growth_ch=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(in_ch + 0 * growth_ch, growth_ch, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_ch + 1 * growth_ch, growth_ch, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(in_ch + 2 * growth_ch, growth_ch, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(in_ch + 3 * growth_ch, growth_ch, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(in_ch + 4 * growth_ch, in_ch,    3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + self.beta * x5


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block — rdb1/rdb2/rdb3 names match checkpoints."""
    def __init__(self, in_ch=64, growth_ch=32, beta=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(in_ch, growth_ch, beta)
        self.rdb2 = ResidualDenseBlock(in_ch, growth_ch, beta)
        self.rdb3 = ResidualDenseBlock(in_ch, growth_ch, beta)
        self.beta = beta

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + self.beta * out


# ── RRDBNet Generator ─────────────────────────────────────────────────────────

class RRDBNet(nn.Module):
    """
    RRDB-based SR generator (ESRGAN / RealESRGAN backbone).
    Layer naming exactly matches xinntao/Real-ESRGAN checkpoint keys:
      conv_first, body.{i}.{rdb1/rdb2/rdb3}.conv{1..5},
      conv_body, conv_up1, conv_up2, conv_hr, conv_last.
    """

    def __init__(self, in_ch=3, out_ch=3, n_feats=64, n_blocks=23,
                 growth_ch=32, scale=4):
        super().__init__()
        self.scale = scale

        # x2 models (RealESRGAN_x2plus) pixel-unshuffle the input by 2 first, so
        # conv_first sees 4x the channels and the two x2 upsamplers give x2 overall
        self.conv_first = nn.Conv2d(in_ch * (4 if scale == 2 else 1), n_feats, 3, 1, 1, bias=True)
        self.body       = nn.Sequential(*[RRDB(n_feats, growth_ch) for _ in range(n_blocks)])
        self.conv_body  = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)

        # Upsampling: nearest-neighbour × 2 twice, then hr + last conv
        self.conv_up1   = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        self.conv_up2   = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        self.conv_hr    = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)
        self.conv_last  = nn.Conv2d(n_feats, out_ch, 3, 1, 1, bias=True)
        self.lrelu      = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.scale == 2:
            # pixel_unshuffle needs even sizes: reflect-pad, crop the output back
            x = F.pad(x, (0, w % 2, 0, h % 2), mode='reflect')
            x = F.pixel_unshuffle(x, 2)
        feat  = self.conv_first(x)
        body  = self.conv_body(self.body(feat))
        feat  = feat + body
        feat  = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat  = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out   = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out[..., :h * self.scale, :w * self.scale].clamp(0., 1.)


_URLS = {
    ('realesrgan', 4): 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    ('realesrgan', 2): 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    ('realesrnet', 4): 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
}


def _load_rrdb(kind, scale, device, cache_dir):
    from utils.paths import PRETRAINED_DIR
    if (kind, scale) not in _URLS:
        raise ValueError(f'{kind} has no official x{scale} weights')
    url = _URLS[(kind, scale)]
    path = os.path.join(cache_dir or PRETRAINED_DIR, url.rsplit('/', 1)[1])
    if not os.path.exists(path):
        download(url, path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    for k in ('params_ema', 'params', 'state_dict', 'model'):
        if isinstance(state, dict) and k in state:
            state = state[k]
            break
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model = RRDBNet(n_blocks=23, scale=scale)
    # Strict on purpose: a silent partial load once gave x2 runs x4 weights
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_realesrgan(scale=4, device='cuda', cache_dir=None):
    """RealESRGAN_x{2,4}plus (RRDB x 23, GAN-trained)."""
    return _load_rrdb('realesrgan', scale, device, cache_dir)


def load_realesrnet(scale=4, device='cuda', cache_dir=None):
    """RealESRNet_x4plus (RRDB x 23, L1-trained; x4 only)."""
    return _load_rrdb('realesrnet', scale, device, cache_dir)
