# IMDN (Hui et al., ACM MM 2019): 6 information multi-distillation blocks, 64 channels, x4.
# Transcribed from github.com/Zheng222/IMDN; module names match the official IMDN_x4.pth.
# Used only as a pretrained baseline. Input/Output: [B, 3, H, W] in [0, 1].

import os

import torch
import torch.nn as nn

from ._download import download


# ── Contrast-Aware Channel Attention ─────────────────────────────────────────

class CCALayer(nn.Module):
    """
    Contrast-Aware Channel Attention (CCA).
    Operates on the 64-channel distilled concatenation.
    Keys: conv_du.0 (squeeze) and conv_du.2 (excite).
    Checkpoint shape: conv_du.0 = [4, 64, 1,1] → reduction=16.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du  = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),  # conv_du.0
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),  # conv_du.2
            nn.Sigmoid(),
        )

    def forward(self, x):
        # contrast = pixel-wise std across H,W then global-avg-pooled
        contrast = x.var(dim=[2, 3], keepdim=True).sqrt()
        y = contrast + self.avg_pool(x)
        return x * self.conv_du(y)


# ── IMDModule (Information Multi-Distillation Block) ─────────────────────────

class IMDModule(nn.Module):
    """
    Exact transcription of IMDModule from official IMDN.
    Checkpoint layout verified — key names match directly:
      c1, c2, c3, c4: plain Conv2d (NO Sequential wrapper)
      c5:             Conv2d(in_channels, in_channels, 1)  ← AFTER cca
      cca:            CCALayer(distilled_ch * 4)

    Forward order: act(c1) → split → act(c2) → split → act(c3) → split →
                   c4 → cat(d1,d2,d3,d4) → cca(cat) → c5(cca_out) + input
    """
    def __init__(self, in_channels, distillation_rate=0.25):
        super().__init__()
        self.distilled_channels = int(in_channels * distillation_rate)   # 16
        self.remaining_channels = int(in_channels - self.distilled_channels)  # 48
        self.c1  = nn.Conv2d(in_channels,             in_channels,              3, 1, 1, bias=True)
        self.c2  = nn.Conv2d(self.remaining_channels, in_channels,              3, 1, 1, bias=True)
        self.c3  = nn.Conv2d(self.remaining_channels, in_channels,              3, 1, 1, bias=True)
        self.c4  = nn.Conv2d(self.remaining_channels, self.distilled_channels,  3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.05, inplace=True)
        # c5 fuses the CCA-attended 64-ch concat → back to in_channels
        self.c5  = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        # CCA operates on distilled_channels×4 = 64-ch
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        d1, r1 = torch.split(out_c1, [self.distilled_channels, self.remaining_channels], dim=1)
        out_c2 = self.act(self.c2(r1))
        d2, r2 = torch.split(out_c2, [self.distilled_channels, self.remaining_channels], dim=1)
        out_c3 = self.act(self.c3(r2))
        d3, r3 = torch.split(out_c3, [self.distilled_channels, self.remaining_channels], dim=1)
        d4     = self.c4(r3)                              # no activation on c4
        out    = torch.cat([d1, d2, d3, d4], dim=1)      # [B, 64, H, W]
        # cca FIRST, then c5, then residual add
        out_fused = self.c5(self.cca(out)) + input
        return out_fused


# ── IMDN ─────────────────────────────────────────────────────────────────────

class IMDN(nn.Module):
    """
    IMDN — 6 IMDModules, 64 features, 4× pixel-shuffle.
    Layer naming matches official checkpoint exactly.
    """
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super().__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # IMDBs registered as IMDB1..IMDB6 so checkpoint keys map directly
        for i in range(1, num_modules + 1):
            self.add_module(f'IMDB{i}', IMDModule(nf))
        self.num_modules = num_modules

        # Fusion conv: conv_block(nf*6, nf, 1, act='lrelu')
        # Checkpoint key: c.0.weight (Conv), no key for LeakyReLU
        self.c = nn.Sequential(
            nn.Conv2d(nf * num_modules, nf, 1, bias=True),  # c.0
            nn.LeakyReLU(0.05, inplace=True),               # c.1 (no params)
        )

        self.LR_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # upsampler: conv_layer(nf, out_nc, ks=3) + PixelShuffle(upscale)
        # Checkpoint upsampler.0.weight = [48, 64, 3, 3]  (48 = 16×3)
        self.upsampler = nn.Sequential(
            nn.Conv2d(nf, out_nc * upscale * upscale, 3, 1, 1, bias=True),  # upsampler.0
            nn.PixelShuffle(upscale),                                         # upsampler.1
        )

    def forward(self, x):
        out_fea = self.fea_conv(x)
        block_outs = []
        cur = out_fea
        for i in range(1, self.num_modules + 1):
            cur = getattr(self, f'IMDB{i}')(cur)
            block_outs.append(cur)
        out_B  = self.c(torch.cat(block_outs, dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        return self.upsampler(out_lr).clamp(0., 1.)


_URL = 'https://raw.githubusercontent.com/Zheng222/IMDN/master/checkpoints/IMDN_x4.pth'


def load_imdn(scale=4, device='cuda', cache_dir=None):
    """IMDN x4 with the official weights."""
    from utils.paths import PRETRAINED_DIR
    if scale != 4:
        raise ValueError(f'IMDN is used at x4 only, not x{scale}')
    path = os.path.join(cache_dir or PRETRAINED_DIR, 'imdn_x4.pth')
    if not os.path.exists(path):
        download(_URL, path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model = IMDN(upscale=scale)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
