# EDSR-baseline (Lim et al., CVPRW 2017): 16 residual blocks, 64 channels.
# Official weights: https://github.com/sanghyun-son/EDSR-PyTorch (edsr_baseline_x2, _x4).
# Input/Output: [B, 3, H, W] in [0, 1]; the [0, 255] range and RGB mean shift of the
# original code are applied inside forward().

import os

import torch
import torch.nn as nn

from ._download import download

_URLS = {   # file names carry the SHA-256 prefix (torch.hub convention)
    2: 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    4: 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
}


class ResBlock(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        m = []
        if scale == 4:
            m += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1), nn.PixelShuffle(2),
                  nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1), nn.PixelShuffle(2)]
        elif scale == 2:
            m += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1), nn.PixelShuffle(3)]
        else:
            raise NotImplementedError(f'Unsupported scale: {scale}')
        super().__init__(*m)


class EDSR(nn.Module):
    def __init__(self, scale=4, n_resblocks=16, n_feats=64, res_scale=1.0, in_chans=3):
        super().__init__()
        self.head = nn.Conv2d(in_chans, n_feats, 3, padding=1)
        body = [ResBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)]
        body += [nn.Conv2d(n_feats, n_feats, 3, padding=1)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, in_chans, 3, padding=1),
        )
        # RGB means in [0, 255]; overwritten from the checkpoint's sub_mean layer at load time
        self.register_buffer('rgb_mean', torch.tensor([114.4440, 111.4605, 103.0200]).view(1, 3, 1, 1))
        self.rgb_range = 255.0

    def forward(self, x):
        x = x * self.rgb_range - self.rgb_mean
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return (x + self.rgb_mean) / self.rgb_range


def load_edsr(scale=4, device='cuda', cache_dir=None):
    """EDSR-baseline with the official weights (downloaded to cache_dir on first use)."""
    from utils.paths import PRETRAINED_DIR
    if scale not in _URLS:
        raise ValueError(f'only the x2 and x4 EDSR-baseline weights are supported, not x{scale}')
    path = os.path.join(cache_dir or PRETRAINED_DIR, f'edsr_baseline_x{scale}.pt')
    if not os.path.exists(path):
        download(_URLS[scale], path, sha256_prefix=_URLS[scale].rsplit('-', 1)[1][:-3])

    state = torch.load(path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    # The official checkpoint stores the mean shift as 1x1 convs (sub_mean / add_mean) and
    # wraps the head conv in a Sequential; here the mean is a buffer and the head is plain.
    remap, rgb_mean = {}, None
    for k, v in state.items():
        if k == 'sub_mean.bias':
            rgb_mean = (-v).detach().clone()          # bias holds the negated channel means
            continue
        if k.startswith(('sub_mean.', 'add_mean.')):
            continue
        if k.startswith('head.0.'):
            k = k.replace('head.0.', 'head.', 1)
        remap[k] = v

    model = EDSR(scale=scale)
    missing, unexpected = model.load_state_dict(remap, strict=False)
    if missing != ['rgb_mean'] or unexpected or rgb_mean is None:
        raise RuntimeError(f'EDSR x{scale}: unexpected checkpoint layout '
                           f'(missing={missing}, unexpected={unexpected})')
    with torch.no_grad():
        model.rgb_mean.copy_(rgb_mean.view(1, 3, 1, 1))
    return model.to(device).eval()
