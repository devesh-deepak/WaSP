# RCAN (Zhang et al., ECCV 2018): 10 residual groups x 20 RCABs, 64 channels.
# Module names match the official checkpoints (yulunzhang/RCAN, RCAN_BIX{2,3,4}.pt) so they
# load with strict=True. Input/Output: [B, 3, H, W] in [0, 1].

import os
import zipfile

import torch
import torch.nn as nn


# ── Channel Attention Layer ───────────────────────────────────────────────────

class CCALayer(nn.Module):
    """
    Channel Attention using global average-pool + squeeze-excite.
    Keys: conv_du.0 (squeeze conv), conv_du.2 (excite conv).
    The nn.Sequential has ReLU at index 1 (no params), Sigmoid at index 3.
    """
    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du  = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=True),  # conv_du.0
            nn.ReLU(inplace=True),                                    # conv_du.1
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=True),  # conv_du.2
            nn.Sigmoid(),                                             # conv_du.3
        )

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


# ── RCAB (Residual Channel Attention Block) ───────────────────────────────────
# Checkpoint key pattern for one RCAB (index m inside group n):
#   body.n.body.m.body.0   — Conv2d (conv1)
#   body.n.body.m.body.1   — ReLU   (no params)
#   body.n.body.m.body.2   — Conv2d (conv2)
#   body.n.body.m.body.3   — CCALayer (conv_du.0 and conv_du.2)
# All wrapped in nn.Sequential named 'body'

class RCAB(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True),   # .body.0
            nn.ReLU(inplace=True),                               # .body.1
            nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True),   # .body.2
            CCALayer(n_feats, reduction),                        # .body.3
        )

    def forward(self, x):
        return x + self.body(x)            # residual scaling=1.0 (RCAN default)


# ── ResidualGroup ─────────────────────────────────────────────────────────────
# Checkpoint key pattern inside group n:
#   body.n.body.0 .. body.n.body.19  — RCAB blocks
#   body.n.body.20                   — group-level Conv2d (no RCAB)
# All wrapped in nn.Sequential named 'body'

class ResidualGroup(nn.Module):
    def __init__(self, n_feats, n_resblocks=20, reduction=16):
        super().__init__()
        modules = [RCAB(n_feats, reduction) for _ in range(n_resblocks)]
        modules.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))  # group conv
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)


# ── RCAN ─────────────────────────────────────────────────────────────────────

class RCAN(nn.Module):
    """
    RCAN with checkpoint-aligned architecture.
    Key names match the official RCAN_BIX4.pt from yulunzhang/RCAN exactly.

    sub_mean / add_mean : 1×1 Conv2d mean-shift layers
    head                : nn.Sequential(Conv2d) — hence 'head.0'
    body                : nn.Sequential of 10 ResidualGroups + final conv
                          (body.10 = group-level conv after all groups)
    tail                : nn.Sequential(
                            nn.Sequential(PixelShuffle×2, PixelShuffle×2),  [tail.0]
                            nn.Conv2d output                                 [tail.1]
                          )
    """

    def __init__(self, scale=4, n_colors=3, n_feats=64,
                 n_resgroups=10, n_resblocks=20, reduction=16):
        super().__init__()
        self.rgb_range = 255.0

        # ── Mean-shift layers (1×1 convs to subtract/add RGB mean) ──
        self.sub_mean = nn.Conv2d(n_colors, n_colors, 1, bias=True)
        self.add_mean = nn.Conv2d(n_colors, n_colors, 1, bias=True)

        # ── Head: single conv wrapped in Sequential → key = head.0 ─────
        self.head = nn.Sequential(
            nn.Conv2d(n_colors, n_feats, 3, 1, 1, bias=True),   # head.0
        )

        # ── Body: n_resgroups ResidualGroups + final conv ────────────────
        modules_body = [ResidualGroup(n_feats, n_resblocks, reduction)
                        for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True))
        self.body = nn.Sequential(*modules_body)

        # ── Tail: pixel-shuffle upsampler + output conv ──────────────────
        # For 4× scale: two PixelShuffle(2) steps, each inside a Sequential
        # tail.0 = nn.Sequential(conv, PixelShuffle, conv, PixelShuffle)
        #          but checkpoint keys are tail.0.0 and tail.0.2 → two convs
        # tail.1 = output conv (n_feats → n_colors)
        if scale == 4:
            upsample = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1, bias=True),  # tail.0.0
                nn.PixelShuffle(2),                                      # tail.0.1
                nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1, bias=True),  # tail.0.2
                nn.PixelShuffle(2),                                      # tail.0.3
            )
        elif scale == 2:
            upsample = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1, bias=True),
                nn.PixelShuffle(2),
            )
        elif scale == 3:
            upsample = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 9, 3, 1, 1, bias=True),
                nn.PixelShuffle(3),
            )

        self.tail = nn.Sequential(
            upsample,                                                  # tail.0
            nn.Conv2d(n_feats, n_colors, 3, 1, 1, bias=True),        # tail.1
        )

    def forward(self, x):
        # [0,1] → [0,255] → subtract mean
        x = x * self.rgb_range
        x = self.sub_mean(x)
        # SR
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        # add mean → clamp to [0,255] → [0,1]
        x = self.add_mean(x)
        return (x / self.rgb_range).clamp(0., 1.)


# Official model zip of yulunzhang/RCAN on Google Drive (all scales)
_GDRIVE_FILE_ID = '1U0RmWkkacyw0HNC7CBts2LcX1ewAulCn'


def load_rcan(scale=4, device='cuda', cache_dir=None):
    """RCAN-BIX{scale} with the official weights (downloaded with gdown on first use)."""
    from utils.paths import PRETRAINED_DIR
    cache_dir = cache_dir or PRETRAINED_DIR
    path = os.path.join(cache_dir, f'rcan_bix{scale}.pth')
    if not os.path.exists(path):
        zip_path = os.path.join(cache_dir, 'RCAN_models.zip')
        if not os.path.exists(zip_path):
            import gdown
            os.makedirs(cache_dir, exist_ok=True)
            gdown.download(id=_GDRIVE_FILE_ID, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path) as zf:
            names = [f for f in zf.namelist() if f.upper().endswith(f'BIX{scale}.PT')]
            if not names:
                raise FileNotFoundError(f'no RCAN_BIX{scale}.pt in {zip_path}')
            with zf.open(names[0]) as src, open(path, 'wb') as dst:
                dst.write(src.read())

    model = RCAN(scale=scale)
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
