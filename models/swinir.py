# "SwinIR" in the paper: the Swin2SR classical-SR models (Conde et al., ECCVW 2022; a SwinV2
# version of SwinIR, Liang et al., 2021) from HuggingFace transformers:
#   caidas/swin2sr-classical-sr-x2-64 and caidas/swin2sr-classical-sr-x4-64
#   (embed_dim 180, 6 residual Swin blocks of depth 6, window 8; 12.2M parameters)
# Input/Output: [B, 3, H, W] in [0, 1].

import torch
import torch.nn.functional as F

_REPOS = {2: 'caidas/swin2sr-classical-sr-x2-64',
          4: 'caidas/swin2sr-classical-sr-x4-64'}


class Swin2SRWrapper(torch.nn.Module):
    """Reflect-pads the input to a multiple of the window size and crops the output back."""

    def __init__(self, hf_model, window_size=8, upscale=4):
        super().__init__()
        self.model = hf_model
        self.window_size = window_size
        self.upscale = upscale

    def _pad(self, x):
        _, _, h, w = x.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w

    def forward(self, x):
        x_padded, h, w = self._pad(x)
        out = self.model(pixel_values=x_padded).reconstruction
        out = out[:, :, :h * self.upscale, :w * self.upscale]
        return out.clamp(0.0, 1.0)


def load_swinir(scale=4, device='cuda', cache_dir=None):
    """Swin2SR classical-SR model for x2 or x4 (weights cached in cache_dir)."""
    from transformers import Swin2SRForImageSuperResolution
    from utils.paths import PRETRAINED_DIR
    if scale not in _REPOS:
        raise ValueError(f'Swin2SR classical-SR weights exist for x2 and x4 only, not x{scale}')
    hf_model = Swin2SRForImageSuperResolution.from_pretrained(_REPOS[scale],
                                                              cache_dir=cache_dir or PRETRAINED_DIR)
    return Swin2SRWrapper(hf_model, window_size=8, upscale=scale).to(device).eval()
