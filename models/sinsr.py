# SinSR (Wang et al., CVPR 2024): single-step diffusion SR distilled from ResShift (Yue et al.,
# NeurIPS 2023), x4 only. A Swin-UNet runs in the latent space of a VQGAN (f4); both are taken
# from the official SinSR / ResShift code in models/third_party/sinsr/ (CC BY-NC-SA 4.0,
# academic research use only; see the LICENSE there).
#
# forward(lr) replicates the "one_step" path of the official sampler (encode -> one UNet call
# -> decode) without its torch.no_grad(), so the model can be fine-tuned. The prior sample
# draws Gaussian noise on every call: seed torch before inference for reproducible outputs.
# Input/Output: [B, 3, H, W] in [0, 1]; reflect/replicate padding is handled internally.

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._download import download

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party', 'sinsr')
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)


# ── Configuration, copied from the official configs/SinSR.yaml (avoids omegaconf) ──

_UNET_PARAMS = dict(
    image_size=64,
    in_channels=6,
    model_channels=160,
    out_channels=3,
    cond_lq=True,
    attention_resolutions=[64, 32, 16, 8],
    dropout=0,
    channel_mult=[1, 2, 2, 4],
    num_res_blocks=[2, 2, 2, 2],
    conv_resample=True,
    dims=2,
    use_fp16=False,
    num_head_channels=32,
    use_scale_shift_norm=True,
    resblock_updown=False,
    swin_depth=2,
    swin_embed_dim=192,
    window_size=8,
    mlp_ratio=4,
)

_DIFFUSION_PARAMS = dict(
    sf=4,
    schedule_name='exponential',
    schedule_kwargs={'power': 0.3},
    etas_end=0.99,
    steps=15,
    min_noise_level=0.04,
    kappa=2.0,
    weighted_mse=False,
    predict_type='xstart',
    timestep_respacing=None,
    scale_factor=1.0,
    normalize_input=True,
    latent_flag=True,
)

_VQGAN_PARAMS = dict(
    embed_dim=3,
    n_embed=8192,
    ddconfig=dict(
        double_z=False,
        z_channels=3,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        padding_mode='zeros',
    ),
)


class SinSRWrapper(nn.Module):
    """
    Wraps the vendored UNetModelSwin + VQModelTorch + residual-shift diffusion
    schedule into a single differentiable forward(lr) -> sr call, replicating
    exactly the "one_step" inference path of the official Sampler
    (sampler.py:sample_func, one_step=True) but WITHOUT the torch.no_grad()
    that path uses internally (that version is inference-only and blocks
    fine-tuning). Every numerical piece (encode_first_stage, prior_sample,
    _scale_input, decode_first_stage) is the diffusion object's own vendored
    method — only the top-level orchestration and grad-enabling are new code.
    """

    # Max LR side (px) processed in one shot. The model is trained at a fixed
    # 64x64 latent (256x256 HR / 64x64 LR); pushing far beyond that in a single
    # forward blows up memory (confirmed empirically: LR 510x354, DIV2K-val
    # scale, requests ~144GB and OOMs even on a full 95GB H100 — the official
    # Sampler hits the same wall and works around it with chopped/tiled
    # inference, `chop_size=512` HR-equivalent = 128 LR). 128 keeps a single
    # tile's peak well under half of a 47GB MIG slice even with gradients on.
    TILE_MAX = 128
    TILE_OVERLAP = 16

    def __init__(self, unet, vqgan, diffusion):
        super().__init__()
        self.unet = unet
        self.vqgan = vqgan
        self.diffusion = diffusion
        # index of the single step used ("t = T-1", the noisiest/starting point)
        self.register_buffer('_t_index', torch.tensor(diffusion.num_timesteps - 1), persistent=False)

    def _forward_single(self, lr):
        """One-step diffusion forward on a single (already tile-sized) chunk.
        lr: [B, 3, h, w] float in [0, 1] -> [B, 3, h*4, w*4] float in [0, 1]
        """
        # Swin windowed-attention (window_size=8) runs at latent resolutions
        # 64/32/16/8 (attention_resolutions), so the deepest stage (latent/8)
        # must itself be a multiple of 8 -> latent multiple of 64 -> LR input
        # (latent = LR*sf/vqgan_f = LR*4/4 = LR) multiple of 64. Matches the
        # official Sampler's own `desired_min_size=64`. 'replicate' (not
        # 'reflect') since reflect-pad can't exceed input_size-1 and small
        # validation crops can need close to a full 64px of padding.
        b, _, h, w = lr.shape
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h or pad_w:
            lr = F.pad(lr, (0, pad_w, 0, pad_h), mode='replicate')

        y = lr * 2.0 - 1.0  # [0,1] -> [-1,1], matches SinSR's own normalisation

        z_y = self.diffusion.encode_first_stage(y, self.vqgan, up_sample=True)
        noise = torch.randn_like(z_y)
        z_t = self.diffusion.prior_sample(z_y, noise)

        t = self._t_index.expand(b).to(device=z_y.device)
        # `lq` is the PIXEL-space LQ in [-1,1] at LR resolution, not the latent
        # z_y — see sampler.py in the official repo: model_kwargs={'lq': y0}.
        # At sf=4 with the f4 autoencoder both are 3-channel and identically
        # sized, so passing z_y here raises no error but silently feeds the
        # UNet latent codes where it expects RGB (cost: ~12 dB PSNR).
        model_output = self.unet(self.diffusion._scale_input(z_t, t), t, lq=y)  # predict_type=xstart -> direct pred_xstart

        sr = self.diffusion.decode_first_stage(model_output, self.vqgan, no_grad=False)  # straight-through grad through VQ
        sr = (sr.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1,1] -> [0,1]

        if pad_h or pad_w:
            sr = sr[:, :, :h * 4, :w * 4]
        return sr

    @staticmethod
    def _tile_starts(size, tile, stride):
        if size <= tile:
            return [0]
        starts = list(range(0, size - tile + 1, stride))
        if starts[-1] != size - tile:
            starts.append(size - tile)
        return starts

    def forward(self, lr):
        """
        lr: [B, 3, h, w] float in [0, 1]
        returns: [B, 3, h*4, w*4] float in [0, 1]

        Small inputs (training crops, always <= TILE_MAX here) take a single
        differentiable pass through _forward_single. Larger inputs (full-size
        validation/eval images) are processed as overlapping TILE_MAX-sized
        tiles and blended by simple overlap-averaging — required to stay in
        memory budget (see TILE_MAX docstring above), matching the official
        Sampler's own chop-based inference for the same reason. Note: tiling
        with gradients enabled retains every tile's graph until backward(),
        so it scales memory with tile count under grad — fine here since
        training crops in this repo never exceed TILE_MAX, but a much larger
        --crop would need gradient checkpointing to stay tiled-and-trainable.
        """
        b, _, h, w = lr.shape
        if h <= self.TILE_MAX and w <= self.TILE_MAX:
            return self._forward_single(lr)

        tile, overlap = self.TILE_MAX, self.TILE_OVERLAP
        stride = tile - overlap
        out = lr.new_zeros(b, 3, h * 4, w * 4)
        weight = lr.new_zeros(1, 1, h * 4, w * 4)
        ones_tile = lr.new_ones(1, 1, tile * 4, tile * 4)

        for y0 in self._tile_starts(h, tile, stride):
            for x0 in self._tile_starts(w, tile, stride):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                patch = lr[:, :, y0:y1, x0:x1]
                sr_patch = self._forward_single(patch)
                out[:, :, y0 * 4:y1 * 4, x0 * 4:x1 * 4] += sr_patch
                weight[:, :, y0 * 4:y1 * 4, x0 * 4:x1 * 4] += ones_tile[:, :, :sr_patch.shape[-2], :sr_patch.shape[-1]]

        return out / weight.clamp(min=1)


def _load_state_dict(model, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state)


_URLS = {
    'unet': 'https://github.com/wyf0912/SinSR/releases/download/v1.0/SinSR_v1.pth',
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
}


def load_sinsr(scale=4, device='cuda', cache_dir=None):
    """SinSR x4 (official SinSR_v1 UNet + ResShift VQGAN f4), all parameters frozen."""
    from utils.paths import PRETRAINED_DIR
    if scale != 4:
        raise ValueError(f'SinSR only provides official x4 weights (got scale={scale})')
    cache_dir = cache_dir or PRETRAINED_DIR
    paths = {k: os.path.join(cache_dir, url.rsplit('/', 1)[1]) for k, url in _URLS.items()}
    for k, p in paths.items():
        if not os.path.exists(p):
            download(_URLS[k], p)

    from sinsr_core.unet import UNetModelSwin
    from sinsr_core.script_util import create_gaussian_diffusion
    from ldm.models.autoencoder import VQModelTorch

    unet = UNetModelSwin(**_UNET_PARAMS)
    _load_state_dict(unet, paths['unet'])
    vqgan = VQModelTorch(**_VQGAN_PARAMS)
    _load_state_dict(vqgan, paths['vqgan'])
    diffusion = create_gaussian_diffusion(**_DIFFUSION_PARAMS)

    model = SinSRWrapper(unet, vqgan, diffusion).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()
