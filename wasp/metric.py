# WaSP: Wavelet-aware Super-resolution Perceptual metric.
#
#   SR / HR images [B, 3, H, W] in [0, 1]
#         |
#   MultiLevelDWT (3-level Haar, fixed)  ->  LL_3  +  {LH_j, HL_j, HH_j}, j = 1..3
#         |
#   Branch A  structural    : MS-SSIM on the coarsest LL band (divided by 2^J)
#   Branch B  perceptual    : spatial L1 between LPIPS-normalised AlexNet features
#   Branch C  hallucination : excess SR high-frequency energy inside the smooth
#                             regions of HR (per-image 25th-percentile threshold)
#         |
#   score = l_s * tanh(L_A / 0.30) + l_p * tanh(L_B / 0.05) + l_h * tanh(L_C / 0.01)
#   (lower = better; default weights (l_s, l_p, l_h) = (1.0, 0.5, 0.1))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

DEFAULT_LAMBDAS = (1.0, 0.5, 0.1)    # (lambda_s, lambda_p, lambda_h)
NORM_SCALES = (0.30, 0.05, 0.01)     # tanh soft-normalisation constants of L_A, L_B, L_C


def score_from_components(l_struct, l_percept, l_hal, lambdas=DEFAULT_LAMBDAS):
    """WaSP score from raw (un-normalised) branch values; works on scalars or NumPy arrays."""
    return (lambdas[0] * np.tanh(np.asarray(l_struct) / NORM_SCALES[0])
            + lambdas[1] * np.tanh(np.asarray(l_percept) / NORM_SCALES[1])
            + lambdas[2] * np.tanh(np.asarray(l_hal) / NORM_SCALES[2]))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Multi-level discrete wavelet transform (Haar, fixed basis)
# ─────────────────────────────────────────────────────────────────────────────

class MultiLevelDWT(nn.Module):
    """
    Applies the 2D Haar DWT recursively for `levels` levels.

    Returns:
        ll : [B, C, H/2^L, W/2^L]  coarsest low-frequency approximation
        hf : list of length `levels`, each a dict {'LH', 'HL', 'HH'} at that scale;
             hf[0] = finest (level 1), hf[-1] = coarsest (level L)

    Haar filters (orthonormal), with x1..x4 the four pixels of a 2x2 block divided by 2:
        LL =  x1 + x2 + x3 + x4
        LH = -x1 + x2 - x3 + x4
        HL = -x1 - x2 + x3 + x4
        HH =  x1 - x2 - x3 + x4
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels
        self.requires_grad = False

    def haar_step(self, x):
        """Single-level 2D Haar DWT of x [B, C, H, W]; odd H/W are reflect-padded."""
        _, _, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x01 = x[:, :, 0::2, :] / 2   # even rows
        x02 = x[:, :, 1::2, :] / 2   # odd rows
        x1  = x01[:, :, :, 0::2]
        x2  = x02[:, :, :, 0::2]
        x3  = x01[:, :, :, 1::2]
        x4  = x02[:, :, :, 1::2]

        LL = x1 + x2 + x3 + x4
        LH = -x1 + x2 - x3 + x4
        HL = -x1 - x2 + x3 + x4
        HH =  x1 - x2 - x3 + x4
        return LL, LH, HL, HH

    def forward(self, x):
        ll = x
        hf_bands = []
        for _ in range(self.levels):
            ll, lh, hl, hh = self.haar_step(ll)
            hf_bands.append({'LH': lh, 'HL': hl, 'HH': hh})
        return ll, hf_bands


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MS-SSIM (Branch A)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_kernel(window_size: int, sigma: float, channels: int, device):
    """2D Gaussian kernel [channels, 1, ws, ws] for depthwise SSIM filtering."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size)
    return kernel_2d.contiguous()


def ssim_cs(x, y, window_size=11, sigma=1.5, data_range=1.0, eps=1e-8):
    """Mean SSIM and mean contrast-structure (cs) term of x, y [B, C, H, W] at one scale."""
    C = x.shape[1]
    kernel = gaussian_kernel(window_size, sigma, C, x.device)
    pad = window_size // 2

    mu_x  = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y  = F.conv2d(y, kernel, padding=pad, groups=C)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sig_x2 = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
    sig_y2 = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
    sig_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2) + eps)
    cs_map = (2 * sig_xy + C2) / (sig_x2 + sig_y2 + C2)
    return ssim_map.mean(), cs_map.mean()


def ms_ssim_loss(x, y, levels=3, weights=None, window_size=11, data_range=1.0, form='product'):
    """
    Multi-scale SSIM loss (1 - MS-SSIM) at up to `levels` scales (2x average pooling).
    A scale is evaluated only if both dimensions are >= window_size; the weights (the
    first `levels` of Wang et al., 2003) are renormalised over the scales evaluated.

    form='product' (default): standard MS-SSIM, the product of the cs terms of the finer
        scales and the SSIM of the coarsest scale, each raised to its weight (negative
        terms clipped at 0, as in common implementations).
    form='mean': weighted arithmetic mean of the SSIM at each scale (the form of an
        earlier version of the paper).
    With a single scale both forms give the same value (for SSIM >= 0).
    x, y: [B, C, H, W] in [0, data_range]. Returns a scalar loss in [0, 1].
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]

    ssim_vals, cs_vals = [], []
    x_curr, y_curr = x, y
    for i in range(levels):
        if x_curr.shape[-1] < window_size or x_curr.shape[-2] < window_size:
            break
        s, c = ssim_cs(x_curr, y_curr, window_size=window_size, data_range=data_range)
        ssim_vals.append(s)
        cs_vals.append(c)
        if i < levels - 1:
            x_curr = F.avg_pool2d(x_curr, kernel_size=2, stride=2)
            y_curr = F.avg_pool2d(y_curr, kernel_size=2, stride=2)

    if not ssim_vals:
        return torch.tensor(0.0, device=x.device)

    w = weights[:len(ssim_vals)]
    w_sum = sum(w)
    w = [wi / w_sum for wi in w]
    if form == 'mean':
        ms_ssim_val = sum(wi * s for wi, s in zip(w, ssim_vals))
    else:
        ms_ssim_val = torch.relu(ssim_vals[-1]) ** w[-1]
        for wi, c in zip(w[:-1], cs_vals[:-1]):
            ms_ssim_val = ms_ssim_val * torch.relu(c) ** wi
    return 1.0 - ms_ssim_val


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Perceptual feature extractors (Branch B)
# ─────────────────────────────────────────────────────────────────────────────

class AlexNetFeatureExtractor(nn.Module):
    """
    ImageNet AlexNet features at its 5 ReLU outputs (64, 192, 384, 256, 256 channels),
    with LPIPS input normalisation and channel-wise unit normalisation. AlexNet is the
    LPIPS backbone with the best agreement with human judgements (Zhang et al., 2018).
    All weights are frozen. Input: [B, 3, H, W] in [0, 1].
    """

    LAYER_INDICES = [1, 4, 7, 9, 11]
    LAYER_WEIGHTS = [0.1, 0.15, 0.25, 0.25, 0.25]

    def __init__(self):
        super().__init__()
        alex = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.LAYER_INDICES:
            self.slices.append(nn.Sequential(*list(alex.children())[prev:idx + 1]))
            prev = idx + 1
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('shift', torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer('scale', torch.tensor([ 0.458,  0.448,  0.450]).view(1, 3, 1, 1))

    def normalize(self, x):
        x = x * 2.0 - 1.0
        return (x - self.shift) / self.scale

    def forward(self, x):
        x = self.normalize(x)
        features = []
        for s in self.slices:
            x = s(x)
            features.append(x / (x.norm(dim=1, keepdim=True) + 1e-10))
        return features


class VGGFeatureExtractor(nn.Module):
    """VGG19 features at relu1_2, relu2_2, relu3_4, relu4_4 (backbone ablation only)."""

    LAYER_INDICES = [3, 8, 17, 26]
    LAYER_WEIGHTS = [0.1, 0.2, 0.3, 0.4]

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.LAYER_INDICES:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:idx + 1]))
            prev = idx + 1
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        x = self.normalize(x)
        features = []
        for s in self.slices:
            x = s(x)
            features.append(x)
        return features


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WaSP
# ─────────────────────────────────────────────────────────────────────────────

class WaSPMetric(nn.Module):
    """
    WaSP: Wavelet-aware Super-resolution Perceptual metric (lower = better).

    Args:
        lambdas     : branch weights (lambda_s, lambda_p, lambda_h) of the score,
                      default (1.0, 0.5, 0.1).
        dwt_levels  : number of Haar levels J (default 3).
        backbone    : Branch B features, 'alexnet' (default) or 'vgg19' (ablation).
        hal_quantile: smooth-region quantile q of Branch C (default 0.25).
        msssim_form : 'product' (default, standard MS-SSIM) or 'mean' (weighted
                      arithmetic mean of SSIM over scales, an earlier revision).
        legacy      : reproduce the originally submitted metric (default False):
                      Branch A clamps the LL band to [0, 1] instead of dividing it by
                      2^J and uses the 'mean' MS-SSIM form, and Branch C uses a strict
                      '<' smooth-region mask.

    All parameters are frozen. The metric is differentiable with respect to its inputs,
    so it can be used as a training loss.

    Usage:
        metric = WaSPMetric().to(device).eval()
        score, l_struct, l_percept, l_hal = metric(sr, hr)
    """

    def __init__(self, lambdas=DEFAULT_LAMBDAS, dwt_levels: int = 3, backbone: str = 'alexnet',
                 hal_quantile: float = 0.25, msssim_form: str = 'product', legacy: bool = False):
        super().__init__()
        if msssim_form not in ('product', 'mean'):
            raise ValueError(f"msssim_form must be 'product' or 'mean', got {msssim_form!r}")
        self.lambdas = tuple(float(v) for v in lambdas)
        self.hal_quantile = hal_quantile
        self.legacy = legacy
        self.msssim_form = 'mean' if legacy else msssim_form

        self.dwt = MultiLevelDWT(levels=dwt_levels)

        backbone = backbone.lower()
        if backbone == 'vgg19':
            self.feat_extractor = VGGFeatureExtractor()
            self.feat_weights = VGGFeatureExtractor.LAYER_WEIGHTS
        elif backbone == 'alexnet':
            self.feat_extractor = AlexNetFeatureExtractor()
            self.feat_weights = AlexNetFeatureExtractor.LAYER_WEIGHTS
        else:
            raise ValueError(f"backbone must be 'alexnet' or 'vgg19', got {backbone!r}")
        self.feat_extractor.eval()

        for param in self.parameters():
            param.requires_grad = False

    # ── Branch A: structural ─────────────────────────────────────────────────

    def structural_loss(self, sr_ll, hr_ll):
        """1 - MS-SSIM on the coarsest LL bands [B, C, H/2^J, W/2^J]; scalar in [0, 1]."""
        if self.legacy:
            # Submitted version: clamping saturates most coefficients, because the
            # orthonormal Haar LL band equals 2^J x the local block mean.
            sr_ll = torch.clamp(sr_ll, 0.0, 1.0)
            hr_ll = torch.clamp(hr_ll, 0.0, 1.0)
        else:
            # Divide by 2^J so the band holds the block means, exactly in [0, 1]
            scale = 2 ** self.dwt.levels
            sr_ll = sr_ll / scale
            hr_ll = hr_ll / scale
        return ms_ssim_loss(sr_ll, hr_ll, levels=3, form=self.msssim_form)

    # ── Branch B: perceptual ─────────────────────────────────────────────────

    def perceptual_loss(self, sr, hr):
        """Layer-weighted spatial L1 distance between features of the full images; scalar >= 0."""
        feats_sr = self.feat_extractor(sr)
        feats_hr = self.feat_extractor(hr)
        loss = 0.0
        for w, f_sr, f_hr in zip(self.feat_weights, feats_sr, feats_hr):
            loss = loss + w * F.l1_loss(f_sr, f_hr)
        return loss

    # ── Branch C: hallucination penalty ──────────────────────────────────────

    def hallucination_penalty(self, sr_hf_list, hr_hf_list):
        """
        Mean over DWT levels of the excess SR high-frequency energy in smooth HR regions:
          1. energy E = sqrt(LH^2 + HL^2 + HH^2), averaged over colour channels;
          2. smooth mask M = E_hr <= q-quantile of E_hr (per image);
          3. penalty = mean(ReLU(E_sr - E_hr) * M).
        sr_hf_list, hr_hf_list: lists of dicts {'LH', 'HL', 'HH'} per level. Scalar >= 0.
        """
        total_penalty = 0.0
        n_levels = len(hr_hf_list)

        for sr_bands, hr_bands in zip(sr_hf_list, hr_hf_list):
            hr_energy = torch.sqrt(
                hr_bands['LH'] ** 2 + hr_bands['HL'] ** 2 + hr_bands['HH'] ** 2 + 1e-8
            )
            sr_energy = torch.sqrt(
                sr_bands['LH'] ** 2 + sr_bands['HL'] ** 2 + sr_bands['HH'] ** 2 + 1e-8
            )
            hr_mag = hr_energy.mean(dim=1, keepdim=True)
            sr_mag = sr_energy.mean(dim=1, keepdim=True)

            B = hr_mag.shape[0]
            threshold = torch.quantile(
                hr_mag.view(B, -1), self.hal_quantile, dim=1
            ).view(B, 1, 1, 1)

            # '<=' keeps perfectly flat regions (many pixels tied at the threshold,
            # e.g. a constant reference) inside the mask; the submitted '<' left it empty.
            if self.legacy:
                smooth_mask = (hr_mag < threshold).float()
            else:
                smooth_mask = (hr_mag <= threshold).float()

            excess = F.relu(sr_mag - hr_mag) * smooth_mask
            total_penalty = total_penalty + excess.mean()

        return total_penalty / n_levels

    # ── Score ────────────────────────────────────────────────────────────────

    def combine(self, l_struct, l_percept, l_hal):
        """Weighted sum of the tanh-normalised branch values (tensors)."""
        return (self.lambdas[0] * torch.tanh(l_struct / NORM_SCALES[0]) +
                self.lambdas[1] * torch.tanh(l_percept / NORM_SCALES[1]) +
                self.lambdas[2] * torch.tanh(l_hal / NORM_SCALES[2]))

    def components(self, sr, hr):
        """Raw branch values (l_struct, l_percept, l_hal) for sr, hr [B, 3, H, W] in [0, 1]."""
        sr_ll, sr_hf = self.dwt(sr)
        hr_ll, hr_hf = self.dwt(hr)
        return (self.structural_loss(sr_ll, hr_ll),
                self.perceptual_loss(sr, hr),
                self.hallucination_penalty(sr_hf, hr_hf))

    def forward(self, sr, hr):
        """
        Args:
            sr : [B, 3, H, W] super-resolved image in [0, 1]
            hr : [B, 3, H, W] reference image in [0, 1]
        Returns:
            score     : WaSP score (scalar, lower = better; batch-averaged components)
            l_struct  : raw Branch A value (1 - MS-SSIM on LL_J)
            l_percept : raw Branch B value (AlexNet feature distance)
            l_hal     : raw Branch C value (hallucination penalty)
        """
        l_struct, l_percept, l_hal = self.components(sr, hr)
        return self.combine(l_struct, l_percept, l_hal), l_struct, l_percept, l_hal
