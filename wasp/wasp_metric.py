# wasp_metric.py  — WaSP v2: Wavelet-aware Super-resolution Perceptual Metric
#
# Architecture overview:
#   Input SR / HR images  [B, 3, H, W]  in [0, 1]
#         │
#         ▼
#   MultiLevelDWT  (3-level Haar, fixed)
#   ┌──────────────────────────────────────────┐
#   │  Level 1: LH1, HL1, HH1  (fine detail)  │
#   │  Level 2: LH2, HL2, HH2  (mid-scale)    │
#   │  Level 3: LH3, HL3, HH3  (coarse edges) │
#   │  LL3: coarsest approximation             │
#   └──────────────────────────────────────────┘
#         │
#   ┌─────┴─────────────────────────────┐
#   │                                   │
#   ▼                                   ▼
# Branch A: Structural (LL3)     Branch B: Perceptual (full image)
# MS-SSIM loss                   Spatial VGG feature L1 (relu1_2,
# (no network, no resize)        relu2_2, relu3_3, relu4_3)
#   │                                   │
#   └─────────────┬─────────────────────┘
#                 │
#   Branch C: Hallucination Penalty (all HF bands)
#   Adaptive per-image threshold (25th percentile of HR energy)
#   Penalises excess SR energy in smooth HR regions
#                 │
#                 ▼
#   score = λ_s·L_struct + λ_p·L_percept + λ_h·L_hal
#   (each component self-normalised before weighting)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

try:
    from config import Config
except ImportError:
    # Fallback defaults if run standalone
    class Config:
        LAMBDA_STRUCT  = 1.0
        LAMBDA_PERCEPT = 0.5
        LAMBDA_HAL     = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Multi-Level Discrete Wavelet Transform (Haar, fixed basis)
# ─────────────────────────────────────────────────────────────────────────────

class MultiLevelDWT(nn.Module):
    """
    Applies the 2D Haar DWT recursively for `levels` levels.

    Returns:
        ll   : [B, C, H/2^L, W/2^L]  — coarsest low-frequency approximation
        hf   : list of length `levels`, each entry is a dict:
               {'LH': ..., 'HL': ..., 'HH': ...}  at that scale
               hf[0] = finest (level 1), hf[-1] = coarsest (level L)

    Haar filter convention (standard):
        LL  =  x1 + x2 + x3 + x4          (low-pass both dims)
        LH  = -x1 + x2 - x3 + x4          (low-pass rows, high-pass cols → vertical edges)
        HL  = -x1 - x2 + x3 + x4          (high-pass rows, low-pass cols → horizontal edges)
        HH  =  x1 - x2 - x3 + x4          (high-pass both dims → diagonal)
    where x1..x4 are the four 2×2 sub-pixels (after /2 normalisation).
    """

    def __init__(self, levels: int = 3):
        super().__init__()
        self.levels = levels
        self.requires_grad = False

    def _haar_step(self, x):
        """Single-level 2D Haar DWT on tensor x: [B, C, H, W] → 4 sub-bands.
        Handles odd H/W by reflect-padding to the next even dimension.
        """
        # Pad to even dimensions if needed (reflect padding preserves statistics)
        _, _, H, W = x.shape
        pad_h = H % 2  # 1 if odd height, 0 if even
        pad_w = W % 2  # 1 if odd width,  0 if even
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Split into 2×2 blocks (no overlap, stride 2)
        x01 = x[:, :, 0::2, :] / 2   # even rows
        x02 = x[:, :, 1::2, :] / 2   # odd rows
        x1  = x01[:, :, :, 0::2]     # even cols of even rows
        x2  = x02[:, :, :, 0::2]     # even cols of odd rows
        x3  = x01[:, :, :, 1::2]     # odd cols of even rows
        x4  = x02[:, :, :, 1::2]     # odd cols of odd rows

        LL = x1 + x2 + x3 + x4       # approximation
        LH = -x1 + x2 - x3 + x4      # vertical edges   (low-row, high-col)
        HL = -x1 - x2 + x3 + x4      # horizontal edges (high-row, low-col)
        HH =  x1 - x2 - x3 + x4      # diagonal details
        return LL, LH, HL, HH

    def forward(self, x):
        ll = x
        hf_bands = []
        for _ in range(self.levels):
            ll, lh, hl, hh = self._haar_step(ll)
            hf_bands.append({'LH': lh, 'HL': hl, 'HH': hh})
        return ll, hf_bands  # ll is LL after all levels; hf_bands[0]=finest


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MS-SSIM (Multi-Scale Structural Similarity) — for Branch A
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float, channels: int, device):
    """Creates a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)          # [ws, ws]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, ws, ws]
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size)
    return kernel_2d.contiguous()


def _ssim_single_scale(x, y, window_size=11, sigma=1.5, data_range=1.0, eps=1e-8):
    """
    Computes SSIM between x and y at a single scale.
    x, y: [B, C, H, W]
    Returns: mean SSIM scalar.
    """
    C = x.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, C, x.device)
    pad = window_size // 2

    mu_x  = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y  = F.conv2d(y, kernel, padding=pad, groups=C)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sig_x2  = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x2
    sig_y2  = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y2
    sig_xy  = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2) + eps)
    return ssim_map.mean()


def ms_ssim_loss(x, y, levels=3, weights=None, window_size=11, data_range=1.0):
    """
    Multi-Scale SSIM loss (1 - MS-SSIM).
    Operates at `levels` spatial scales by progressively downsampling.
    x, y: [B, C, H, W] in [0, data_range]
    Returns: scalar loss in [0, 1] (0 = identical, 1 = maximally different).
    """
    if weights is None:
        # Weights from Wang et al. (2003) MS-SSIM paper
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = weights[:levels]
        total = sum(weights)
        weights = [w / total for w in weights]

    ssim_vals = []
    x_curr, y_curr = x, y
    for i in range(levels):
        # Ensure spatial dims are large enough for the window
        if x_curr.shape[-1] < window_size or x_curr.shape[-2] < window_size:
            break
        s = _ssim_single_scale(x_curr, y_curr, window_size=window_size,
                                data_range=data_range)
        ssim_vals.append(s)
        if i < levels - 1:
            # Downsample by 2 with anti-aliasing (avg pool)
            x_curr = F.avg_pool2d(x_curr, kernel_size=2, stride=2)
            y_curr = F.avg_pool2d(y_curr, kernel_size=2, stride=2)

    if not ssim_vals:
        return torch.tensor(0.0, device=x.device)

    # Weighted average of SSIM values across scales
    w = weights[:len(ssim_vals)]
    w_sum = sum(w)
    ms_ssim_val = sum(wi * s for wi, s in zip(w, ssim_vals)) / w_sum
    return 1.0 - ms_ssim_val   # loss: lower = better


# ─────────────────────────────────────────────────────────────────────────────
# 3a.  VGG-19 Perceptual Feature Extractor (legacy)
# ─────────────────────────────────────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate features from VGG19 at 4 layers:
        relu1_2 (idx  3), relu2_2 (idx  8), relu3_3 (idx 17), relu4_4 (idx 26)
    Kept for backward compatibility; AlexNetFeatureExtractor is preferred.
    """
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
# 3b.  AlexNet Perceptual Feature Extractor (DEFAULT — replaces VGG19)
# ─────────────────────────────────────────────────────────────────────────────

class AlexNetFeatureExtractor(nn.Module):
    """
    Extracts intermediate features from AlexNet at 5 ReLU tap points:
        Layer 1:  relu1  (64  ch, stride-4 conv)   → coarse spatial, fine receptive
        Layer 4:  relu2  (192 ch, after maxpool)    → mid-level texture
        Layer 7:  relu3  (384 ch, full-size conv)   → rich texture patterns
        Layer 9:  relu4  (256 ch)                   → semantic shapes
        Layer 11: relu5  (256 ch)                   → high-level semantics

    **Why AlexNet over VGG19 for WaSP:**
      - LPIPS (Zhang et al., CVPR 2018) showed AlexNet features have *higher*
        Spearman rank correlation with human MOS on SR/distortions than VGG19.
      - Features are calibrated specifically for perceptual distance in LPIPS;
        using the same backbone ensures our metric aligns with human perception.
      - Much faster inference: AlexNet (60M params) vs VGG19 (143M params).
      - More diverse receptive fields due to the 5 distinct stride/size layers.

    All weights are frozen. Input: [B, 3, H, W] in [0, 1].
    """

    # Verified layer indices from torchvision AlexNet.features
    # Layer output channels: [64, 192, 384, 256, 256]
    LAYER_INDICES = [1, 4, 7, 9, 11]
    LAYER_WEIGHTS = [0.1, 0.15, 0.25, 0.25, 0.25]  # progressive depth emphasis

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
        # LPIPS normalisation (same as ImageNet mean/std but AlexNet-specific shift)
        self.register_buffer('shift', torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer('scale', torch.tensor([ 0.458,  0.448,  0.450]).view(1, 3, 1, 1))

    def normalize(self, x):
        """[0,1] → LPIPS-style AlexNet normalisation."""
        x = x * 2.0 - 1.0          # [0,1] → [-1,1]
        return (x - self.shift) / self.scale

    def forward(self, x):
        """
        Returns list of feature maps at 5 tap points.
        x: [B, 3, H, W] in [0, 1]
        """
        x = self.normalize(x)
        features = []
        for s in self.slices:
            x = s(x)
            # L2-normalise in channel dim (standard LPIPS feature normalisation)
            features.append(x / (x.norm(dim=1, keepdim=True) + 1e-10))
        return features


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WaSP v2 Metric Network
# ─────────────────────────────────────────────────────────────────────────────

class WaSPMetric(nn.Module):
    """
    WaSP — Wavelet-aware Super-resolution Perceptual Metric.

    Three branches:
        A. Structural    (L_struct)  : MS-SSIM on multi-level DWT LL band
        B. Perceptual    (L_percept) : Spatial feature L1 (AlexNet by default)
        C. Hallucination (L_hal)     : Adaptive HF energy penalty on DWT bands

    All weights frozen. No gradients through the metric networks.

    Args:
        dwt_levels : Number of DWT decomposition levels (default 3)
        backbone   : Perceptual feature backbone for Branch B.
                     'alexnet' (default) — AlexNet, calibrated to human MOS
                                           (same backbone as LPIPS, Zhang et al. CVPR 2018)
                     'vgg19'             — VGG19 (legacy, included for ablation)

    Usage:
        metric = WaSPMetric().to(device)                # uses AlexNet
        score, l_struct, l_percept, l_hal = metric(sr, hr)
    """

    def __init__(self, dwt_levels: int = 3, backbone: str = 'alexnet'):
        super().__init__()

        # --- Wavelet decomposition ---
        self.dwt = MultiLevelDWT(levels=dwt_levels)

        # --- Branch B: Perceptual feature extractor ---
        backbone = backbone.lower()
        if backbone == 'vgg19':
            self.feat_extractor = VGGFeatureExtractor()
            self._feat_weights  = VGGFeatureExtractor.LAYER_WEIGHTS
            print('[WaSP] Branch B backbone: VGG19')
        else:  # default: alexnet
            self.feat_extractor = AlexNetFeatureExtractor()
            self._feat_weights  = AlexNetFeatureExtractor.LAYER_WEIGHTS
            print('[WaSP] Branch B backbone: AlexNet (LPIPS-aligned)')
        self.feat_extractor.eval()

        # Freeze everything
        for param in self.parameters():
            param.requires_grad = False

        # Small epsilon for self-normalisation
        self._eps = 1e-6

    # ── Branch A: Structural ──────────────────────────────────────────────────

    def _structural_loss(self, sr_ll, hr_ll):
        """
        MS-SSIM loss on the coarsest LL approximation band.
        sr_ll, hr_ll: [B, C, H/2^L, W/2^L]
        Returns scalar in [0, 1].
        """
        # Clamp to valid range (wavelet LL can slightly exceed [0,1])
        sr_ll = torch.clamp(sr_ll, 0.0, 1.0)
        hr_ll = torch.clamp(hr_ll, 0.0, 1.0)
        return ms_ssim_loss(sr_ll, hr_ll, levels=3)

    # ── Branch B: Perceptual ──────────────────────────────────────────────────

    def _perceptual_loss(self, sr, hr):
        """
        Weighted spatial L1 distance in feature space (AlexNet or VGG19).
        Operates on the original full-resolution SR and HR images.
        Returns scalar >= 0.
        """
        feats_sr = self.feat_extractor(sr)
        feats_hr = self.feat_extractor(hr)

        loss = 0.0
        for w, f_sr, f_hr in zip(self._feat_weights, feats_sr, feats_hr):
            loss = loss + w * F.l1_loss(f_sr, f_hr)
        return loss

    # ── Branch C: Hallucination Penalty ──────────────────────────────────────

    def _hallucination_penalty(self, sr_hf_list, hr_hf_list):
        """
        Adaptive hallucination penalty across all DWT levels.

        For each level:
          1. Compute per-pixel HF energy magnitude for SR and HR.
          2. Derive a per-image adaptive smoothness threshold =
             25th percentile of HR energy at that level.
          3. Create a smooth-region mask where HR energy < threshold.
          4. Penalise excess SR energy (ReLU(sr_energy - hr_energy))
             inside the smooth mask.

        sr_hf_list, hr_hf_list: lists of dicts {'LH', 'HL', 'HH'} per level.
        Returns scalar >= 0.
        """
        total_penalty = 0.0
        n_levels = len(hr_hf_list)

        for sr_bands, hr_bands in zip(sr_hf_list, hr_hf_list):
            # Energy magnitude: sqrt(LH² + HL² + HH²)  [B, C, H, W]
            hr_energy = torch.sqrt(
                hr_bands['LH'] ** 2 + hr_bands['HL'] ** 2 + hr_bands['HH'] ** 2 + 1e-8
            )
            sr_energy = torch.sqrt(
                sr_bands['LH'] ** 2 + sr_bands['HL'] ** 2 + sr_bands['HH'] ** 2 + 1e-8
            )

            # Collapse channels → spatial energy map  [B, 1, H, W]
            hr_mag = hr_energy.mean(dim=1, keepdim=True)
            sr_mag = sr_energy.mean(dim=1, keepdim=True)

            # Adaptive threshold: 25th percentile of HR energy per image
            B = hr_mag.shape[0]
            threshold = torch.quantile(
                hr_mag.view(B, -1), 0.25, dim=1
            ).view(B, 1, 1, 1)

            # Smooth-region mask (1 where HR is smooth, 0 elsewhere)
            smooth_mask = (hr_mag < threshold).float()

            # Excess SR energy in smooth regions (only penalise SR > HR)
            excess = F.relu(sr_mag - hr_mag) * smooth_mask
            total_penalty = total_penalty + excess.mean()

        return total_penalty / n_levels   # average across levels

    # ── Soft normalisation helper ─────────────────────────────────────────────

    def _soft_norm(self, loss, scale):
        """
        Soft normalisation via tanh: maps [0, ∞) → [0, 1).
        - loss = 0   → 0.0  (perfect reconstruction correctly gives 0)
        - loss = scale → tanh(1) ≈ 0.76  (typical distortion level)
        - loss >> scale → approaches 1.0  (saturates for large distortions)

        This avoids the self-normalisation pitfall where floating-point
        residuals (~1e-7) get amplified to ~0.23.

        Scale constants are set to typical loss magnitudes at moderate distortion:
            l_struct  scale = 0.3   (MS-SSIM loss range: 0 to ~0.5)
            l_percept scale = 0.05  (VGG L1 feature loss range: 0 to ~0.2)
            l_hal     scale = 0.01  (HF energy excess range: 0 to ~0.05)
        """
        return torch.tanh(loss / scale)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, sr, hr):
        """
        Compute WaSP score between super-resolved image and HR reference.

        Args:
            sr : [B, 3, H, W]  Super-resolved image, values in [0, 1]
            hr : [B, 3, H, W]  High-resolution reference, values in [0, 1]

        Returns:
            score    : Weighted total WaSP score (scalar, lower = better)
            l_struct : Structural component (MS-SSIM on LL band)
            l_percept: Perceptual component (VGG spatial feature L1)
            l_hal    : Hallucination penalty (adaptive HF excess energy)
        """
        # 1. Multi-level wavelet decomposition
        sr_ll, sr_hf = self.dwt(sr)
        hr_ll, hr_hf = self.dwt(hr)

        # 2. Branch A — Structural loss on LL band
        l_struct = self._structural_loss(sr_ll, hr_ll)

        # 3. Branch B — Perceptual loss on full image
        l_percept = self._perceptual_loss(sr, hr)

        # 4. Branch C — Hallucination penalty on all HF bands
        l_hal = self._hallucination_penalty(sr_hf, hr_hf)

        # 5. Soft-normalise each component to [0, 1) using tanh scaling
        l_struct_n  = self._soft_norm(l_struct,  scale=0.30)
        l_percept_n = self._soft_norm(l_percept, scale=0.05)
        l_hal_n     = self._soft_norm(l_hal,     scale=0.01)

        # 6. Weighted aggregate
        score = (Config.LAMBDA_STRUCT  * l_struct_n  +
                 Config.LAMBDA_PERCEPT * l_percept_n +
                 Config.LAMBDA_HAL     * l_hal_n)

        # Return raw (un-normalised) components for logging/analysis
        return score, l_struct, l_percept, l_hal