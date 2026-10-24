"""Phase-2 fine-tuning objectives: L1 + alpha * auxiliary term.

Every criterion returns (total, l1, aux1, aux2, aux3) for logging. The loss controls use the
same recipe as WaSP (L1 stays the primary term, the auxiliary term enters at the same global
scale alpha), so differences between runs come from the auxiliary term alone.
"""

import torch
import torch.nn as nn

LOSSES = ('wasp', 'l1', 'lpips', 'dists', 'msssim')


class WaSPLoss(nn.Module):
    """L1 + alpha * (l_s' L_A + l_p' L_B + l_h' L_C), on the raw (un-normalised) WaSP branches."""

    def __init__(self, wasp, weights=(0.1, 1.0, 0.05), alpha=0.01, lambda_l1=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.wasp = wasp
        self.weights = weights
        self.alpha = alpha
        self.lambda_l1 = lambda_l1

    def forward(self, sr, hr):
        l_pix = self.l1(sr, hr)
        _, l_struct, l_percept, l_hal = self.wasp(sr, hr)
        wasp_total = (self.weights[0] * l_struct +
                      self.weights[1] * l_percept +
                      self.weights[2] * l_hal)
        loss = self.lambda_l1 * l_pix + self.alpha * wasp_total
        return loss, l_pix, l_struct, l_percept, l_hal


class L1OnlyLoss(nn.Module):
    """Pure-L1 control: Phase 2 continues Phase 1 unchanged, so it measures what the extra
    fine-tuning alone recovers (large for RealESRGAN / RealESRNet, which were trained for
    real-world degradations, not bicubic)."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        l = self.l1(sr, hr)
        z = torch.zeros((), device=sr.device)
        return l, l, z, z, z


class AuxPerceptualLoss(nn.Module):
    """L1 + alpha * fn(sr, hr), for the LPIPS / DISTS / MS-SSIM controls."""

    def __init__(self, fn, alpha=0.01, lambda_l1=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.fn = fn
        self.alpha = alpha
        self.lambda_l1 = lambda_l1

    def forward(self, sr, hr):
        l_pix = self.l1(sr, hr)
        l_perc = self.fn(sr, hr)
        loss = self.lambda_l1 * l_pix + self.alpha * l_perc
        z = torch.zeros((), device=sr.device)
        return loss, l_pix, l_perc, z, z


def _frozen(net):
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval()


def build_phase2_loss(name, wasp, device, weights=(0.1, 1.0, 0.05), alpha=0.01, lambda_l1=1.0):
    """The Phase-2 criterion `name` (one of LOSSES). `wasp` is a frozen WaSPMetric."""
    if name == 'wasp':
        return WaSPLoss(wasp, weights, alpha, lambda_l1)
    if name == 'l1':
        return L1OnlyLoss()
    if name == 'lpips':
        import lpips
        net = _frozen(lpips.LPIPS(net='alex', verbose=False).to(device))
        return AuxPerceptualLoss(lambda sr, hr: net(sr * 2 - 1, hr * 2 - 1).mean(), alpha, lambda_l1)
    if name == 'dists':
        # piq's DISTS: differentiable and ~100x faster than torchmetrics' here
        import piq
        net = _frozen(piq.DISTS().to(device))
        return AuxPerceptualLoss(lambda sr, hr: net(sr, hr), alpha, lambda_l1)
    if name == 'msssim':
        from wasp import ms_ssim_loss
        return AuxPerceptualLoss(lambda sr, hr: ms_ssim_loss(sr, hr, levels=3), alpha, lambda_l1)
    raise ValueError(f'unknown loss {name!r} (choose from {LOSSES})')
