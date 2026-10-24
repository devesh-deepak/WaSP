"""Sanity tests of the metric (CPU, no datasets; torchvision downloads AlexNet once).

  pytest -q tests
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from wasp import MultiLevelDWT, WaSPMetric, ms_ssim_loss, score_from_components


@pytest.fixture(scope='module')
def metric():
    return WaSPMetric().eval()


def smooth_image(seed=0, size=128):
    """Smooth random image: low-resolution noise upsampled bicubically."""
    g = torch.Generator().manual_seed(seed)
    return F.interpolate(torch.rand(1, 3, size // 16, size // 16, generator=g), size=size,
                         mode='bicubic', align_corners=False).clamp(0, 1)


def test_identical_images_score_zero(metric):
    x = smooth_image()
    score, l_s, l_p, l_h = metric(x, x)
    assert abs(l_s.item()) < 1e-6 and l_p.item() < 1e-6 and l_h.item() == 0
    assert score.item() < 1e-5


def test_score_range_and_components(metric):
    hr, sr = smooth_image(0), torch.rand(1, 3, 128, 128)
    score, l_s, l_p, l_h = metric(sr, hr)
    assert 0 < score.item() < sum(metric.lambdas)
    assert min(l_s.item(), l_p.item(), l_h.item()) >= 0
    np.testing.assert_allclose(score.item(), score_from_components(l_s.item(), l_p.item(), l_h.item()),
                               rtol=1e-5)


def test_added_texture_is_penalised_blur_is_not(metric):
    hr = smooth_image(1)
    noisy = (hr + 0.05 * torch.randn_like(hr)).clamp(0, 1)
    blurred = F.avg_pool2d(F.pad(hr, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
    assert metric(noisy, hr)[3].item() > 10 * metric(blurred, hr)[3].item()


def test_lambdas_weight_the_branches():
    hr, sr = smooth_image(2), torch.rand(1, 3, 64, 64)
    only_hal = WaSPMetric(lambdas=(0.0, 0.0, 1.0)).eval()
    score, _, _, l_h = only_hal(sr, F.interpolate(hr, size=64))
    np.testing.assert_allclose(score.item(), np.tanh(l_h.item() / 0.01), rtol=1e-5)


def test_variants_change_only_their_branches(metric):
    hr, sr = smooth_image(3), smooth_image(4)
    final = metric(sr, hr)
    mean = WaSPMetric(msssim_form='mean').eval()(sr, hr)
    legacy = WaSPMetric(legacy=True).eval()(sr, hr)
    assert final[2].item() == mean[2].item() == legacy[2].item()       # Branch B unchanged
    assert final[3].item() == mean[3].item()                            # same mask
    assert final[1].item() != legacy[1].item()                          # Branch A differs


def test_differentiable(metric):
    hr = smooth_image(5)
    sr = torch.rand(1, 3, 128, 128, requires_grad=True)
    metric(sr, hr)[0].backward()
    assert sr.grad is not None and torch.isfinite(sr.grad).all() and sr.grad.abs().sum() > 0


def test_haar_dwt_is_orthonormal():
    x = torch.rand(2, 3, 64, 48)
    ll, hf = MultiLevelDWT(levels=3)(x)
    energy = ll.pow(2).sum() + sum(b[k].pow(2).sum() for b in hf for k in ('LH', 'HL', 'HH'))
    np.testing.assert_allclose(energy.item(), x.pow(2).sum().item(), rtol=1e-5)
    assert ll.shape[-2:] == (8, 6)


def test_ms_ssim_forms_agree_on_one_scale():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 16, 16, generator=g)                        # only one scale >= 11 px
    y = (x + 0.1 * torch.randn(1, 3, 16, 16, generator=g)).clamp(0, 1)
    np.testing.assert_allclose(ms_ssim_loss(x, y, form='mean').item(),
                               ms_ssim_loss(x, y, form='product').item(), rtol=1e-6)
