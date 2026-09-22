# WaSP — Wavelet-aware Super-resolution Perceptual Metric

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **WaSP**, a composite full-reference image quality metric for single-image super-resolution (SISR) that combines structural fidelity, perceptual feature similarity, and an explicit hallucination penalty via multi-level Haar wavelet decomposition.

> **Paper:** *WaSP: Wavelet-based Semantically-guided Perceptual Metric for Super-Resolution* (under review)

---

## Overview

Existing SR metrics — PSNR, SSIM, and LPIPS — lack an explicit mechanism to detect **hallucinated textures**: plausible-but-fictitious high-frequency detail injected by GAN- and diffusion-based SR models. WaSP addresses this with three complementary branches:

| Branch | Component | Description |
|---|---|---|
| **A** | $\mathcal{L}_\text{struct}$ | MS-SSIM on the 3-level Haar LL approximation subband |
| **B** | $\mathcal{L}_\text{percept}$ | Spatially-aware $\ell_1$ distance across 5 AlexNet feature layers |
| **C** | $\mathcal{L}_\text{hal}$ | Adaptive hallucination penalty on HF wavelet detail subbands |

Each component is normalised via a stateless $\tanh$ function before weighted aggregation:

$$\mathcal{L}_\text{WaSP} = \lambda_s\,\hat{\mathcal{L}}_\text{struct} + \lambda_p\,\hat{\mathcal{L}}_\text{percept} + \lambda_h\,\hat{\mathcal{L}}_\text{hal}$$

WaSP is **fully differentiable** and can be used as both an evaluation metric and a perceptual training loss.

---

## Repository Structure

```
wasp/
├── wasp/
│   ├── __init__.py           # Package entry — exposes WaSPMetric
│   └── wasp_metric.py        # Core WaSP implementation
├── utils/
│   ├── metrics.py            # PSNR / SSIM utilities
│   └── dataset.py            # SR dataset helpers
├── training/
│   └── finetune.py           # Fine-tune any SR model with WaSP loss
├── evaluation/
│   └── evaluate.py           # Evaluate on Set5/Set14/BSD100/Urban100
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/wasp.git
cd wasp
pip install -r requirements.txt
# Optional: install as a package
pip install -e .
```

**Dependencies:** `torch>=1.13`, `torchvision>=0.14`, `Pillow>=9.0`, `numpy>=1.21`

---

## Quick Start

### Use WaSP as an evaluation metric

```python
import torch
from wasp import WaSPMetric

metric = WaSPMetric(backbone='alexnet').cuda().eval()

# SR and HR images: float tensors in [0, 1], shape [B, 3, H, W]
sr = torch.rand(1, 3, 256, 256).cuda()
hr = torch.rand(1, 3, 256, 256).cuda()

with torch.no_grad():
    score, l_struct, l_percept, l_hal = metric(sr, hr)

print(f'WaSP: {score.item():.4f}  '
      f'(struct={l_struct.item():.4f}, '
      f'percept={l_percept.item():.4f}, '
      f'hal={l_hal.item():.4f})')
```

### Use WaSP as a training loss

```python
from wasp import WaSPMetric
import torch.nn as nn

metric = WaSPMetric(backbone='alexnet').cuda().eval()
for p in metric.parameters():
    p.requires_grad_(False)   # Keep the metric frozen during SR training

l1_loss = nn.L1Loss()
alpha   = 0.01   # WaSP loss scale

# Inside your training loop:
sr = model(lr)
l_pixel = l1_loss(sr, hr)
score, *_ = metric(sr, hr)
loss = l_pixel + alpha * score
loss.backward()
```

---

## Fine-Tuning SR Models

Fine-tune a pretrained SR model using the WaSP loss over two phases:
- **Phase 1 (warm-up):** L1 only — stabilises adaptation to the new dataset.
- **Phase 2 (fine-tune):** L1 + WaSP — perceptual refinement with hallucination penalty.

```bash
python training/finetune.py \
    --model rcan \
    --scale 4 \
    --hr_dir /path/to/DIV2K/DIV2K_train_HR \
    --warmup_epochs 20 \
    --finetune_epochs 200 \
    --lr 5e-5 \
    --lambda_struct 0.1 \
    --lambda_percept 1.0 \
    --lambda_hal 0.05 \
    --wasp_scale 0.01
```

**Supported models:** `rcan` | `edsr` | `realesrnet` | `realesrgan` | `swinir`

Run `python training/finetune.py --help` for all options.

---

## Evaluation

Evaluate any SR model on standard benchmarks (Set5, Set14, BSD100, Urban100):

```bash
python evaluation/evaluate.py \
    --model rcan \
    --model_path checkpoints/rcan_x4_wasp/rcan_wasp_best.pth \
    --scale 4 \
    --data_dir /path/to/benchmarks \
    --out evaluation/results.json
```

**Expected data layout:**
```
benchmarks/
  Set5/HR/         *.png
  Set5/LR_x4/      *.png   (optional; generated via bicubic if absent)
  Set14/HR/
  ...
```

---

## Key Results

### ×4 SR — WaSP fine-tuning gains

| Model | ΔPSNR (dB) | ΔWASP (%) |
|---|---|---|
| SRResNet+WaSP | +0.14 | −1.9 |
| RealESRNet+WaSP | +2.54 | −51.9 |
| **RealESRGAN+WaSP** | **+4.80** | **−61.2** |
| SwinIR+WaSP | −0.04 | −0.4 |

### Human MOS correlation (SRCC) — PIPAL benchmark

| Metric | SRCC ↑ |
|---|---|
| PSNR | 0.413 |
| SSIM | 0.521 |
| LPIPS | 0.591 |
| **WaSP-Percept** | **0.603** |

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `backbone` | `alexnet` | Perceptual backbone (`alexnet`) |
| `lambda_s` | 1.0 | Structural branch weight $\lambda_s$ |
| `lambda_p` | 0.5 | Perceptual branch weight $\lambda_p$ |
| `lambda_h` | 0.1 | Hallucination branch weight $\lambda_h$ |
| `J` | 3 | Number of wavelet decomposition levels |
| `q` | 0.25 | Smooth-region quantile threshold |
| `alpha` | 0.01 | WaSP loss scale in combined training objective |

## Model weights will be released soon

