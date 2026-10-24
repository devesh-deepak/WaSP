# WaSP: a wavelet-based perceptual metric for image super-resolution

Code for the paper *WaSP: A Wavelet-based Semantically-guided Perceptual Metric for Image
Super-Resolution* (D. Kumar, G. Sikka, C.-C. Lin, S. Singh; submitted to *Scientific Reports*).

WaSP is a full-reference quality metric for super-resolved (SR) images. It decomposes both images
with a 3-level Haar wavelet transform and combines three branches:

| Branch | Measures | Computed on |
|---|---|---|
| A, structural (L_A) | 1 − MS-SSIM | coarsest LL band, divided by 2^J |
| B, perceptual (L_B) | layer-weighted L1 distance of LPIPS-normalised AlexNet features | full images |
| C, hallucination (L_C) | SR high-frequency energy in excess of HR energy, inside the smooth regions of HR (per-image 25th-percentile threshold) | LH/HL/HH bands of every level |

`WaSP = λ_s·tanh(L_A/0.30) + λ_p·tanh(L_B/0.05) + λ_h·tanh(L_C/0.01)` with
(λ_s, λ_p, λ_h) = (1, 0.5, 0.1). Lower is better. The metric is differentiable, so it can also be
used as a training loss.

This repository contains the metric, the SR models and fine-tuning code, every evaluation and
analysis script behind the paper's tables and figures, and the per-image results the tables were
built from.

## Installation

```bash
git clone https://github.com/devesh-deepak/WaSP.git
cd WaSP
pip install -e .                  # the metric only (torch, torchvision, numpy)
pip install -r requirements.txt   # everything needed for the experiments
```

Tested with Python 3.9.21, PyTorch 2.4.0, torchvision 0.19 and CUDA GPUs (H100). The tested
version of each package is listed in `requirements.txt`. The AlexNet weights of Branch B are
downloaded by torchvision on first use.

## Using the metric

```python
import torch
from wasp import WaSPMetric

metric = WaSPMetric().cuda().eval()        # default weights (1.0, 0.5, 0.1)
sr = torch.rand(1, 3, 256, 256).cuda()     # [B, 3, H, W] in [0, 1]
hr = torch.rand(1, 3, 256, 256).cuda()
with torch.no_grad():
    score, l_struct, l_percept, l_hal = metric(sr, hr)
```

`score` uses the batch-averaged branch values. To score images separately, call the metric once
per image. The raw branch values can be re-weighted later with
`wasp.score_from_components(l_struct, l_percept, l_hal, lambdas)`.

Scoring a folder of SR images against references with the same file names (writes a CSV with WaSP,
its branches, and Y-channel PSNR / SSIM):

```bash
python evaluation/score_images.py --sr path/to/sr_images --hr path/to/hr_images
```

As a training loss (the fine-tuning recipe of the paper, on the raw branch values):

```python
from training.losses import WaSPLoss
criterion = WaSPLoss(WaSPMetric().cuda().eval(), weights=(0.1, 1.0, 0.05), alpha=0.01)
loss, l1, l_struct, l_percept, l_hal = criterion(sr, hr)  # L1 + 0.01·(0.1·L_A + 1.0·L_B + 0.05·L_C)
```

### Metric versions

| Version | Constructor | Branch A | Branch C mask | Used for |
|---|---|---|---|---|
| final (default) | `WaSPMetric()` | LL / 2^J, standard (product) MS-SSIM | `<=` quantile | main results of the paper |
| mean form | `WaSPMetric(msssim_form='mean')` | LL / 2^J, weighted arithmetic mean of SSIM | `<=` | x2 training runs; Supplementary S7 |
| submitted | `WaSPMetric(legacy=True)` | LL clamped to [0, 1], arithmetic mean | `<` | x4 training runs; Supplementary S5.2 |

Other options: `backbone='vgg19'` (Branch B ablation), `dwt_levels`, `hal_quantile`, `lambdas`.

## Repository layout

```
wasp/                 the metric (pip-installable package)
models/               SR models: RCAN, EDSR, RealESRNet / RealESRGAN (RRDBNet), SwinIR (Swin2SR),
                      SinSR, IMDN, SRResNet; build_model() / load_checkpoint() registry
  third_party/sinsr/  SinSR / ResShift code (CC BY-NC-SA 4.0, see below)
training/             two-phase fine-tuning (finetune.py), losses, SRResNet-L1 baseline
scripts/              the exact training recipes of the paper; reproduce_tables.sh
evaluation/           benchmark evaluation, MOS correlation and all analyses
figures/              the paper's figures
results/              per-image caches and result files the tables and figures were built from
checkpoints/          SHA256SUMS of the released fine-tuned checkpoints
tests/                metric tests (pytest -q tests)
```

## Data

Paths default to folders inside the repository and can be changed with environment variables
(or with `--data`-style flags of the scripts):

| Variable | Default | Contents |
|---|---|---|
| `WASP_SR_DATA` | `datasets/sr` | `Set5/ Set14/ BSD100/ Urban100/` and `DIV2K/DIV2K_train_HR/`, `DIV2K/DIV2K_valid_HR/` |
| `WASP_IQA_DATA` | `datasets/iqa` | `TID2013/ KADID-10K/ PIPAL/` |
| `WASP_CKPT` | `checkpoints` | released fine-tuned checkpoints |
| `WASP_PRETRAINED` | `pretrained` | official pretrained weights (downloaded automatically) |
| `WASP_RESULTS` | `results` | caches, result JSON, tables (`results/tables/`), figures (`results/figures/`) |

* **SR benchmarks** use the official HR/LR pairs of the SelfExSR release (Huang et al., 2015):
  `<dataset>/image_SRF_<s>/*_HR.png` with the LR images either next to them (`*_LR.png`) or in
  `<dataset>/LR_SRF_<s>/`. HR images are cropped to a multiple of the scale.
* **DIV2K** (training): the 800 training HR images; fine-tuning holds out the last 20 for validation.
* **TID2013**: `distorted_images/`, `reference_images/`, `mos_with_names.txt`.
  **KADID-10K**: `images/`, `dmos.csv`. **PIPAL** (training split): `Train_Ref/`,
  `Distortion_1..4/`, `Train_Label/`. The paper uses a fixed 5,000-pair sample of PIPAL (seed 42).

## Models and checkpoints

Official pretrained weights are downloaded on first use: EDSR-baseline, RCAN (Google Drive, via
`gdown`), RealESRNet / RealESRGAN, IMDN, SinSR and the ResShift VQGAN. "SwinIR" in the paper is
the Swin2SR classical-SR model (`caidas/swin2sr-classical-sr-x{2,4}-64` on the HuggingFace Hub), a
SwinV2 version of SwinIR. SRResNet has no official checkpoint: its starting point is trained with
`training/train_srresnet_l1.py`.

The 80 fine-tuned checkpoints evaluated in the paper (7.5 GB) will be attached to a GitHub release.
Place them as `checkpoints/x<scale>/<model>_<variant>.pth` and check them with
`cd checkpoints && sha256sum -c SHA256SUMS`:

| Variant | Meaning |
|---|---|
| `wasp` | +WaSP (Phase 2: L1 + 0.01·WaSP) |
| `ctrl_l1`, `ctrl_lpips`, `ctrl_dists`, `ctrl_msssim` | loss controls (same recipe, other auxiliary loss) |
| `abla_la`, `abla_lb`, `abla_lc`, `abla_lalb` | branch ablations (x4) |
| `l1` | SRResNet-L1 starting point (`x4/srresnet_l1.pth`) |

```python
from models import build_model, load_checkpoint, checkpoint_path, super_resolve
model = load_checkpoint(build_model('rcan', 4), checkpoint_path('rcan', 4, 'wasp'))
sr = super_resolve(model, lr, size=hr.shape[-2:])
```

Each released checkpoint is the epoch with the lowest validation WaSP over the whole run (20 held-out
DIV2K images), except x4 SwinIR+WaSP, which is the final epoch. In 14 runs this epoch falls in the
L1-only warm-up, so the checkpoint was never trained with its auxiliary loss: x4 RCAN+WaSP,
SinSR+WaSP, the SinSR LPIPS / DISTS / MS-SSIM controls, the SinSR ablations, three RCAN ablations
(L_A, L_B, L_C), the SwinIR L_A+L_B ablation and the x2 EDSR MS-SSIM control. `training/finetune.py`
also saves `best_phase2.pth`, the best epoch after the auxiliary loss starts. The SinSR checkpoints
hold the UNet only; the frozen VQGAN keeps its official weights.

## Reproducing the paper

All tables and most figures can be regenerated from `results/` without a GPU:

```bash
bash scripts/reproduce_tables.sh
```

| Paper | Script(s) | Output in `results/` |
|---|---|---|
| SR benchmarks x4 / x2, WaSP gains, branch ablation, loss controls | `evaluation/eval_benchmarks.py` → `evaluation/make_tables.py` | `bench_x<s>/`, `benchmarks_x<s>.json`, `tables/` |
| MOS correlation and baselines | `evaluation/mos_components.py`, `evaluation/mos_baselines.py` → `evaluation/mos_analysis.py` | `mos_components_*.npz`, `mos_baselines.json`, `mos_analysis.json` |
| Significance (paired bootstrap), SRCC confidence intervals, weight sweeps | `evaluation/mos_analysis.py` | `mos_analysis.json` |
| Controlled hallucination test, hallucination vs MOS trade-off | `evaluation/hal_validation.py` | `hal_controlled.npz`, `hal_validation.json` |
| Hyperparameter sensitivity (q, levels, wavelet, backbone, weights) | `evaluation/hparam_sweep.py` | `hparam_*.npz`, `hparam_sensitivity.json` |
| Real SR outputs: PIPAL GAN vs PSNR-oriented, RRDB interpolation | `evaluation/real_sr_hallucination.py` | `real_sr_hallucination.json` |
| Branch A scales, patch vs full image | `evaluation/patch_scale_analysis.py mos / bench / report` | `patch_scale*.npz`, `patch_scale.json` |
| Runtime and training cost | `evaluation/runtime.py`, `evaluation/training_cost.py` | `runtime.json`, `training_cost.json` |
| Test / training set overlap (Suppl. S8) | `evaluation/data_overlap.py` | `data_overlap.json` |
| Figures 1–4, 7, hallucination, hyperparameters, weight trade-off, S1–S3 | `figures/fig_*.py` | `figures/` |

Recomputing from scratch: `eval_benchmarks.py --scale 4` (and `--scale 2`; `--list` shows the 71 / 25
configurations), `mos_components.py`, `mos_baselines.py`. Existing cache files are skipped unless
`--overwrite` is given or `--out_dir` points elsewhere.

**Training.** `scripts/train_x4.sh`, `scripts/train_ablation_x4.sh` and `scripts/train_x2.sh`
contain the exact settings of every released checkpoint: Adam, StepLR (×0.5 every 50 epochs),
128×128 patches, α = 0.01, (λ'_s, λ'_p, λ'_h) = (0.1, 1.0, 0.05), 20 + 200 epochs, learning rate
5e-5 and batch 8 (RealESRGAN 1e-5; SwinIR 2e-5 with batch 4); the ablations use 10 + 50 epochs with
5e-5 and batch 8 for every model. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python training/finetune.py --model rcan --scale 4 --wasp_variant legacy
python training/train_srresnet_l1.py         # SRResNet-L1 starting point (1000 epochs)
```

**Exactness.** Re-running this code on an H100 reproduced the paper's values bitwise for the metric
(all three versions), every pretrained model, 11 of 12 re-evaluated x4 benchmark configurations
(from the released checkpoints), the TID2013 component cache, and figures 1–3, the
hyperparameter, trade-off and S1 figures (pixel-identical). SwinIR+WaSP differed by up to
2·10⁻⁴ dB PSNR and the x2 models by about 10⁻⁷ (GPU kernel selection). `mos_baselines.py`
reproduced 71 of the 72 baseline correlations to 4 decimals (CLIP-IQA KRCC on TID2013: 0.4107 vs
0.4108 in the paper). The training runs were not seeded. Two of the seven distortions of
the controlled hallucination test add Gaussian noise drawn on the GPU, whose random stream depends
on the device; `results/hal_controlled.npz` is the paper's. SinSR samples noise in every forward
pass, so the scripts seed torch (42) before inference. `make_tables.py --msssim_form mean`
reproduces the Supplementary S7 snapshot except for one value on a rounding boundary.

## Results at a glance

Spearman correlation with human scores (TID2013, KADID-10K, PIPAL 5,000-pair sample), from
`results/mos_baselines.json` and `results/mos_analysis.json`:

| Metric | TID2013 | KADID-10K | PIPAL | Mean |
|---|---|---|---|---|
| PSNR | 0.687 | 0.676 | 0.413 | 0.592 |
| SSIM | 0.570 | 0.641 | 0.521 | 0.577 |
| LPIPS | 0.745 | 0.822 | 0.591 | 0.719 |
| DISTS | 0.708 | 0.814 | 0.584 | 0.702 |
| NLPD | 0.799 | 0.812 | 0.491 | 0.701 |
| WaSP-Percept (Branch B) | 0.725 | 0.815 | 0.603 | 0.714 |
| WaSP | 0.791 | 0.712 | 0.552 | 0.685 |

In the controlled test (44 references, distortions matched to 31 and 34 dB PSNR), WaSP scores
hallucinated texture in smooth regions as worse than the equal-PSNR control in 99.8% of the
comparisons, against 91.4% for LPIPS.

## Licenses

The code in this repository is released under the MIT license (`LICENSE`), except
`models/third_party/sinsr/`, which comes from [SinSR](https://github.com/wyf0912/SinSR) and
[ResShift](https://github.com/zsyOAOA/ResShift) and is licensed CC BY-NC-SA 4.0 for non-commercial
research use (see its `LICENSE`). Pretrained weights downloaded by the loaders are subject to the
terms of their authors. The released fine-tuned SinSR checkpoints derive from SinSR weights and fall
under the same non-commercial terms.

## Citation

```bibtex
@article{kumar_wasp,
  title   = {{WaSP}: A Wavelet-based Semantically-guided Perceptual Metric for Image Super-Resolution},
  author  = {Kumar, Devesh and Sikka, Geeta and Lin, Chia-Chen and Singh, Samayveer},
  journal = {Scientific Reports},
  note    = {Submitted}
}
```
