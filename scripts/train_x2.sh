#!/usr/bin/env bash
# x2 fine-tuning runs: 4 models with official x2 weights x (WaSP + 4 controls), 20 + 200 epochs.
# The paper's x2 runs used the corrected WaSP with the arithmetic-mean MS-SSIM form
# (WASP_VARIANT=mean, the default here).
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_x2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT=${WASP_VARIANT:-mean}
MODELS=${MODELS:-"rcan edsr realesrgan swinir"}
LOSSES=${LOSSES:-"wasp l1 lpips dists msssim"}

hparams() {
  case $1 in
    realesrgan) echo "--lr 1e-5 --batch 8" ;;
    swinir)     echo "--lr 2e-5 --batch 4" ;;
    *)          echo "--lr 5e-5 --batch 8" ;;
  esac
}

for model in $MODELS; do
  for loss in $LOSSES; do
    [ "$loss" = wasp ] && name=${model}_x2_wasp || name=${model}_x2_ctrl_${loss}
    python training/finetune.py --model "$model" --scale 2 --loss "$loss" --wasp_variant "$VARIANT" \
      --warmup_epochs 20 --finetune_epochs 200 $(hparams "$model") --run_name "$name" "$@"
  done
done
