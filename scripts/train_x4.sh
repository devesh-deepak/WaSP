#!/usr/bin/env bash
# x4 fine-tuning runs of the benchmark and loss-control tables: 7 models x (WaSP + 4 controls),
# 20 warm-up + 200 epochs each. The paper's x4 checkpoints were trained with the originally
# submitted WaSP as the loss (WASP_VARIANT=legacy, the default here).
# SRResNet starts from checkpoints/x4/srresnet_l1.pth (training/train_srresnet_l1.py).
# Runs one after another on the visible GPU; extra arguments are passed to finetune.py.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_x4.sh
#   MODELS="rcan edsr" LOSSES=wasp bash scripts/train_x4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT=${WASP_VARIANT:-legacy}
MODELS=${MODELS:-"srresnet rcan edsr realesrnet realesrgan swinir sinsr"}
LOSSES=${LOSSES:-"wasp l1 lpips dists msssim"}

hparams() {   # learning rate and batch size per model (fine-tuning hyperparameter table)
  case $1 in
    realesrgan) echo "--lr 1e-5 --batch 8" ;;
    swinir)     echo "--lr 2e-5 --batch 4" ;;
    *)          echo "--lr 5e-5 --batch 8" ;;
  esac
}

for model in $MODELS; do
  for loss in $LOSSES; do
    [ "$loss" = wasp ] && name=${model}_x4_wasp || name=${model}_x4_ctrl_${loss}
    python training/finetune.py --model "$model" --scale 4 --loss "$loss" --wasp_variant "$VARIANT" \
      --warmup_epochs 20 --finetune_epochs 200 $(hparams "$model") --run_name "$name" "$@"
  done
done
