#!/usr/bin/env bash
# x4 branch ablation: 6 models x 4 branch subsets of the WaSP loss, 10 warm-up + 50 epochs,
# lr 5e-5 and batch 8 for every model, submitted WaSP as the loss (WASP_VARIANT=legacy).
#   abla_la   L1 + L_A        abla_lb  L1 + L_B
#   abla_lc   L1 + L_C        abla_lalb L1 + L_A + L_B
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_ablation_x4.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT=${WASP_VARIANT:-legacy}
MODELS=${MODELS:-"rcan edsr realesrnet realesrgan swinir sinsr"}

declare -A LAMBDAS=(
  [la]="--lambda_struct 0.1 --lambda_percept 0.0 --lambda_hal 0.0"
  [lb]="--lambda_struct 0.0 --lambda_percept 1.0 --lambda_hal 0.0"
  [lc]="--lambda_struct 0.0 --lambda_percept 0.0 --lambda_hal 0.05"
  [lalb]="--lambda_struct 0.1 --lambda_percept 1.0 --lambda_hal 0.0"
)

for model in $MODELS; do
  for k in la lb lc lalb; do
    python training/finetune.py --model "$model" --scale 4 --loss wasp --wasp_variant "$VARIANT" \
      --warmup_epochs 10 --finetune_epochs 50 --lr 5e-5 --batch 8 ${LAMBDAS[$k]} \
      --run_name "${model}_x4_abla_${k}" "$@"
  done
done
