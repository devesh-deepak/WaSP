#!/usr/bin/env bash
# Regenerates every table and figure that needs no GPU from the caches in results/:
# benchmark / gain / ablation / loss-control tables (x4, x2, and the mean-form snapshot),
# the MOS analysis, the hallucination validation and the figures built from them.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 4 2; do
  python evaluation/make_tables.py --scale "$s"
  python evaluation/make_tables.py --scale "$s" --msssim_form mean
done
python evaluation/mos_analysis.py
python evaluation/hal_validation.py
python evaluation/real_sr_hallucination.py          # uses results/pipal_sr_outputs.npz
python evaluation/patch_scale_analysis.py report
python figures/fig_tradeoff.py
python figures/fig_mos_correlation.py
python figures/fig_hparam.py
python figures/fig_weight_tradeoff.py
echo "tables: results/tables/   figures: results/figures/"
