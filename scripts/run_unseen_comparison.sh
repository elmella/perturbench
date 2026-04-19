#!/usr/bin/env bash
# Unseen perturbation comparison (Sciplex3): test perturbations are completely
# unseen during training.
#
# Usage:
#   bash scripts/run_unseen_comparison.sh                       # fixed embeddings only
#   bash scripts/run_unseen_comparison.sh --learnable           # add learnable ECFP/LPM
#   bash scripts/run_unseen_comparison.sh --force               # retrain everything
#   bash scripts/run_unseen_comparison.sh --early-stopping 50
#
# Automatically resumes from checkpoint if a run was interrupted mid-training.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_run_common.sh"

RESULTS_DIR="results/unseen_comparison"
FORCE=false
LEARNABLE=false
LEARNABLE_ONEHOT=false
EARLY_STOPPING_PATIENCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=true; shift ;;
    --learnable) LEARNABLE=true; shift ;;
    --learnable-onehot) LEARNABLE=true; LEARNABLE_ONEHOT=true; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${1#*=}"; shift ;;
    --early-stopping)
      EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

TRAIN_OVERRIDES=("callbacks=no_early_stopping")
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  TRAIN_OVERRIDES=(
    "callbacks=default"
    "callbacks.early_stopping.patience=$EARLY_STOPPING_PATIENCE"
  )
fi

EXPERIMENTS=(
  fair_comparison/sciplex3_unseen/linear_onehot
  fair_comparison/sciplex3_unseen/linear_ecfp
  fair_comparison/sciplex3_unseen/linear_lpm
  fair_comparison/sciplex3_unseen/latent_onehot
  fair_comparison/sciplex3_unseen/latent_ecfp
  fair_comparison/sciplex3_unseen/latent_lpm
  fair_comparison/sciplex3_unseen/decoder_onehot
  fair_comparison/sciplex3_unseen/decoder_ecfp
  fair_comparison/sciplex3_unseen/decoder_lpm
  fair_comparison/sciplex3_unseen/cpa_onehot
  fair_comparison/sciplex3_unseen/cpa_ecfp
  fair_comparison/sciplex3_unseen/cpa_lpm
  fair_comparison/sciplex3_unseen/cpa_noadv_onehot
  fair_comparison/sciplex3_unseen/cpa_noadv_ecfp
  fair_comparison/sciplex3_unseen/cpa_noadv_lpm
)

if [ "$LEARNABLE" = true ]; then
  EXPERIMENTS+=(
    fair_comparison/sciplex3_unseen/linear_ecfp_learnable
    fair_comparison/sciplex3_unseen/linear_lpm_learnable
    fair_comparison/sciplex3_unseen/latent_ecfp_learnable
    fair_comparison/sciplex3_unseen/latent_lpm_learnable
    fair_comparison/sciplex3_unseen/decoder_ecfp_learnable
    fair_comparison/sciplex3_unseen/decoder_lpm_learnable
    fair_comparison/sciplex3_unseen/cpa_ecfp_learnable
    fair_comparison/sciplex3_unseen/cpa_lpm_learnable
    fair_comparison/sciplex3_unseen/cpa_noadv_ecfp_learnable
    fair_comparison/sciplex3_unseen/cpa_noadv_lpm_learnable
  )
fi

filter_experiments "$FORCE" "${EXPERIMENTS[@]}"
run_parallel "${to_run[@]}"
