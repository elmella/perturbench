#!/usr/bin/env bash
# Fair comparison (Sciplex3): run all experiments on the common perturbation subset.
#
# Usage:
#   bash scripts/run_fair_comparison.sh                       # fixed embeddings only
#   bash scripts/run_fair_comparison.sh --learnable           # add learnable ECFP/LPM
#   bash scripts/run_fair_comparison.sh --learnable-onehot    # also add learnable one-hot
#   bash scripts/run_fair_comparison.sh --force               # retrain everything
#
# Automatically resumes from checkpoint if a run was interrupted mid-training.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_run_common.sh"

RESULTS_DIR="results/fair_comparison"
FORCE=false
LEARNABLE=false
LEARNABLE_ONEHOT=false

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --force) FORCE=true; shift ;;
    --learnable) LEARNABLE=true; shift ;;
    --learnable-onehot) LEARNABLE=true; LEARNABLE_ONEHOT=true; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${arg#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

EXPERIMENTS=(
  fair_comparison/sciplex3/linear_onehot
  fair_comparison/sciplex3/linear_ecfp
  fair_comparison/sciplex3/linear_lpm
  fair_comparison/sciplex3/latent_onehot
  fair_comparison/sciplex3/latent_ecfp
  fair_comparison/sciplex3/latent_lpm
  fair_comparison/sciplex3/decoder_onehot
  fair_comparison/sciplex3/decoder_ecfp
  fair_comparison/sciplex3/decoder_lpm
  fair_comparison/sciplex3/cpa_onehot
  fair_comparison/sciplex3/cpa_ecfp
  fair_comparison/sciplex3/cpa_lpm
  fair_comparison/sciplex3/cpa_noadv_onehot
  fair_comparison/sciplex3/cpa_noadv_ecfp
  fair_comparison/sciplex3/cpa_noadv_lpm
  fair_comparison/sciplex3/sams_onehot
  fair_comparison/sciplex3/sams_s_onehot
)

if [ "$LEARNABLE" = true ]; then
  EXPERIMENTS+=(
    fair_comparison/sciplex3/linear_ecfp_learnable
    fair_comparison/sciplex3/linear_lpm_learnable
    fair_comparison/sciplex3/latent_ecfp_learnable
    fair_comparison/sciplex3/latent_lpm_learnable
    fair_comparison/sciplex3/decoder_ecfp_learnable
    fair_comparison/sciplex3/decoder_lpm_learnable
    fair_comparison/sciplex3/cpa_ecfp_learnable
    fair_comparison/sciplex3/cpa_lpm_learnable
    fair_comparison/sciplex3/cpa_noadv_ecfp_learnable
    fair_comparison/sciplex3/cpa_noadv_lpm_learnable
  )
  if [ "$LEARNABLE_ONEHOT" = true ]; then
    EXPERIMENTS+=(
      fair_comparison/sciplex3/linear_onehot_learnable
      fair_comparison/sciplex3/latent_onehot_learnable
      fair_comparison/sciplex3/decoder_onehot_learnable
      fair_comparison/sciplex3/cpa_onehot_learnable
      fair_comparison/sciplex3/cpa_noadv_onehot_learnable
    )
  fi
fi

filter_experiments "$FORCE" "${EXPERIMENTS[@]}"
run_parallel "${to_run[@]}"
