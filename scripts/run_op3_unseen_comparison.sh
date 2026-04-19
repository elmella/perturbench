#!/usr/bin/env bash
# OP3 unseen perturbation comparison: tests generalization to novel compounds.
#
# Usage:
#   bash scripts/run_op3_unseen_comparison.sh                       # fixed embeddings only
#   bash scripts/run_op3_unseen_comparison.sh --learnable           # add learnable ECFP/LPM
#   bash scripts/run_op3_unseen_comparison.sh --force               # retrain everything
#
# Automatically resumes from checkpoint if a run was interrupted mid-training.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_run_common.sh"

RESULTS_DIR="results/op3_unseen_comparison"
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
  fair_comparison/op3_unseen/linear_onehot
  fair_comparison/op3_unseen/linear_ecfp
  fair_comparison/op3_unseen/linear_lpm
  fair_comparison/op3_unseen/latent_onehot
  fair_comparison/op3_unseen/latent_ecfp
  fair_comparison/op3_unseen/latent_lpm
  fair_comparison/op3_unseen/decoder_onehot
  fair_comparison/op3_unseen/decoder_ecfp
  fair_comparison/op3_unseen/decoder_lpm
  fair_comparison/op3_unseen/cpa_onehot
  fair_comparison/op3_unseen/cpa_ecfp
  fair_comparison/op3_unseen/cpa_lpm
  fair_comparison/op3_unseen/cpa_noadv_onehot
  fair_comparison/op3_unseen/cpa_noadv_ecfp
  fair_comparison/op3_unseen/cpa_noadv_lpm
)

if [ "$LEARNABLE" = true ]; then
  EXPERIMENTS+=(
    fair_comparison/op3_unseen/linear_ecfp_learnable
    fair_comparison/op3_unseen/linear_lpm_learnable
    fair_comparison/op3_unseen/latent_ecfp_learnable
    fair_comparison/op3_unseen/latent_lpm_learnable
    fair_comparison/op3_unseen/decoder_ecfp_learnable
    fair_comparison/op3_unseen/decoder_lpm_learnable
    fair_comparison/op3_unseen/cpa_ecfp_learnable
    fair_comparison/op3_unseen/cpa_lpm_learnable
    fair_comparison/op3_unseen/cpa_noadv_ecfp_learnable
    fair_comparison/op3_unseen/cpa_noadv_lpm_learnable
  )
fi

filter_experiments "$FORCE" "${EXPERIMENTS[@]}"
run_parallel "${to_run[@]}"
