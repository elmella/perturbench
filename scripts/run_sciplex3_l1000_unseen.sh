#!/usr/bin/env bash
# Sciplex3 unseen-perturbation comparison: L1000 vs ECFP vs onehot baselines.
# Test perturbations are completely unseen during training.
#
# Usage:
#   bash scripts/run_sciplex3_l1000_unseen.sh                          # epoch 25, _all column
#   bash scripts/run_sciplex3_l1000_unseen.sh --epoch 19               # different epoch
#   bash scripts/run_sciplex3_l1000_unseen.sh --column l1000           # _l1000 head
#   bash scripts/run_sciplex3_l1000_unseen.sh --emb-path PATH --column COL
#   bash scripts/run_sciplex3_l1000_unseen.sh --tag mytag              # custom subdir
#   bash scripts/run_sciplex3_l1000_unseen.sh --force
#   bash scripts/run_sciplex3_l1000_unseen.sh --models latent,decoder
#   bash scripts/run_sciplex3_l1000_unseen.sh --early-stopping 50
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_run_common.sh"

EMB_DIR="all_l1000_25_epochs/sci_single_run_all_l1000_25_epochs"
EMB_PATH=""
EPOCH=""
COLUMN_SUFFIX="all"
TAG=""
FORCE=false
EARLY_STOPPING_PATIENCE=""

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --emb-path) EMB_PATH="$2"; shift 2 ;;
    --emb-path=*) EMB_PATH="${arg#*=}"; shift ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --epoch=*) EPOCH="${arg#*=}"; shift ;;
    --column) COLUMN_SUFFIX="$2"; shift 2 ;;
    --column=*) COLUMN_SUFFIX="${arg#*=}"; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --tag=*) TAG="${arg#*=}"; shift ;;
    --force) FORCE=true; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${arg#*=}"; shift ;;
    --early-stopping) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ -z "$EMB_PATH" ]; then
  EPOCH="${EPOCH:-25}"
  EMB_PATH="$EMB_DIR/sci_lpm_style_embeddings_epoch_${EPOCH}.pkl"
fi

case "$COLUMN_SUFFIX" in
  all|l1000) EMB_COLUMN="lpm_style_embeddings_${COLUMN_SUFFIX}" ;;
  lpm_style_embeddings_*) EMB_COLUMN="$COLUMN_SUFFIX" ;;
  *) echo "Unknown --column: $COLUMN_SUFFIX (expected: all, l1000, or full column name)"; exit 1 ;;
esac

if [ ! -f "$EMB_PATH" ]; then
  echo "Embedding file not found: $EMB_PATH"
  exit 1
fi

if [ -z "$TAG" ]; then
  base=$(basename "$EMB_PATH" .pkl)
  TAG="${base}__${EMB_COLUMN}"
fi

RESULTS_DIR="results/sciplex3_l1000_unseen/$TAG"

# Match run_unseen_comparison.sh: disable early stopping by default
TRAIN_OVERRIDES=("callbacks=no_early_stopping")
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  TRAIN_OVERRIDES=(
    "callbacks=default"
    "callbacks.early_stopping.patience=$EARLY_STOPPING_PATIENCE"
  )
fi
TRAIN_OVERRIDES+=(
  "data.perturbation_embedding_path=$(realpath "$EMB_PATH")"
  "data.perturbation_embedding_column=$EMB_COLUMN"
)

echo "Embeddings: $EMB_PATH"
echo "Column:     $EMB_COLUMN"
echo "Splitter:   unseen_perturbation_task"
echo "Results:    $RESULTS_DIR"
echo "---"

EXPERIMENTS=(
  # L1000 (your new embeddings)
  fair_comparison/sciplex3_unseen/linear_l1000
  fair_comparison/sciplex3_unseen/latent_l1000
  fair_comparison/sciplex3_unseen/decoder_l1000
  fair_comparison/sciplex3_unseen/cpa_l1000
  fair_comparison/sciplex3_unseen/cpa_noadv_l1000
  # ECFP (Morgan fingerprint) baselines
  fair_comparison/sciplex3_unseen/linear_ecfp
  fair_comparison/sciplex3_unseen/latent_ecfp
  fair_comparison/sciplex3_unseen/decoder_ecfp
  fair_comparison/sciplex3_unseen/cpa_ecfp
  fair_comparison/sciplex3_unseen/cpa_noadv_ecfp
  # one-hot baselines (only meaningful here for cell-type generalization;
  # on unseen perts they can't represent test perturbations, so they serve
  # as a sanity-check floor)
  fair_comparison/sciplex3_unseen/linear_onehot
  fair_comparison/sciplex3_unseen/latent_onehot
  fair_comparison/sciplex3_unseen/decoder_onehot
  fair_comparison/sciplex3_unseen/cpa_onehot
  fair_comparison/sciplex3_unseen/cpa_noadv_onehot
)

filter_experiments "$FORCE" "${EXPERIMENTS[@]}"
run_parallel "${to_run[@]}"
