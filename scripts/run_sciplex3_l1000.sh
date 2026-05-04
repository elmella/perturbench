#!/usr/bin/env bash
# Sciplex3 evaluation with L1000-derived perturbation embeddings.
#
# Usage:
#   bash scripts/run_sciplex3_l1000.sh                          # defaults: epoch 25, _all column
#   bash scripts/run_sciplex3_l1000.sh --epoch 19               # pick an epoch
#   bash scripts/run_sciplex3_l1000.sh --epoch 19 --column l1000  # _l1000 head instead of _all
#   bash scripts/run_sciplex3_l1000.sh --emb-path PATH --column COL  # arbitrary embeddings file
#   bash scripts/run_sciplex3_l1000.sh --tag mytag              # custom results subdir
#   bash scripts/run_sciplex3_l1000.sh --force                  # retrain everything
#   bash scripts/run_sciplex3_l1000.sh --models latent,decoder  # filter by model prefix
#
# Results land in results/sciplex3_l1000/<tag>/<model>/. Default tag is derived
# from the embedding filename + column so different runs don't overwrite each other.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_run_common.sh"

EMB_DIR="all_l1000_25_epochs/sci_single_run_all_l1000_25_epochs"
EMB_PATH=""
EPOCH=""
COLUMN_SUFFIX="all"   # "all" -> lpm_style_embeddings_all, "l1000" -> lpm_style_embeddings_l1000
TAG=""
FORCE=false

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
  *) echo "Unknown --column value: $COLUMN_SUFFIX (expected: all, l1000, or full column name)"; exit 1 ;;
esac

if [ ! -f "$EMB_PATH" ]; then
  echo "Embedding file not found: $EMB_PATH"
  exit 1
fi

if [ -z "$TAG" ]; then
  base=$(basename "$EMB_PATH" .pkl)
  TAG="${base}__${EMB_COLUMN}"
fi

RESULTS_DIR="results/sciplex3_l1000/$TAG"
TRAIN_OVERRIDES=(
  "data.perturbation_embedding_path=$(realpath "$EMB_PATH")"
  "data.perturbation_embedding_column=$EMB_COLUMN"
)

echo "Embeddings: $EMB_PATH"
echo "Column:     $EMB_COLUMN"
echo "Results:    $RESULTS_DIR"
echo "---"

EXPERIMENTS=(
  # L1000 (your new embeddings)
  fair_comparison/sciplex3/linear_l1000
  fair_comparison/sciplex3/latent_l1000
  fair_comparison/sciplex3/decoder_l1000
  fair_comparison/sciplex3/cpa_l1000
  fair_comparison/sciplex3/cpa_noadv_l1000
  # ECFP (Morgan) baselines
  fair_comparison/sciplex3/linear_ecfp
  fair_comparison/sciplex3/latent_ecfp
  fair_comparison/sciplex3/decoder_ecfp
  fair_comparison/sciplex3/cpa_ecfp
  fair_comparison/sciplex3/cpa_noadv_ecfp
  # one-hot baselines
  fair_comparison/sciplex3/linear_onehot
  fair_comparison/sciplex3/latent_onehot
  fair_comparison/sciplex3/decoder_onehot
  fair_comparison/sciplex3/cpa_onehot
  fair_comparison/sciplex3/cpa_noadv_onehot
)

filter_experiments "$FORCE" "${EXPERIMENTS[@]}"
run_parallel "${to_run[@]}"
