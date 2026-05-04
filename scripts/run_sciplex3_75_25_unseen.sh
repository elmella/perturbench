#!/usr/bin/env bash
# Sciplex3 75/25 unseen-perturbation comparison: L1000 vs ECFP vs onehot.
#
# - Perturbation pool: notebooks/neurips2025/perturbench_data/sciplex3_lpm_available_perturbations.txt
#   (177 compounds — every sciplex3 pert that has a non-null _all LPM embedding
#    in epoch_20.pkl, after the (+)-JQ1 -> JQ1 remap). ECFP is pre-computed for
#    all 177 in fp_lmp_embeddings/tahoe_sci_op3_updated.pkl.
# - Test set: 44 compounds (25%, deterministic with seed=42), held out at the
#   compound level — never seen during training in any cell type.
# - Train: 133 compounds (75%) across all 3 cell types.
# - Val: 8% per-(pert, cell_type) stratum carved from train cells (in-distribution).
# - Test compounds list: notebooks/neurips2025/perturbench_data/sciplex3_test_compounds_75_25_lpm_seed42.txt
#
# Usage:
#   bash scripts/run_sciplex3_75_25_unseen.sh                 # epoch 20, _all column
#   bash scripts/run_sciplex3_75_25_unseen.sh --epoch 25
#   bash scripts/run_sciplex3_75_25_unseen.sh --column l1000
#   bash scripts/run_sciplex3_75_25_unseen.sh --emb-path PATH --column COL
#   bash scripts/run_sciplex3_75_25_unseen.sh --tag mytag
#   bash scripts/run_sciplex3_75_25_unseen.sh --force
#   bash scripts/run_sciplex3_75_25_unseen.sh --models latent,decoder
#   bash scripts/run_sciplex3_75_25_unseen.sh --early-stopping 50
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_run_common.sh"

EMB_DIR="all_l1000_25_epochs/sci_single_run_all_l1000_25_epochs"
EMB_PATH=""
EPOCH="20"
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

RESULTS_DIR="results/sciplex3_75_25_unseen/$TAG"

# Disable early stopping by default (perts already held out for test).
# When early stopping is off we also skip val entirely so train gets 100%
# of the training-compound cells. With --early-stopping we carve a small
# in-distribution val slice for val_loss-based stopping.
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  TRAIN_OVERRIDES=(
    "callbacks=default"
    "callbacks.early_stopping.patience=$EARLY_STOPPING_PATIENCE"
    "data.splitter.val_cell_fraction=0.08"
  )
else
  # No val carving: train uses 100% of training-compound cells. Switch the
  # model_checkpoint monitor and the LR scheduler monitor to train_loss since
  # val_loss won't be logged (ModelCheckpoint defaults to mode='min').
  TRAIN_OVERRIDES=(
    "callbacks=no_early_stopping"
    "data.splitter.val_cell_fraction=0.0"
    "callbacks.model_checkpoint.monitor=train_loss"
    "+lr_monitor_key=train_loss"
  )
fi
# Switch all experiments onto the 75/25 fixed-test splitter (177-pert LPM pool)
# and evaluate on the held-out test set (default evaluation is on val).
# Also save last.ckpt every epoch so killed runs can resume from where they left off.
TRAIN_OVERRIDES+=(
  "data/splitter=sciplex3_unseen_75_25_lpm"
  "data/evaluation=final_test"
  "data.perturbation_subset_path=${PWD}/notebooks/neurips2025/perturbench_data/sciplex3_lpm_available_perturbations.txt"
  "+callbacks.model_checkpoint.save_last=true"
)
# L1000-specific: point at the chosen embedding file/column (no-op for ECFP/onehot)
EMB_ABS="$(realpath "$EMB_PATH")"
LPM_OVERRIDES=(
  "data.perturbation_embedding_path=$EMB_ABS"
  "data.perturbation_embedding_column=$EMB_COLUMN"
)

echo "Pool:       sciplex3_lpm_available_perturbations.txt (177 compounds)"
echo "Test list:  sciplex3_test_compounds_75_25_lpm_seed42.txt (44 compounds)"
echo "Train:      133 compounds, all cell types"
echo "L1000 emb:  $EMB_PATH ($EMB_COLUMN)"
echo "Results:    $RESULTS_DIR"
echo "---"

# Reuses fair_comparison/sciplex3/* model configs. The splitter and
# perturbation_subset_path overrides above swap them onto the 177-pert
# LPM-available pool with a held-out 25% test set.
EXPERIMENTS_L1000=(
  fair_comparison/sciplex3/linear_l1000
  fair_comparison/sciplex3/latent_l1000
  fair_comparison/sciplex3/decoder_l1000
  fair_comparison/sciplex3/cpa_l1000
  fair_comparison/sciplex3/cpa_noadv_l1000
)
EXPERIMENTS_OTHER=(
  fair_comparison/sciplex3/linear_ecfp
  fair_comparison/sciplex3/latent_ecfp
  fair_comparison/sciplex3/decoder_ecfp
  fair_comparison/sciplex3/cpa_ecfp
  fair_comparison/sciplex3/cpa_noadv_ecfp
  fair_comparison/sciplex3/linear_onehot
  fair_comparison/sciplex3/latent_onehot
  fair_comparison/sciplex3/decoder_onehot
  fair_comparison/sciplex3/cpa_onehot
  fair_comparison/sciplex3/cpa_noadv_onehot
)

# We need different TRAIN_OVERRIDES for L1000 vs others, so run them in two
# passes (each pass parallelizes across 2 GPUs).
ALL_OVERRIDES_BASE=("${TRAIN_OVERRIDES[@]}")

run_pass() {
  local label="$1"; shift
  local -a extra_overrides=()
  if [ "$1" = "--with-l1000" ]; then
    extra_overrides=("${LPM_OVERRIDES[@]}")
    shift
  fi
  TRAIN_OVERRIDES=("${ALL_OVERRIDES_BASE[@]}" "${extra_overrides[@]}")
  echo ""
  echo "=== Pass: $label ==="
  filter_experiments "$FORCE" "$@"
  run_parallel "${to_run[@]}"
}

run_pass "L1000" --with-l1000 "${EXPERIMENTS_L1000[@]}"
run_pass "ECFP + onehot" "${EXPERIMENTS_OTHER[@]}"

echo "All passes complete. Results in $RESULTS_DIR/"
