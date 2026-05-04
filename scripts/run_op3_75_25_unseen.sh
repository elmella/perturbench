#!/usr/bin/env bash
# OP3 single-split unseen-perturbation evaluation — mirrors sciplex3 75/25 setup.
#
# Test set: op3_signatures_test_compounds_lpm.txt (34 compounds — the original
#   op3-signatures default test list, intersected with the 138-pert LPM-available
#   pool; BMS-265246 dropped). Test compounds are completely unseen.
# Train pool: 138 - 34 = 104 compounds (~75%).
# Dataset: op3_de_genes.h5ad.
#
# Trains 5 archs × 3 embeddings = 15 jobs across 2 GPUs in parallel.
#
# Usage:
#   bash scripts/run_op3_75_25_unseen.sh
#   bash scripts/run_op3_75_25_unseen.sh --emb-path PATH --column COL
#   bash scripts/run_op3_75_25_unseen.sh --tag mytag
#   bash scripts/run_op3_75_25_unseen.sh --models latent,cpa
#   bash scripts/run_op3_75_25_unseen.sh --max-epochs 200
#   bash scripts/run_op3_75_25_unseen.sh --early-stopping 50
#   bash scripts/run_op3_75_25_unseen.sh --force
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_run_common.sh"

DEFAULT_EMB_PATH="all_l1000_25_epochs/op3_emb_all_l1000_20.pkl"
EMB_PATH=""
COLUMN_SUFFIX="all"
TAG=""
FORCE=false
MAX_EPOCHS=""
EARLY_STOPPING_PATIENCE=""

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --emb-path) EMB_PATH="$2"; shift 2 ;;
    --emb-path=*) EMB_PATH="${arg#*=}"; shift ;;
    --column) COLUMN_SUFFIX="$2"; shift 2 ;;
    --column=*) COLUMN_SUFFIX="${arg#*=}"; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --tag=*) TAG="${arg#*=}"; shift ;;
    --force) FORCE=true; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${arg#*=}"; shift ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --max-epochs=*) MAX_EPOCHS="${arg#*=}"; shift ;;
    --early-stopping) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[ -z "$EMB_PATH" ] && EMB_PATH="$DEFAULT_EMB_PATH"
case "$COLUMN_SUFFIX" in
  all|l1000) EMB_COLUMN="lpm_style_embeddings_${COLUMN_SUFFIX}" ;;
  lpm_style_embeddings_*) EMB_COLUMN="$COLUMN_SUFFIX" ;;
  *) echo "Unknown --column: $COLUMN_SUFFIX"; exit 1 ;;
esac
[ ! -f "$EMB_PATH" ] && { echo "Embedding file not found: $EMB_PATH"; exit 1; }

if [ -z "$TAG" ]; then
  base=$(basename "$EMB_PATH" .pkl)
  TAG="${base}__${EMB_COLUMN}"
fi
RESULTS_DIR="results/op3_75_25_unseen/$TAG"
EMB_ABS="$(realpath "$EMB_PATH")"

if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  TRAIN_OVERRIDES=(
    "callbacks=default"
    "callbacks.early_stopping.patience=$EARLY_STOPPING_PATIENCE"
  )
else
  # No val carving: train uses 100% of training-compound cells; check train_loss.
  TRAIN_OVERRIDES=(
    "callbacks=no_early_stopping"
    "callbacks.model_checkpoint.monitor=train_loss"
    "+lr_monitor_key=train_loss"
  )
fi
# Switch all experiments onto the op3-signatures fixed-test splitter (138-pert pool)
# and evaluate on the held-out test set.
TRAIN_OVERRIDES+=(
  "data/splitter=op3_signatures_test_lpm"
  "data/evaluation=final_test"
  "+callbacks.model_checkpoint.save_last=true"
)
[ -n "$MAX_EPOCHS" ] && TRAIN_OVERRIDES+=("trainer.max_epochs=$MAX_EPOCHS")
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  TRAIN_OVERRIDES+=("data.splitter.val_cell_fraction=0.08")
else
  TRAIN_OVERRIDES+=("data.splitter.val_cell_fraction=0.0")
fi

echo "Pool:       op3_lpm_available_perturbations.txt (138 compounds)"
echo "Test list:  op3_signatures_test_compounds_lpm.txt (34 compounds)"
echo "Train:      104 compounds, all cell types"
echo "LPM emb:    $EMB_PATH ($EMB_COLUMN)"
echo "Results:    $RESULTS_DIR"
echo "---"

# Reuse the CV experiment configs (they all point at op3_*_cv data configs which
# already filter to the 138-pert pool with the new LPM file). The splitter
# override above swaps the unseen_cv splitter for fixed_test.
EXPERIMENTS_LPM=(
  fair_comparison/op3_cv/linear_lpm
  fair_comparison/op3_cv/latent_lpm
  fair_comparison/op3_cv/decoder_lpm
  fair_comparison/op3_cv/cpa_lpm
  fair_comparison/op3_cv/cpa_noadv_lpm
)
EXPERIMENTS_OTHER=(
  fair_comparison/op3_cv/linear_ecfp
  fair_comparison/op3_cv/latent_ecfp
  fair_comparison/op3_cv/decoder_ecfp
  fair_comparison/op3_cv/cpa_ecfp
  fair_comparison/op3_cv/cpa_noadv_ecfp
  fair_comparison/op3_cv/linear_onehot
  fair_comparison/op3_cv/latent_onehot
  fair_comparison/op3_cv/decoder_onehot
  fair_comparison/op3_cv/cpa_onehot
  fair_comparison/op3_cv/cpa_noadv_onehot
)

ALL_OVERRIDES_BASE=("${TRAIN_OVERRIDES[@]}")
LPM_OVERRIDES=(
  "data.perturbation_embedding_path=$EMB_ABS"
  "data.perturbation_embedding_column=$EMB_COLUMN"
)

run_pass() {
  local label="$1"; shift
  local -a extra=()
  if [ "$1" = "--with-lpm" ]; then
    extra=("${LPM_OVERRIDES[@]}")
    shift
  fi
  TRAIN_OVERRIDES=("${ALL_OVERRIDES_BASE[@]}" "${extra[@]}")
  echo ""
  echo "=== Pass: $label ==="
  filter_experiments "$FORCE" "$@"
  run_parallel "${to_run[@]}"
}

run_pass "LPM" --with-lpm "${EXPERIMENTS_LPM[@]}"
run_pass "ECFP + onehot" "${EXPERIMENTS_OTHER[@]}"

echo "All passes complete. Results in $RESULTS_DIR/"
echo ""
echo "Plot:"
echo "  uv run python scripts/plot_sciplex3_cv.py --results-dir $RESULTS_DIR"
