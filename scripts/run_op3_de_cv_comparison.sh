#!/usr/bin/env bash
# OP3 (de_test gene subset) 4-fold CV over unseen perturbations.
#
# Runs each (model, embedding) pair across all 4 folds. Test is always 25% of
# compounds (one fold group). The val set depends on --val-type:
#   unseen          (default): val = rotated 25% of compounds (also held out).
#                   train = 50% of compounds.
#   in_distribution:           val = stratified cell-level slice of train perts.
#                              train = 75% of compounds.
#
# Usage:
#   bash scripts/run_op3_de_cv_comparison.sh                  # unseen val, default dir
#   bash scripts/run_op3_de_cv_comparison.sh --val-type in_distribution
#   bash scripts/run_op3_de_cv_comparison.sh --val-cell-fraction 0.05 ...
#   bash scripts/run_op3_de_cv_comparison.sh --out-dir results/my_sweep
#   bash scripts/run_op3_de_cv_comparison.sh --learnable      # add learnable ECFP/LPM
#   bash scripts/run_op3_de_cv_comparison.sh --force          # retrain everything
#   bash scripts/run_op3_de_cv_comparison.sh --folds 0,1      # subset of folds
#   bash scripts/run_op3_de_cv_comparison.sh --models latent,cpa
#
# Default output directory depends on --val-type:
#   unseen          -> results/op3_de_cv_comparison
#   in_distribution -> results/op3_de_cv_comparison_indist
# Override with --out-dir. Existing runs with a completed summary.csv are
# skipped unless --force. Interrupted runs resume from the latest checkpoint.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RESULTS_DIR=""
FORCE=false
LEARNABLE=false
N_FOLDS=4
FOLDS=""              # comma-separated list; default = all folds
MODEL_FILTER=""
VAL_TYPE="unseen"
VAL_CELL_FRACTION=""  # empty -> use splitter config default

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --force) FORCE=true; shift ;;
    --learnable) LEARNABLE=true; shift ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --folds=*) FOLDS="${arg#*=}"; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${arg#*=}"; shift ;;
    --val-type) VAL_TYPE="$2"; shift 2 ;;
    --val-type=*) VAL_TYPE="${arg#*=}"; shift ;;
    --val-cell-fraction) VAL_CELL_FRACTION="$2"; shift 2 ;;
    --val-cell-fraction=*) VAL_CELL_FRACTION="${arg#*=}"; shift ;;
    --out-dir) RESULTS_DIR="$2"; shift 2 ;;
    --out-dir=*) RESULTS_DIR="${arg#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

case "$VAL_TYPE" in
  unseen|in_distribution) ;;
  *) echo "--val-type must be 'unseen' or 'in_distribution'"; exit 1 ;;
esac

if [ -z "$RESULTS_DIR" ]; then
  if [ "$VAL_TYPE" = "in_distribution" ]; then
    RESULTS_DIR="results/op3_de_cv_comparison_indist"
  else
    RESULTS_DIR="results/op3_de_cv_comparison"
  fi
fi

EXPERIMENTS=(
  fair_comparison/op3_de_cv/linear_onehot
  fair_comparison/op3_de_cv/linear_ecfp
  fair_comparison/op3_de_cv/linear_lpm
  fair_comparison/op3_de_cv/latent_onehot
  fair_comparison/op3_de_cv/latent_ecfp
  fair_comparison/op3_de_cv/latent_lpm
  fair_comparison/op3_de_cv/decoder_onehot
  fair_comparison/op3_de_cv/decoder_ecfp
  fair_comparison/op3_de_cv/decoder_lpm
  fair_comparison/op3_de_cv/cpa_onehot
  fair_comparison/op3_de_cv/cpa_ecfp
  fair_comparison/op3_de_cv/cpa_lpm
  fair_comparison/op3_de_cv/cpa_noadv_onehot
  fair_comparison/op3_de_cv/cpa_noadv_ecfp
  fair_comparison/op3_de_cv/cpa_noadv_lpm
)

if [ "$LEARNABLE" = true ]; then
  EXPERIMENTS+=(
    fair_comparison/op3_de_cv/linear_ecfp_learnable
    fair_comparison/op3_de_cv/linear_lpm_learnable
    fair_comparison/op3_de_cv/latent_ecfp_learnable
    fair_comparison/op3_de_cv/latent_lpm_learnable
    fair_comparison/op3_de_cv/decoder_ecfp_learnable
    fair_comparison/op3_de_cv/decoder_lpm_learnable
    fair_comparison/op3_de_cv/cpa_ecfp_learnable
    fair_comparison/op3_de_cv/cpa_lpm_learnable
    fair_comparison/op3_de_cv/cpa_noadv_ecfp_learnable
    fair_comparison/op3_de_cv/cpa_noadv_lpm_learnable
  )
fi

# Resolve fold list
if [ -z "$FOLDS" ]; then
  FOLD_LIST=()
  for ((k=0; k<N_FOLDS; k++)); do FOLD_LIST+=("$k"); done
else
  IFS=',' read -ra FOLD_LIST <<< "$FOLDS"
fi

extract_model_family() {
  # Return the model-family portion of an experiment name (everything before
  # the first embedding token). e.g., cpa_noadv_ecfp_learnable -> cpa_noadv;
  # latent_lpm -> latent. If no embedding token is present, returns the name.
  local name="$1"
  local IFS='_'
  local -a parts
  read -ra parts <<< "$name"
  local family=""
  for p in "${parts[@]}"; do
    case "$p" in
      onehot|ecfp|lpm) break ;;
    esac
    if [ -z "$family" ]; then family="$p"; else family="${family}_$p"; fi
  done
  [ -z "$family" ] && family="$name"
  echo "$family"
}

matches_model_filter() {
  # A filter matches a name if:
  #   - filter == full name (e.g., cpa_ecfp matches only cpa_ecfp), OR
  #   - filter == model family (e.g., cpa matches cpa_* but NOT cpa_noadv_*;
  #     cpa_noadv matches only cpa_noadv_*).
  local name="$1"
  [ -z "$MODEL_FILTER" ] && return 0
  local family
  family="$(extract_model_family "$name")"
  IFS=',' read -ra filters <<< "$MODEL_FILTER"
  for filter in "${filters[@]}"; do
    filter=$(echo "$filter" | xargs)
    [ "$name" = "$filter" ] && return 0
    [ "$family" = "$filter" ] && return 0
  done
  return 1
}

# Build the list of (experiment, fold) jobs to run.
JOBS=()   # each entry is "<experiment>|<fold>"
for exp in "${EXPERIMENTS[@]}"; do
  name=$(basename "$exp")
  matches_model_filter "$name" || continue
  for fold in "${FOLD_LIST[@]}"; do
    out_dir="$RESULTS_DIR/$name/fold${fold}"
    summary="$out_dir/evaluation/summary.csv"
    if [ "$FORCE" = true ] || [ ! -f "$summary" ]; then
      JOBS+=("$exp|$fold")
    else
      echo "Skipping $name fold$fold (results exist)"
    fi
  done
done

if [ ${#JOBS[@]} -eq 0 ]; then
  echo "Nothing to run."
  exit 0
fi

echo "Running ${#JOBS[@]} jobs across folds ${FOLD_LIST[*]}"
echo "val_type: $VAL_TYPE${VAL_CELL_FRACTION:+, val_cell_fraction=$VAL_CELL_FRACTION}"
echo "Output:   $RESULTS_DIR/"
[ -n "$MODEL_FILTER" ] && echo "Model filter: $MODEL_FILTER"

run_gpu() {
  local gpu_id=$1
  shift
  for job in "$@"; do
    local exp="${job%|*}"
    local fold="${job##*|}"
    local name
    name=$(basename "$exp")
    local out_dir="$RESULTS_DIR/$name/fold${fold}"
    local ckpt_dir="$out_dir/checkpoints"

    local ckpt_arg=""
    if [ -d "$ckpt_dir" ] && ls "$ckpt_dir"/*.ckpt &>/dev/null; then
      local ckpt
      ckpt=$(ls -t "$ckpt_dir"/*.ckpt | head -1)
      ckpt_arg="ckpt_path='$ckpt'"
      echo "[GPU $gpu_id] Resuming $name fold$fold from $(basename $ckpt)"
    else
      echo "[GPU $gpu_id] Starting $name fold$fold (fresh)"
    fi

    local extra_overrides=("data.splitter.val_type=$VAL_TYPE")
    if [ -n "$VAL_CELL_FRACTION" ]; then
      extra_overrides+=("data.splitter.val_cell_fraction=$VAL_CELL_FRACTION")
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id uv run train \
      experiment="$exp" \
      "hydra.run.dir=$out_dir" \
      "data.splitter.fold=$fold" \
      "${extra_overrides[@]}" \
      $ckpt_arg

    echo "[GPU $gpu_id] Finished $name fold$fold"
  done
}

# Round-robin across 2 GPUs
gpu0_jobs=()
gpu1_jobs=()
for i in "${!JOBS[@]}"; do
  if (( i % 2 == 0 )); then
    gpu0_jobs+=("${JOBS[$i]}")
  else
    gpu1_jobs+=("${JOBS[$i]}")
  fi
done

echo "GPU 0 queue (${#gpu0_jobs[@]} jobs)"
echo "GPU 1 queue (${#gpu1_jobs[@]} jobs)"
echo "---"

run_gpu 0 "${gpu0_jobs[@]}" &
run_gpu 1 "${gpu1_jobs[@]}" &
wait

echo "All CV jobs complete. Results in $RESULTS_DIR/"
echo "Run aggregation:"
echo "  uv run python scripts/aggregate_cv_metrics.py --results-dir $RESULTS_DIR"
