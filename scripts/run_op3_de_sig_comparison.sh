#!/usr/bin/env bash
# OP3 (de-genes) single-split comparison aligned to op3_signatures.
#
# Uses a fixed 35-compound test set (the op3_signatures seed=42 / 25%-of-140
# random choice). Same data, same gene subset as the 4-fold CV sweep — just a
# fixed single train/test split to enable direct comparison with numbers
# produced by the op3_signatures pipeline.
#
# Usage:
#   bash scripts/run_op3_de_sig_comparison.sh                         # default: in_distribution val, fixed embeddings
#   bash scripts/run_op3_de_sig_comparison.sh --val-type unseen
#   bash scripts/run_op3_de_sig_comparison.sh --models cpa,latent,linear
#   bash scripts/run_op3_de_sig_comparison.sh --force                 # retrain everything
#   bash scripts/run_op3_de_sig_comparison.sh --learnable             # add 10 learnable ECFP/LPM variants
#   bash scripts/run_op3_de_sig_comparison.sh --learnable-only        # only the learnable variants
#   bash scripts/run_op3_de_sig_comparison.sh --out-dir results/my_sig_sweep
#
# Output layout: $RESULTS_DIR/<experiment>/ (no fold subdir — single split).
# Existing runs with a completed evaluation/summary.csv are skipped unless
# --force. Interrupted runs resume from the latest checkpoint.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RESULTS_DIR=""
FORCE=false
LEARNABLE=false
LEARNABLE_ONLY=false
MODEL_FILTER=""
VAL_TYPE="in_distribution"
VAL_CELL_FRACTION=""
VAL_UNSEEN_FRACTION=""

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --force) FORCE=true; shift ;;
    --learnable) LEARNABLE=true; shift ;;
    --learnable-only) LEARNABLE=true; LEARNABLE_ONLY=true; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${arg#*=}"; shift ;;
    --val-type) VAL_TYPE="$2"; shift 2 ;;
    --val-type=*) VAL_TYPE="${arg#*=}"; shift ;;
    --val-cell-fraction) VAL_CELL_FRACTION="$2"; shift 2 ;;
    --val-cell-fraction=*) VAL_CELL_FRACTION="${arg#*=}"; shift ;;
    --val-unseen-fraction) VAL_UNSEEN_FRACTION="$2"; shift 2 ;;
    --val-unseen-fraction=*) VAL_UNSEEN_FRACTION="${arg#*=}"; shift ;;
    --out-dir) RESULTS_DIR="$2"; shift 2 ;;
    --out-dir=*) RESULTS_DIR="${arg#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

case "$VAL_TYPE" in
  in_distribution|unseen) ;;
  *) echo "--val-type must be 'in_distribution' or 'unseen'"; exit 1 ;;
esac

if [ -z "$RESULTS_DIR" ]; then
  if [ "$VAL_TYPE" = "unseen" ]; then
    RESULTS_DIR="results/op3_de_sig_comparison_unseenval"
  else
    RESULTS_DIR="results/op3_de_sig_comparison"
  fi
fi

EXPERIMENTS=()

if [ "$LEARNABLE_ONLY" != true ]; then
  EXPERIMENTS+=(
    fair_comparison/op3_de_sig/linear_onehot
    fair_comparison/op3_de_sig/linear_ecfp
    fair_comparison/op3_de_sig/linear_lpm
    fair_comparison/op3_de_sig/latent_onehot
    fair_comparison/op3_de_sig/latent_ecfp
    fair_comparison/op3_de_sig/latent_lpm
    fair_comparison/op3_de_sig/decoder_onehot
    fair_comparison/op3_de_sig/decoder_ecfp
    fair_comparison/op3_de_sig/decoder_lpm
    fair_comparison/op3_de_sig/cpa_onehot
    fair_comparison/op3_de_sig/cpa_ecfp
    fair_comparison/op3_de_sig/cpa_lpm
    fair_comparison/op3_de_sig/cpa_noadv_onehot
    fair_comparison/op3_de_sig/cpa_noadv_ecfp
    fair_comparison/op3_de_sig/cpa_noadv_lpm
  )
fi

if [ "$LEARNABLE" = true ]; then
  EXPERIMENTS+=(
    fair_comparison/op3_de_sig/linear_ecfp_learnable
    fair_comparison/op3_de_sig/linear_lpm_learnable
    fair_comparison/op3_de_sig/latent_ecfp_learnable
    fair_comparison/op3_de_sig/latent_lpm_learnable
    fair_comparison/op3_de_sig/decoder_ecfp_learnable
    fair_comparison/op3_de_sig/decoder_lpm_learnable
    fair_comparison/op3_de_sig/cpa_ecfp_learnable
    fair_comparison/op3_de_sig/cpa_lpm_learnable
    fair_comparison/op3_de_sig/cpa_noadv_ecfp_learnable
    fair_comparison/op3_de_sig/cpa_noadv_lpm_learnable
  )
fi

extract_model_family() {
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

JOBS=()
for exp in "${EXPERIMENTS[@]}"; do
  name=$(basename "$exp")
  matches_model_filter "$name" || continue
  out_dir="$RESULTS_DIR/$name"
  summary="$out_dir/evaluation/summary.csv"
  if [ "$FORCE" = true ] || [ ! -f "$summary" ]; then
    JOBS+=("$exp")
  else
    echo "Skipping $name (results exist)"
  fi
done

if [ ${#JOBS[@]} -eq 0 ]; then
  echo "Nothing to run."
  exit 0
fi

echo "Running ${#JOBS[@]} jobs"
echo "val_type: $VAL_TYPE${VAL_CELL_FRACTION:+, val_cell_fraction=$VAL_CELL_FRACTION}${VAL_UNSEEN_FRACTION:+, val_unseen_fraction=$VAL_UNSEEN_FRACTION}"
echo "Output:   $RESULTS_DIR/"
[ -n "$MODEL_FILTER" ] && echo "Model filter: $MODEL_FILTER"

run_gpu() {
  local gpu_id=$1
  shift
  for exp in "$@"; do
    local name
    name=$(basename "$exp")
    local out_dir="$RESULTS_DIR/$name"
    local ckpt_dir="$out_dir/checkpoints"

    local ckpt_arg=""
    if [ -d "$ckpt_dir" ] && ls "$ckpt_dir"/*.ckpt &>/dev/null; then
      local ckpt
      ckpt=$(ls -t "$ckpt_dir"/*.ckpt | head -1)
      ckpt_arg="ckpt_path='$ckpt'"
      echo "[GPU $gpu_id] Resuming $name from $(basename $ckpt)"
    else
      echo "[GPU $gpu_id] Starting $name (fresh)"
    fi

    local extra_overrides=("data.splitter.val_type=$VAL_TYPE")
    if [ -n "$VAL_CELL_FRACTION" ]; then
      extra_overrides+=("data.splitter.val_cell_fraction=$VAL_CELL_FRACTION")
    fi
    if [ -n "$VAL_UNSEEN_FRACTION" ]; then
      extra_overrides+=("data.splitter.val_unseen_fraction=$VAL_UNSEEN_FRACTION")
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id uv run train \
      experiment="$exp" \
      "hydra.run.dir=$out_dir" \
      "${extra_overrides[@]}" \
      $ckpt_arg

    echo "[GPU $gpu_id] Finished $name"
  done
}

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

echo "All jobs complete. Results in $RESULTS_DIR/"
