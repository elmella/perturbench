#!/usr/bin/env bash
# OP3 (de_test gene subset) 4-fold CV over unseen perturbations.
#
# Runs each (model, embedding) pair across all 4 folds. Each fold holds out
# 25% of perturbations as test; val is a rotated 25%; train is the remaining 50%.
#
# Usage:
#   bash scripts/run_op3_de_cv_comparison.sh                  # fixed embeddings only
#   bash scripts/run_op3_de_cv_comparison.sh --learnable      # add learnable ECFP/LPM
#   bash scripts/run_op3_de_cv_comparison.sh --force          # retrain everything
#   bash scripts/run_op3_de_cv_comparison.sh --folds 0,1      # subset of folds
#   bash scripts/run_op3_de_cv_comparison.sh --models latent,cpa
#
# Results are organized as: $RESULTS_DIR/<experiment>/fold<k>/
# Automatically resumes from checkpoint if a run was interrupted mid-training.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RESULTS_DIR="results/op3_de_cv_comparison"
FORCE=false
LEARNABLE=false
N_FOLDS=4
FOLDS=""              # comma-separated list; default = all folds
MODEL_FILTER=""

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --force) FORCE=true; shift ;;
    --learnable) LEARNABLE=true; shift ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --folds=*) FOLDS="${arg#*=}"; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${arg#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

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

matches_model_filter() {
  local name="$1"
  [ -z "$MODEL_FILTER" ] && return 0
  IFS=',' read -ra filters <<< "$MODEL_FILTER"
  for prefix in "${filters[@]}"; do
    prefix=$(echo "$prefix" | xargs)
    [[ "$name" == ${prefix}* ]] && return 0
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

    CUDA_VISIBLE_DEVICES=$gpu_id uv run train \
      experiment="$exp" \
      "hydra.run.dir=$out_dir" \
      "data.splitter.fold=$fold" \
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
