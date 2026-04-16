#!/usr/bin/env bash
# OP3 unseen perturbation comparison: test perturbations are completely unseen
# during training. Tests whether molecule embeddings enable generalization
# to novel compounds in primary PBMC cell types.
#
# Usage:
#   bash scripts/run_op3_unseen_comparison.sh           # skip completed experiments
#   bash scripts/run_op3_unseen_comparison.sh --force   # retrain everything
set -e

RESULTS_DIR="results/op3_unseen_comparison"
FORCE=false

for arg in "$@"; do
  case $arg in
    --force) FORCE=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

EXPERIMENTS=(
  # Baselines (control matching)
  fair_comparison/op3_unseen/linear_onehot
  fair_comparison/op3_unseen/linear_ecfp
  fair_comparison/op3_unseen/linear_lpm
  fair_comparison/op3_unseen/latent_onehot
  fair_comparison/op3_unseen/latent_ecfp
  fair_comparison/op3_unseen/latent_lpm
  fair_comparison/op3_unseen/decoder_onehot
  fair_comparison/op3_unseen/decoder_ecfp
  fair_comparison/op3_unseen/decoder_lpm
  # CPA
  fair_comparison/op3_unseen/cpa_onehot
  fair_comparison/op3_unseen/cpa_ecfp
  fair_comparison/op3_unseen/cpa_lpm
  # CPA noAdv
  fair_comparison/op3_unseen/cpa_noadv_onehot
  fair_comparison/op3_unseen/cpa_noadv_ecfp
  fair_comparison/op3_unseen/cpa_noadv_lpm
)

# Filter to only experiments that need to run
to_run=()
for exp in "${EXPERIMENTS[@]}"; do
  name=$(basename "$exp")
  summary="$RESULTS_DIR/$name/evaluation/summary.csv"
  if [ "$FORCE" = true ] || [ ! -f "$summary" ]; then
    to_run+=("$exp")
  else
    echo "Skipping $name (results exist at $summary)"
  fi
done

if [ ${#to_run[@]} -eq 0 ]; then
  echo "All experiments already complete. Use --force to retrain."
  exit 0
fi

echo "Running ${#to_run[@]} of ${#EXPERIMENTS[@]} experiments"

run_gpu() {
  local gpu_id=$1
  shift
  for exp in "$@"; do
    local name
    name=$(basename "$exp")
    local out_dir="$RESULTS_DIR/$name"
    echo "[GPU $gpu_id] Starting $name"
    CUDA_VISIBLE_DEVICES=$gpu_id uv run train \
      experiment="$exp" \
      "hydra.run.dir=$out_dir"
    echo "[GPU $gpu_id] Finished $name"
  done
}

# Round-robin across 2 GPUs
gpu0_exps=()
gpu1_exps=()
for i in "${!to_run[@]}"; do
  if (( i % 2 == 0 )); then
    gpu0_exps+=("${to_run[$i]}")
  else
    gpu1_exps+=("${to_run[$i]}")
  fi
done

echo "GPU 0 queue (${#gpu0_exps[@]}): $(printf '%s ' "${gpu0_exps[@]}" | sed 's|fair_comparison/op3_unseen/||g')"
echo "GPU 1 queue (${#gpu1_exps[@]}): $(printf '%s ' "${gpu1_exps[@]}" | sed 's|fair_comparison/op3_unseen/||g')"
echo "---"

run_gpu 0 "${gpu0_exps[@]}" &
run_gpu 1 "${gpu1_exps[@]}" &

wait
echo "All experiments complete. Results in $RESULTS_DIR/"
