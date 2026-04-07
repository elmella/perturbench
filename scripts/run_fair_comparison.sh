#!/usr/bin/env bash
# Fair comparison: run all experiments on the common perturbation subset,
# distributing sequentially across 2 GPUs (one queue per GPU, running in parallel).
#
# Usage:
#   bash scripts/run_fair_comparison.sh           # skip completed experiments
#   bash scripts/run_fair_comparison.sh --force    # retrain everything
set -e

RESULTS_DIR="results/fair_comparison"
FORCE=false

for arg in "$@"; do
  case $arg in
    --force) FORCE=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

EXPERIMENTS=(
  # Baselines (control matching)
  fair_comparison/sciplex3/linear_onehot
  fair_comparison/sciplex3/linear_ecfp
  fair_comparison/sciplex3/linear_lpm
  fair_comparison/sciplex3/latent_onehot
  fair_comparison/sciplex3/latent_ecfp
  fair_comparison/sciplex3/latent_lpm
  fair_comparison/sciplex3/decoder_onehot
  fair_comparison/sciplex3/decoder_ecfp
  fair_comparison/sciplex3/decoder_lpm
  # CPA (disentanglement)
  fair_comparison/sciplex3/cpa_onehot
  fair_comparison/sciplex3/cpa_ecfp
  fair_comparison/sciplex3/cpa_lpm
  # CPA noAdv (disentanglement, no adversary)
  fair_comparison/sciplex3/cpa_noadv_onehot
  fair_comparison/sciplex3/cpa_noadv_ecfp
  fair_comparison/sciplex3/cpa_noadv_lpm
  # SAMS-VAE (sparse additive mechanism)
  fair_comparison/sciplex3/sams_onehot
  fair_comparison/sciplex3/sams_ecfp
  fair_comparison/sciplex3/sams_lpm
  # SAMS-VAE(S) (simplified, no sparsity)
  fair_comparison/sciplex3/sams_s_onehot
  fair_comparison/sciplex3/sams_s_ecfp
  fair_comparison/sciplex3/sams_s_lpm
)

# Derive a short name from the experiment path for the output directory
exp_name() {
  # fair_comparison/sciplex3/cpa_onehot -> cpa_onehot
  basename "$1"
}

# Filter to only experiments that need to run
to_run=()
for exp in "${EXPERIMENTS[@]}"; do
  name=$(exp_name "$exp")
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

echo "GPU 0 queue (${#gpu0_exps[@]}): $(printf '%s ' "${gpu0_exps[@]}" | sed 's|fair_comparison/sciplex3/||g')"
echo "GPU 1 queue (${#gpu1_exps[@]}): $(printf '%s ' "${gpu1_exps[@]}" | sed 's|fair_comparison/sciplex3/||g')"
echo "---"

run_gpu 0 "${gpu0_exps[@]}" &
run_gpu 1 "${gpu1_exps[@]}" &

wait
echo "All experiments complete. Results in $RESULTS_DIR/"
