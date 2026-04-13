#!/usr/bin/env bash
# Unseen perturbation comparison: run all experiments where test perturbations
# are completely unseen during training. Tests whether molecule embeddings
# enable generalization to novel compounds.
#
# Usage:
#   bash scripts/run_unseen_comparison.sh           # skip completed experiments
#   bash scripts/run_unseen_comparison.sh --force    # retrain everything
#   bash scripts/run_unseen_comparison.sh --early-stopping 50
set -e

RESULTS_DIR="results/unseen_comparison"
FORCE=false
EARLY_STOPPING_PATIENCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --force)
      FORCE=true
      shift
      ;;
    --early-stopping)
      if [ $# -lt 2 ]; then
        echo "Missing value for --early-stopping"
        exit 1
      fi
      EARLY_STOPPING_PATIENCE="$2"
      if ! [[ "$EARLY_STOPPING_PATIENCE" =~ ^[0-9]+$ ]] || [ "$EARLY_STOPPING_PATIENCE" -lt 1 ]; then
        echo "--early-stopping must be a positive integer (patience in epochs)"
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

TRAIN_OVERRIDES=("callbacks=no_early_stopping")
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  TRAIN_OVERRIDES=(
    "callbacks=default"
    "callbacks.early_stopping.patience=$EARLY_STOPPING_PATIENCE"
  )
fi

EXPERIMENTS=(
  # Baselines (control matching)
  fair_comparison/sciplex3_unseen/linear_onehot
  fair_comparison/sciplex3_unseen/linear_ecfp
  fair_comparison/sciplex3_unseen/linear_lpm
  fair_comparison/sciplex3_unseen/latent_onehot
  fair_comparison/sciplex3_unseen/latent_ecfp
  fair_comparison/sciplex3_unseen/latent_lpm
  fair_comparison/sciplex3_unseen/decoder_onehot
  fair_comparison/sciplex3_unseen/decoder_ecfp
  fair_comparison/sciplex3_unseen/decoder_lpm
  # CPA (disentanglement)
  fair_comparison/sciplex3_unseen/cpa_onehot
  fair_comparison/sciplex3_unseen/cpa_ecfp
  fair_comparison/sciplex3_unseen/cpa_lpm
  # CPA noAdv (disentanglement, no adversary)
  fair_comparison/sciplex3_unseen/cpa_noadv_onehot
  fair_comparison/sciplex3_unseen/cpa_noadv_ecfp
  fair_comparison/sciplex3_unseen/cpa_noadv_lpm
)

# Derive a short name from the experiment path for the output directory
exp_name() {
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
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  echo "Early stopping enabled (patience=$EARLY_STOPPING_PATIENCE)"
else
  echo "Early stopping disabled"
fi

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
      "hydra.run.dir=$out_dir" \
      "${TRAIN_OVERRIDES[@]}"
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

echo "GPU 0 queue (${#gpu0_exps[@]}): $(printf '%s ' "${gpu0_exps[@]}" | sed 's|fair_comparison/sciplex3_unseen/||g')"
echo "GPU 1 queue (${#gpu1_exps[@]}): $(printf '%s ' "${gpu1_exps[@]}" | sed 's|fair_comparison/sciplex3_unseen/||g')"
echo "---"

run_gpu 0 "${gpu0_exps[@]}" &
run_gpu 1 "${gpu1_exps[@]}" &

wait
echo "All experiments complete. Results in $RESULTS_DIR/"
