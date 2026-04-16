#!/usr/bin/env bash
# Re-run test/eval only on experiments that trained but failed evaluation.
# Uses existing checkpoints — no retraining.
#
# Usage:
#   bash scripts/reeval_incomplete.sh results/op3_unseen_comparison
#   bash scripts/reeval_incomplete.sh results/op3_fair_comparison
set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <results_dir> [gpu_id]"
  echo "  results_dir: e.g. results/op3_unseen_comparison"
  echo "  gpu_id: GPU to use (default: 0)"
  exit 1
fi

RESULTS_DIR="$1"
GPU_ID="${2:-0}"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "Results directory not found: $RESULTS_DIR"
  exit 1
fi

count=0
for d in "$RESULTS_DIR"/*/; do
  name=$(basename "$d")
  summary="$d/evaluation/summary.csv"
  ckpt_dir="$d/checkpoints"
  overrides="$d/.hydra/overrides.yaml"

  # Skip if already has evaluation
  [ -f "$summary" ] && continue

  # Skip if no checkpoint (never trained)
  if [ ! -d "$ckpt_dir" ] || [ -z "$(ls "$ckpt_dir"/*.ckpt 2>/dev/null)" ]; then
    echo "Skipping $name (no checkpoint)"
    continue
  fi

  # Get the experiment override from the original hydra config
  if [ ! -f "$overrides" ]; then
    echo "Skipping $name (no .hydra/overrides.yaml)"
    continue
  fi

  experiment=$(grep "experiment=" "$overrides" | sed 's/- //')
  ckpt=$(ls "$ckpt_dir"/*.ckpt | head -1)

  echo "Re-evaluating: $name"
  echo "  Checkpoint: $ckpt"
  echo "  Override: $experiment"

  CUDA_VISIBLE_DEVICES=$GPU_ID uv run train \
    "$experiment" \
    "hydra.run.dir=$d" \
    train=false \
    "ckpt_path='$ckpt'"

  if [ -f "$summary" ]; then
    echo "  Success: $summary"
  else
    echo "  WARNING: evaluation still missing"
  fi

  count=$((count + 1))
done

if [ $count -eq 0 ]; then
  echo "No incomplete experiments found in $RESULTS_DIR"
else
  echo "Re-evaluated $count experiments"
fi
