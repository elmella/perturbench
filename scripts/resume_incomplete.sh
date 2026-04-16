#!/usr/bin/env bash
# Resume training from the last checkpoint for incomplete experiments,
# distributing across 2 GPUs (one queue per GPU, running in parallel).
# Continues from saved optimizer state via Lightning's ckpt_path.
#
# Usage:
#   bash scripts/resume_incomplete.sh results/op3_fair_comparison
#   bash scripts/resume_incomplete.sh results/op3_unseen_comparison
set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <results_dir>"
  exit 1
fi

RESULTS_DIR="$1"

# Find all incomplete experiments with checkpoints
to_resume=()
ckpts=()
overrides_list=()

for d in "$RESULTS_DIR"/*/; do
  name=$(basename "$d")
  summary="$d/evaluation/summary.csv"
  ckpt_dir="$d/checkpoints"
  overrides="$d/.hydra/overrides.yaml"

  [ -f "$summary" ] && continue

  if [ ! -d "$ckpt_dir" ] || [ -z "$(ls "$ckpt_dir"/*.ckpt 2>/dev/null)" ]; then
    echo "Skipping $name (no checkpoint)"
    continue
  fi

  if [ ! -f "$overrides" ]; then
    echo "Skipping $name (no .hydra/overrides.yaml)"
    continue
  fi

  ckpt=$(ls -t "$ckpt_dir"/*.ckpt | head -1)
  experiment=$(grep "experiment=" "$overrides" | sed 's/- //')

  to_resume+=("$d")
  ckpts+=("$ckpt")
  overrides_list+=("$experiment")
done

if [ ${#to_resume[@]} -eq 0 ]; then
  echo "No incomplete experiments found"
  exit 0
fi

echo "Found ${#to_resume[@]} experiments to resume"

run_gpu() {
  local gpu_id=$1
  shift
  while [ $# -ge 3 ]; do
    local d=$1
    local ckpt=$2
    local experiment=$3
    shift 3
    local name
    name=$(basename "$d")
    echo "[GPU $gpu_id] Resuming $name from $(basename $ckpt)"
    CUDA_VISIBLE_DEVICES=$gpu_id uv run train \
      "$experiment" \
      "hydra.run.dir=$d" \
      "ckpt_path='$ckpt'"
    echo "[GPU $gpu_id] Finished $name"
  done
}

# Round-robin assignments
gpu0_args=()
gpu1_args=()
for i in "${!to_resume[@]}"; do
  if (( i % 2 == 0 )); then
    gpu0_args+=("${to_resume[$i]}" "${ckpts[$i]}" "${overrides_list[$i]}")
  else
    gpu1_args+=("${to_resume[$i]}" "${ckpts[$i]}" "${overrides_list[$i]}")
  fi
done

echo "GPU 0: $(( ${#gpu0_args[@]} / 3 )) experiments"
echo "GPU 1: $(( ${#gpu1_args[@]} / 3 )) experiments"
echo "---"

run_gpu 0 "${gpu0_args[@]}" &
run_gpu 1 "${gpu1_args[@]}" &

wait
echo "All resume jobs complete"
