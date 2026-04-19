#!/usr/bin/env bash
# Shared functions for run scripts. Source this, don't execute directly.
# Expects RESULTS_DIR, to_run array, and TRAIN_OVERRIDES (optional) to be set.
# Optional: MODEL_FILTER — comma-separated list of model prefixes to include
#   e.g. "latent,decoder,cpa_noadv" matches latent_*, decoder_*, cpa_noadv_*

run_gpu() {
  local gpu_id=$1
  shift
  for exp in "$@"; do
    local name
    name=$(basename "$exp")
    local out_dir="$RESULTS_DIR/$name"
    local ckpt_dir="$out_dir/checkpoints"

    # Check if there's an existing checkpoint to resume from
    local ckpt_arg=""
    if [ -d "$ckpt_dir" ] && ls "$ckpt_dir"/*.ckpt &>/dev/null; then
      local ckpt
      ckpt=$(ls -t "$ckpt_dir"/*.ckpt | head -1)
      ckpt_arg="ckpt_path='$ckpt'"
      echo "[GPU $gpu_id] Resuming $name from $(basename $ckpt)"
    else
      echo "[GPU $gpu_id] Starting $name (fresh)"
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id uv run train \
      experiment="$exp" \
      "hydra.run.dir=$out_dir" \
      ${TRAIN_OVERRIDES[@]:+"${TRAIN_OVERRIDES[@]}"} \
      $ckpt_arg

    echo "[GPU $gpu_id] Finished $name"
  done
}

run_parallel() {
  local -a to_run=("$@")

  if [ ${#to_run[@]} -eq 0 ]; then
    echo "Nothing to run."
    return
  fi

  # Round-robin across 2 GPUs
  local -a gpu0_exps=()
  local -a gpu1_exps=()
  for i in "${!to_run[@]}"; do
    if (( i % 2 == 0 )); then
      gpu0_exps+=("${to_run[$i]}")
    else
      gpu1_exps+=("${to_run[$i]}")
    fi
  done

  echo "GPU 0 queue (${#gpu0_exps[@]}): $(printf '%s ' "${gpu0_exps[@]}" | xargs -n1 basename | tr '\n' ' ')"
  echo "GPU 1 queue (${#gpu1_exps[@]}): $(printf '%s ' "${gpu1_exps[@]}" | xargs -n1 basename | tr '\n' ' ')"
  echo "---"

  run_gpu 0 "${gpu0_exps[@]}" &
  run_gpu 1 "${gpu1_exps[@]}" &

  wait
  echo "All experiments complete. Results in $RESULTS_DIR/"
}

matches_model_filter() {
  # Check if experiment name matches MODEL_FILTER.
  # Returns 0 (true) if no filter set, or if name starts with any filter prefix.
  local name="$1"
  [ -z "$MODEL_FILTER" ] && return 0

  IFS=',' read -ra filters <<< "$MODEL_FILTER"
  for prefix in "${filters[@]}"; do
    prefix=$(echo "$prefix" | xargs)  # trim whitespace
    if [[ "$name" == ${prefix}* ]]; then
      return 0
    fi
  done
  return 1
}

filter_experiments() {
  # Filters EXPERIMENTS array into to_run, skipping completed ones
  # and applying MODEL_FILTER if set. Sets global to_run array.
  local force="$1"
  shift
  local -a experiments=("$@")

  to_run=()
  for exp in "${experiments[@]}"; do
    local name
    name=$(basename "$exp")

    # Apply model filter
    if ! matches_model_filter "$name"; then
      continue
    fi

    local summary="$RESULTS_DIR/$name/evaluation/summary.csv"
    if [ "$force" = true ] || [ ! -f "$summary" ]; then
      to_run+=("$exp")
    else
      echo "Skipping $name (results exist)"
    fi
  done

  if [ -n "$MODEL_FILTER" ]; then
    echo "Model filter: $MODEL_FILTER"
  fi
  echo "Running ${#to_run[@]} of ${#experiments[@]} experiments"
}
