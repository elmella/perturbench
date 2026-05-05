#!/usr/bin/env bash
# Sciplex3 4-fold CV over completely unseen perturbations.
#
# Compound pool: 177 sciplex3 compounds with epoch_20 LPM embeddings (also
# covered by ECFP). The unseen_cv_task splitter partitions them into 4 disjoint
# groups via stride; each fold's test = group[fold]. Every compound appears in
# test exactly once across the 4 folds.
#
# Trains 5 architectures × 3 embeddings (L1000, ECFP, onehot) × 4 folds = 60 runs.
# Each (model, fold) combo writes to its own subdirectory; the directory shape
# is identical across embeddings and folds, so aggregation/plotting is generic.
#
# Resume: each run saves a `last.ckpt` after every epoch. If a run is killed,
# rerunning the script resumes the last (model, fold) pair from `last.ckpt`.
#
# Usage:
#   bash scripts/run_sciplex3_cv.sh                       # all 60 jobs
#   bash scripts/run_sciplex3_cv.sh --epoch 25            # different LPM epoch
#   bash scripts/run_sciplex3_cv.sh --column l1000        # _l1000 head
#   bash scripts/run_sciplex3_cv.sh --emb-path PATH --column COL
#   bash scripts/run_sciplex3_cv.sh --tag mytag           # custom subdir name
#   bash scripts/run_sciplex3_cv.sh --folds 0,1           # subset of folds
#   bash scripts/run_sciplex3_cv.sh --models latent,cpa   # filter architectures
#   bash scripts/run_sciplex3_cv.sh --max-epochs 200      # override training length
#   bash scripts/run_sciplex3_cv.sh --early-stopping 50   # enable val carve + early stop
#   bash scripts/run_sciplex3_cv.sh --accelerator mps     # Apple Silicon / Metal
#   bash scripts/run_sciplex3_cv.sh --parallel-jobs 1     # process-level parallelism
#   bash scripts/run_sciplex3_cv.sh --num-workers 2       # dataloader workers per job
#   bash scripts/run_sciplex3_cv.sh --force               # retrain everything
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EMB_DIR="all_l1000_25_epochs/sci_single_run_all_l1000_25_epochs"
EMB_PATH=""
EPOCH="20"
COLUMN_SUFFIX="all"
TAG=""
FORCE=false
N_FOLDS=4
FOLDS=""
MODEL_FILTER=""
MAX_EPOCHS=""
EARLY_STOPPING_PATIENCE=""
ACCELERATOR="gpu"
DEVICES="1"
PRECISION=""
PARALLEL_JOBS=""
NUM_WORKERS=""
# Lower default than the framework's 400 to avoid OOMs during the test phase
# of larger architectures (latent_additive in particular). Override with --eval-chunk-size.
EVAL_CHUNK_SIZE="100"

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --emb-path) EMB_PATH="$2"; shift 2 ;;
    --emb-path=*) EMB_PATH="${arg#*=}"; shift ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --epoch=*) EPOCH="${arg#*=}"; shift ;;
    --column) COLUMN_SUFFIX="$2"; shift 2 ;;
    --column=*) COLUMN_SUFFIX="${arg#*=}"; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    --tag=*) TAG="${arg#*=}"; shift ;;
    --force) FORCE=true; shift ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --folds=*) FOLDS="${arg#*=}"; shift ;;
    --models) MODEL_FILTER="$2"; shift 2 ;;
    --models=*) MODEL_FILTER="${arg#*=}"; shift ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --max-epochs=*) MAX_EPOCHS="${arg#*=}"; shift ;;
    --early-stopping) EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --accelerator=*) ACCELERATOR="${arg#*=}"; shift ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --devices=*) DEVICES="${arg#*=}"; shift ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --precision=*) PRECISION="${arg#*=}"; shift ;;
    --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
    --parallel-jobs=*) PARALLEL_JOBS="${arg#*=}"; shift ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --num-workers=*) NUM_WORKERS="${arg#*=}"; shift ;;
    --eval-chunk-size) EVAL_CHUNK_SIZE="$2"; shift 2 ;;
    --eval-chunk-size=*) EVAL_CHUNK_SIZE="${arg#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ -z "$EMB_PATH" ]; then
  EMB_PATH="$EMB_DIR/sci_lpm_style_embeddings_epoch_${EPOCH}.pkl"
fi
case "$COLUMN_SUFFIX" in
  all|l1000) EMB_COLUMN="lpm_style_embeddings_${COLUMN_SUFFIX}" ;;
  lpm_style_embeddings_*) EMB_COLUMN="$COLUMN_SUFFIX" ;;
  *) echo "Unknown --column: $COLUMN_SUFFIX"; exit 1 ;;
esac
if [ ! -f "$EMB_PATH" ]; then
  echo "Embedding file not found: $EMB_PATH"; exit 1
fi
if [ -z "$TAG" ]; then
  base=$(basename "$EMB_PATH" .pkl)
  TAG="${base}__${EMB_COLUMN}"
fi

RESULTS_DIR="results/sciplex3_cv/$TAG"
EMB_ABS="$(realpath "$EMB_PATH")"

# --------- Build common Hydra overrides ---------
TRAIN_OVERRIDES=()
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
  TRAIN_OVERRIDES+=(
    "callbacks=default"
    "callbacks.early_stopping.patience=$EARLY_STOPPING_PATIENCE"
    "data.splitter.val_cell_fraction=0.08"
    "data.splitter.val_type=in_distribution"
  )
else
  TRAIN_OVERRIDES+=(
    "callbacks=no_early_stopping"
    "callbacks.model_checkpoint.monitor=train_loss"
    "+lr_monitor_key=train_loss"
    "data.splitter.val_cell_fraction=0.0"
    "data.splitter.val_type=in_distribution"
  )
fi
# Always-on: save last.ckpt for resume (model_checkpoint config has no save_last
# field by default, so we add it).
TRAIN_OVERRIDES+=("+callbacks.model_checkpoint.save_last=true")
TRAIN_OVERRIDES+=("data.evaluation.chunk_size=$EVAL_CHUNK_SIZE")
TRAIN_OVERRIDES+=("trainer.accelerator=$ACCELERATOR")
TRAIN_OVERRIDES+=("trainer.devices=$DEVICES")
if [ -n "$MAX_EPOCHS" ]; then
  TRAIN_OVERRIDES+=("trainer.max_epochs=$MAX_EPOCHS")
fi
if [ -n "$NUM_WORKERS" ]; then
  TRAIN_OVERRIDES+=("data.num_workers=$NUM_WORKERS")
  TRAIN_OVERRIDES+=("+data.num_val_workers=$NUM_WORKERS")
  TRAIN_OVERRIDES+=("+data.num_test_workers=$NUM_WORKERS")
fi
if [ -n "$PRECISION" ]; then
  TRAIN_OVERRIDES+=("trainer.precision=$PRECISION")
elif [ "$ACCELERATOR" = "mps" ]; then
  # Lightning's mixed precision settings are CUDA-oriented in this project.
  # Use full precision on Apple MPS unless the caller explicitly overrides it.
  TRAIN_OVERRIDES+=("trainer.precision=32-true")
fi

if [ -z "$PARALLEL_JOBS" ]; then
  case "$ACCELERATOR" in
    gpu) PARALLEL_JOBS=2 ;;
    *) PARALLEL_JOBS=1 ;;
  esac
fi

# --------- Experiment list ---------
EXPERIMENTS=(
  fair_comparison/sciplex3_cv/linear_l1000
  fair_comparison/sciplex3_cv/latent_l1000
  fair_comparison/sciplex3_cv/decoder_l1000
  fair_comparison/sciplex3_cv/cpa_l1000
  fair_comparison/sciplex3_cv/cpa_noadv_l1000
  fair_comparison/sciplex3_cv/linear_ecfp
  fair_comparison/sciplex3_cv/latent_ecfp
  fair_comparison/sciplex3_cv/decoder_ecfp
  fair_comparison/sciplex3_cv/cpa_ecfp
  fair_comparison/sciplex3_cv/cpa_noadv_ecfp
  fair_comparison/sciplex3_cv/linear_onehot
  fair_comparison/sciplex3_cv/latent_onehot
  fair_comparison/sciplex3_cv/decoder_onehot
  fair_comparison/sciplex3_cv/cpa_onehot
  fair_comparison/sciplex3_cv/cpa_noadv_onehot
)

# --------- Resolve fold list ---------
if [ -z "$FOLDS" ]; then
  FOLD_LIST=()
  for ((k=0; k<N_FOLDS; k++)); do FOLD_LIST+=("$k"); done
else
  IFS=',' read -ra FOLD_LIST <<< "$FOLDS"
fi

# --------- Model filter ---------
extract_model_family() {
  local name="$1"
  local IFS='_'
  local -a parts
  read -ra parts <<< "$name"
  local family=""
  for p in "${parts[@]}"; do
    case "$p" in onehot|ecfp|lpm|l1000) break ;; esac
    [ -z "$family" ] && family="$p" || family="${family}_$p"
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

# --------- Job list (fold-major) ---------
# We run all (model × embedding) jobs for fold k before starting fold k+1.
# Within each fold, the jobs are split round-robin across 2 GPUs and run in
# parallel; the next fold doesn't start until every job in the current fold
# has finished.
JOBS_PER_FOLD=()  # parallel arrays: index = fold position in FOLD_LIST
                  # value = newline-separated "exp|fold" entries
TOTAL_JOBS=0
for fold in "${FOLD_LIST[@]}"; do
  fold_jobs=""
  for exp in "${EXPERIMENTS[@]}"; do
    name=$(basename "$exp")
    matches_model_filter "$name" || continue
    out_dir="$RESULTS_DIR/$name/fold${fold}"
    summary="$out_dir/evaluation/summary.csv"
    if [ "$FORCE" = true ] || [ ! -f "$summary" ]; then
      fold_jobs+="$exp|$fold"$'\n'
      TOTAL_JOBS=$((TOTAL_JOBS + 1))
    else
      echo "Skipping $name fold$fold (results exist)"
    fi
  done
  JOBS_PER_FOLD+=("$fold_jobs")
done

if [ "$TOTAL_JOBS" -eq 0 ]; then
  echo "Nothing to run."
  exit 0
fi

echo "Pool:       sciplex3_lpm_available_perturbations.txt (177 compounds)"
echo "Folds:      ${FOLD_LIST[*]} of $N_FOLDS"
echo "L1000 emb:  $EMB_PATH ($EMB_COLUMN)"
echo "Output:     $RESULTS_DIR"
echo "Jobs:       $TOTAL_JOBS (run fold-major, all models in fold k complete before fold k+1)"
echo "Accelerator: $ACCELERATOR (devices=$DEVICES, parallel jobs=$PARALLEL_JOBS)"
[ -n "$NUM_WORKERS" ] && echo "Workers:    $NUM_WORKERS per dataloader"
[ -n "$MODEL_FILTER" ] && echo "Model filter: $MODEL_FILTER"
echo "---"

run_worker() {
  local worker_id=$1
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
      # Prefer last.ckpt for resume; fallback to most-recent
      if [ -f "$ckpt_dir/last.ckpt" ]; then
        ckpt="$ckpt_dir/last.ckpt"
      else
        ckpt=$(ls -t "$ckpt_dir"/*.ckpt | head -1)
      fi
      ckpt_arg="ckpt_path='$ckpt'"
      echo "[worker $worker_id] Resuming $name fold$fold from $(basename $ckpt)"
    else
      echo "[worker $worker_id] Starting $name fold$fold (fresh)"
    fi

    local extra=()
    case "$name" in
      *_l1000)
        extra+=(
          "data.perturbation_embedding_path=$EMB_ABS"
          "data.perturbation_embedding_column=$EMB_COLUMN"
        )
        ;;
    esac

    # Disable set -e for the per-job command so a failure (OOM, transient bug)
    # doesn't cascade-skip the rest of this worker's queue.
    set +e
    if [ "$ACCELERATOR" = "gpu" ]; then
      CUDA_VISIBLE_DEVICES=$worker_id uv run train \
        experiment="$exp" \
        "hydra.run.dir=$out_dir" \
        "data.splitter.fold=$fold" \
        "data.splitter.n_folds=$N_FOLDS" \
        "${TRAIN_OVERRIDES[@]}" \
        "${extra[@]}" \
        $ckpt_arg
    else
      PYTORCH_ENABLE_MPS_FALLBACK=1 uv run train \
        experiment="$exp" \
        "hydra.run.dir=$out_dir" \
        "data.splitter.fold=$fold" \
        "data.splitter.n_folds=$N_FOLDS" \
        "${TRAIN_OVERRIDES[@]}" \
        "${extra[@]}" \
        $ckpt_arg
    fi
    local rc=$?
    set -e

    if [ $rc -ne 0 ]; then
      echo "[worker $worker_id] Job FAILED (rc=$rc): $name fold$fold (continuing)"
    else
      echo "[worker $worker_id] Finished $name fold$fold"
    fi
  done
}

# Run each fold's jobs to completion before moving on. Within a fold, all
# workers pull from a shared queue (work-stealing): as soon as a worker
# finishes one job, it grabs the next one from the queue. This eliminates
# the idle time we'd otherwise see when one worker happens to draw all the
# slow jobs (e.g. latent_*) and the other finishes its lighter queue early.
for idx in "${!FOLD_LIST[@]}"; do
  fold="${FOLD_LIST[$idx]}"
  fold_jobs_str="${JOBS_PER_FOLD[$idx]}"
  [ -z "$fold_jobs_str" ] && continue

  # Split the newline-separated string back into an array
  fold_jobs=()
  while IFS= read -r line; do
    [ -n "$line" ] && fold_jobs+=("$line")
  done <<< "$fold_jobs_str"

  echo ""
  echo "=== Fold $fold (${#fold_jobs[@]} jobs, $PARALLEL_JOBS workers, work-stealing) ==="
  echo "---"

  # Shared FIFO queue file. Workers atomically pop one job at a time using
  # flock on a sibling lock file, so faster workers automatically pick up
  # extra work while slower ones are still busy.
  queue_file=$(mktemp)
  queue_lock="${queue_file}.lock"
  : > "$queue_lock"
  printf '%s\n' "${fold_jobs[@]}" > "$queue_file"

  for ((worker_id=0; worker_id<PARALLEL_JOBS; worker_id++)); do
    (
      while true; do
        # Atomically pop the first line from the shared queue.
        exec 9>"$queue_lock"
        flock -x 9
        job=$(head -n 1 "$queue_file" 2>/dev/null)
        if [ -n "$job" ]; then
          sed -i '1d' "$queue_file"
        fi
        exec 9>&-
        [ -z "$job" ] && break
        run_worker "$worker_id" "$job"
      done
    ) &
  done
  wait
  rm -f "$queue_file" "$queue_lock"
  echo "=== Fold $fold complete ==="
done

echo "All CV jobs complete. Results in $RESULTS_DIR/"
echo ""
echo "Aggregate + plot:"
echo "  uv run python scripts/aggregate_sciplex3_cv.py --results-dir $RESULTS_DIR"
echo "  uv run python scripts/plot_sciplex3_cv.py --results-dir $RESULTS_DIR"
