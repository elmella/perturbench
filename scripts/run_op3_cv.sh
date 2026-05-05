#!/usr/bin/env bash
# OP3 4-fold CV over completely unseen perturbations — mirrors the sciplex3 setup.
#
# Compound pool: 138 op3 compounds with new LPM embeddings in
#   all_l1000_25_epochs/op3_emb_all_l1000_20.pkl. Onehot/ECFP/LPM all train
#   and evaluate on the same 138 compounds. ECFP comes from the same .pkl.
# Splitter: unseen_cv_task with stride partitioning (every compound rotates
#   through the test set exactly once across 4 folds).
# Dataset: op3_de_genes.h5ad (DE-genes-filtered op3, matches the existing
#   fair_comparison/op3_de_cv hyperparameter tuning).
#
# Trains 5 architectures × 3 embeddings × 4 folds = 60 runs, fold-major: all
# 15 jobs in fold k complete before fold k+1 starts.
#
# Resume: each run saves last.ckpt every epoch. Re-run the script to resume.
#
# Usage:
#   bash scripts/run_op3_cv.sh                       # all 60 jobs
#   bash scripts/run_op3_cv.sh --emb-path PATH       # different LPM checkpoint
#   bash scripts/run_op3_cv.sh --column l1000        # _l1000 head
#   bash scripts/run_op3_cv.sh --tag mytag           # custom subdir name
#   bash scripts/run_op3_cv.sh --folds 0,1
#   bash scripts/run_op3_cv.sh --models latent,cpa
#   bash scripts/run_op3_cv.sh --max-epochs 200
#   bash scripts/run_op3_cv.sh --early-stopping 50   # enable val carve + early stop
#   bash scripts/run_op3_cv.sh --force
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DEFAULT_EMB_PATH="all_l1000_25_epochs/op3_emb_all_l1000_20.pkl"
EMB_PATH=""
COLUMN_SUFFIX="all"
TAG=""
FORCE=false
N_FOLDS=4
FOLDS=""
MODEL_FILTER=""
MAX_EPOCHS=""
EARLY_STOPPING_PATIENCE=""
# Lower than the framework's 400 to avoid OOMs in the test phase of the larger
# architectures. Override with --eval-chunk-size.
EVAL_CHUNK_SIZE="50"

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --emb-path) EMB_PATH="$2"; shift 2 ;;
    --emb-path=*) EMB_PATH="${arg#*=}"; shift ;;
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
    --eval-chunk-size) EVAL_CHUNK_SIZE="$2"; shift 2 ;;
    --eval-chunk-size=*) EVAL_CHUNK_SIZE="${arg#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[ -z "$EMB_PATH" ] && EMB_PATH="$DEFAULT_EMB_PATH"
case "$COLUMN_SUFFIX" in
  all|l1000) EMB_COLUMN="lpm_style_embeddings_${COLUMN_SUFFIX}" ;;
  lpm_style_embeddings_*) EMB_COLUMN="$COLUMN_SUFFIX" ;;
  *) echo "Unknown --column: $COLUMN_SUFFIX"; exit 1 ;;
esac
[ ! -f "$EMB_PATH" ] && { echo "Embedding file not found: $EMB_PATH"; exit 1; }

if [ -z "$TAG" ]; then
  base=$(basename "$EMB_PATH" .pkl)
  TAG="${base}__${EMB_COLUMN}"
fi
RESULTS_DIR="results/op3_cv/$TAG"
EMB_ABS="$(realpath "$EMB_PATH")"

# --------- Common Hydra overrides ---------
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
TRAIN_OVERRIDES+=("+callbacks.model_checkpoint.save_last=true")
TRAIN_OVERRIDES+=("data.evaluation.chunk_size=$EVAL_CHUNK_SIZE")
[ -n "$MAX_EPOCHS" ] && TRAIN_OVERRIDES+=("trainer.max_epochs=$MAX_EPOCHS")

# --------- Experiment list ---------
EXPERIMENTS=(
  fair_comparison/op3_cv/linear_lpm
  fair_comparison/op3_cv/latent_lpm
  fair_comparison/op3_cv/decoder_lpm
  fair_comparison/op3_cv/cpa_lpm
  fair_comparison/op3_cv/cpa_noadv_lpm
  fair_comparison/op3_cv/linear_ecfp
  fair_comparison/op3_cv/latent_ecfp
  fair_comparison/op3_cv/decoder_ecfp
  fair_comparison/op3_cv/cpa_ecfp
  fair_comparison/op3_cv/cpa_noadv_ecfp
  fair_comparison/op3_cv/linear_onehot
  fair_comparison/op3_cv/latent_onehot
  fair_comparison/op3_cv/decoder_onehot
  fair_comparison/op3_cv/cpa_onehot
  fair_comparison/op3_cv/cpa_noadv_onehot
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
JOBS_PER_FOLD=()
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
  echo "Nothing to run."; exit 0
fi

echo "Pool:       op3_lpm_available_perturbations.txt (138 compounds)"
echo "Folds:      ${FOLD_LIST[*]} of $N_FOLDS"
echo "LPM emb:    $EMB_PATH ($EMB_COLUMN)"
echo "Output:     $RESULTS_DIR"
echo "Jobs:       $TOTAL_JOBS (run fold-major)"
[ -n "$MODEL_FILTER" ] && echo "Model filter: $MODEL_FILTER"
echo "---"

run_gpu() {
  local gpu_id=$1
  shift
  for job in "$@"; do
    local exp="${job%|*}"
    local fold="${job##*|}"
    local name; name=$(basename "$exp")
    local out_dir="$RESULTS_DIR/$name/fold${fold}"
    local ckpt_dir="$out_dir/checkpoints"

    local ckpt_arg=""
    if [ -d "$ckpt_dir" ] && ls "$ckpt_dir"/*.ckpt &>/dev/null; then
      local ckpt
      if [ -f "$ckpt_dir/last.ckpt" ]; then
        ckpt="$ckpt_dir/last.ckpt"
      else
        ckpt=$(ls -t "$ckpt_dir"/*.ckpt | head -1)
      fi
      ckpt_arg="ckpt_path='$ckpt'"
      echo "[GPU $gpu_id] Resuming $name fold$fold from $(basename $ckpt)"
    else
      echo "[GPU $gpu_id] Starting $name fold$fold (fresh)"
    fi

    local extra=()
    case "$name" in
      *_lpm)
        extra+=(
          "data.perturbation_embedding_path=$EMB_ABS"
          "data.perturbation_embedding_column=$EMB_COLUMN"
        )
        ;;
    esac

    set +e
    CUDA_VISIBLE_DEVICES=$gpu_id uv run train \
      experiment="$exp" \
      "hydra.run.dir=$out_dir" \
      "data.splitter.fold=$fold" \
      "data.splitter.n_folds=$N_FOLDS" \
      "${TRAIN_OVERRIDES[@]}" \
      "${extra[@]}" \
      $ckpt_arg
    local rc=$?
    set -e

    if [ $rc -ne 0 ]; then
      echo "[GPU $gpu_id] Job FAILED (rc=$rc): $name fold$fold (continuing)"
    else
      echo "[GPU $gpu_id] Finished $name fold$fold"
    fi
  done
}

# Run fold-major with a work-stealing pool inside each fold. Both GPUs
# share a single queue and each grabs the next job as soon as it's free,
# so a heavy run on one GPU doesn't leave the other one idle.
N_GPUS=2
for idx in "${!FOLD_LIST[@]}"; do
  fold="${FOLD_LIST[$idx]}"
  fold_jobs_str="${JOBS_PER_FOLD[$idx]}"
  [ -z "$fold_jobs_str" ] && continue
  fold_jobs=()
  while IFS= read -r line; do
    [ -n "$line" ] && fold_jobs+=("$line")
  done <<< "$fold_jobs_str"

  echo ""
  echo "=== Fold $fold (${#fold_jobs[@]} jobs, $N_GPUS GPUs, work-stealing) ==="
  echo "---"

  queue_file=$(mktemp)
  queue_lock="${queue_file}.lock"
  : > "$queue_lock"
  printf '%s\n' "${fold_jobs[@]}" > "$queue_file"

  for ((gpu_id=0; gpu_id<N_GPUS; gpu_id++)); do
    (
      while true; do
        exec 9>"$queue_lock"
        flock -x 9
        job=$(head -n 1 "$queue_file" 2>/dev/null)
        if [ -n "$job" ]; then
          sed -i '1d' "$queue_file"
        fi
        exec 9>&-
        [ -z "$job" ] && break
        run_gpu "$gpu_id" "$job"
      done
    ) &
  done
  wait
  rm -f "$queue_file" "$queue_lock"
  echo "=== Fold $fold complete ==="
done

echo "All CV jobs complete. Results in $RESULTS_DIR/"
echo ""
echo "Plot:"
echo "  uv run python scripts/plot_sciplex3_cv.py --results-dir $RESULTS_DIR"
