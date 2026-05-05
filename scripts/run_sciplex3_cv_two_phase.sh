#!/usr/bin/env bash
# Two-phase sciplex3 CV training. Runs the standard runner twice:
#   Phase 1: train every (model × fold) for the selected embedding(s) up to
#            100 epochs. Output goes to results/sciplex3_cv/max100/.
#            Existing summary.csv files in max100 are skipped, so this fills
#            in gaps from prior partial runs without restarting anything.
#   Phase 2: continue training from each max100 checkpoint up to 500 epochs,
#            then evaluate. Output goes to results/sciplex3_cv/max500/ so
#            the 100-epoch results are preserved.
#
# Designed for distributed work: run this on multiple machines with different
# --embedding values so each machine handles one embedding type. All machines
# write to the same shared results/sciplex3_cv/max100 and max500 directories.
#
# Usage:
#   bash scripts/run_sciplex3_cv_two_phase.sh --embedding lpm
#   bash scripts/run_sciplex3_cv_two_phase.sh --embedding ecfp
#   bash scripts/run_sciplex3_cv_two_phase.sh --embedding onehot
#   bash scripts/run_sciplex3_cv_two_phase.sh --embedding lpm,ecfp     # multi
#   bash scripts/run_sciplex3_cv_two_phase.sh --embedding lpm --skip-phase-1
#   bash scripts/run_sciplex3_cv_two_phase.sh --embedding lpm --skip-phase-2
#
# Any other flags accepted by run_sciplex3_cv.sh (--folds, --models, --epoch,
# --column, --emb-path, --accelerator, --parallel-jobs, --num-workers,
# --eval-chunk-size, --early-stopping, etc.) are passed through to both phases.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EMBEDDING=""
SKIP_PHASE_1=false
SKIP_PHASE_2=false
PHASE_1_TAG="max100"
PHASE_2_TAG="max500"
PHASE_1_EPOCHS=100
PHASE_2_EPOCHS=500
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --embedding|--embeddings) EMBEDDING="$2"; shift 2 ;;
    --embedding=*|--embeddings=*) EMBEDDING="${arg#*=}"; shift ;;
    --skip-phase-1) SKIP_PHASE_1=true; shift ;;
    --skip-phase-2) SKIP_PHASE_2=true; shift ;;
    --phase-1-tag) PHASE_1_TAG="$2"; shift 2 ;;
    --phase-1-tag=*) PHASE_1_TAG="${arg#*=}"; shift ;;
    --phase-2-tag) PHASE_2_TAG="$2"; shift 2 ;;
    --phase-2-tag=*) PHASE_2_TAG="${arg#*=}"; shift ;;
    --phase-1-epochs) PHASE_1_EPOCHS="$2"; shift 2 ;;
    --phase-1-epochs=*) PHASE_1_EPOCHS="${arg#*=}"; shift ;;
    --phase-2-epochs) PHASE_2_EPOCHS="$2"; shift 2 ;;
    --phase-2-epochs=*) PHASE_2_EPOCHS="${arg#*=}"; shift ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [ -z "$EMBEDDING" ]; then
  echo "Required: --embedding <lpm|ecfp|onehot> (comma-separated for multiple)"
  echo ""
  echo "Other flags get passed through to scripts/run_sciplex3_cv.sh."
  echo "Run-once flags this wrapper handles:"
  echo "  --skip-phase-1            run Phase 2 only"
  echo "  --skip-phase-2            run Phase 1 only"
  echo "  --phase-1-tag NAME        Phase 1 output dir tag (default: max100)"
  echo "  --phase-2-tag NAME        Phase 2 output dir tag (default: max500)"
  echo "  --phase-1-epochs N        Phase 1 max_epochs (default: 100)"
  echo "  --phase-2-epochs N        Phase 2 max_epochs (default: 500)"
  exit 1
fi

if [ "$SKIP_PHASE_1" != true ]; then
  echo ""
  echo "######################################################################"
  echo "### Phase 1: train to $PHASE_1_EPOCHS epochs (--tag $PHASE_1_TAG, --embedding $EMBEDDING)"
  echo "###          Existing summary.csv files in this tag are skipped."
  echo "######################################################################"
  bash "$SCRIPT_DIR/run_sciplex3_cv.sh" \
    --max-epochs "$PHASE_1_EPOCHS" \
    --tag "$PHASE_1_TAG" \
    --embeddings "$EMBEDDING" \
    "${PASSTHROUGH[@]}"
fi

if [ "$SKIP_PHASE_2" != true ]; then
  echo ""
  echo "######################################################################"
  echo "### Phase 2: continue from $PHASE_1_TAG ckpts to $PHASE_2_EPOCHS epochs"
  echo "###          --tag $PHASE_2_TAG (separate dir; $PHASE_1_TAG is preserved)"
  echo "######################################################################"
  bash "$SCRIPT_DIR/run_sciplex3_cv.sh" \
    --max-epochs "$PHASE_2_EPOCHS" \
    --tag "$PHASE_2_TAG" \
    --starting-ckpt-tag "$PHASE_1_TAG" \
    --embeddings "$EMBEDDING" \
    "${PASSTHROUGH[@]}"
fi

echo ""
echo "Two-phase run complete for embedding: $EMBEDDING"
echo "  Phase 1 (100 epochs): results/sciplex3_cv/$PHASE_1_TAG/"
echo "  Phase 2 (500 epochs): results/sciplex3_cv/$PHASE_2_TAG/"
