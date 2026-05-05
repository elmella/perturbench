#!/usr/bin/env bash
# Aggregate one Sciplex3 CV embedding's evaluated fold metrics.
#
# Examples:
#   bash scripts/aggregate_sciplex3_embedding_results.sh --embedding ecfp --input-folder results/max100
#   bash scripts/aggregate_sciplex3_embedding_results.sh --embedding onehot --input-folder max100
#   bash scripts/aggregate_sciplex3_embedding_results.sh --embedding lpm --input-folder max500
#
# Outputs, by default under summary_tables/sciplex3_cv/:
#   sciplex3_cv_<folder>_<embedding>_per_fold.csv
#   sciplex3_cv_<folder>_<embedding>_summary.csv
#   sciplex3_cv_<folder>_<embedding>_missing.csv
set -euo pipefail

EMBEDDING=""
INPUT_FOLDER=""
OUTPUT_DIR="summary_tables/sciplex3_cv"
N_FOLDS=4

while [ $# -gt 0 ]; do
  case "$1" in
    --embedding)
      EMBEDDING="$2"
      shift 2
      ;;
    --embedding=*)
      EMBEDDING="${1#*=}"
      shift
      ;;
    --input-folder|--input-dir)
      INPUT_FOLDER="$2"
      shift 2
      ;;
    --input-folder=*|--input-dir=*)
      INPUT_FOLDER="${1#*=}"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --folds)
      N_FOLDS="$2"
      shift 2
      ;;
    --folds=*)
      N_FOLDS="${1#*=}"
      shift
      ;;
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$EMBEDDING" ] || [ -z "$INPUT_FOLDER" ]; then
  echo "Usage: $0 --embedding <lpm|l1000|ecfp|onehot> --input-folder <max100|max500|path>" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON=(python)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
else
  echo "Could not find python3, python, or uv." >&2
  exit 1
fi

"${PYTHON[@]}" - "$EMBEDDING" "$INPUT_FOLDER" "$OUTPUT_DIR" "$N_FOLDS" <<'PY'
import sys
from pathlib import Path

import pandas as pd


embedding_arg, input_arg, output_arg, n_folds_arg = sys.argv[1:5]
n_folds = int(n_folds_arg)

embedding = embedding_arg.strip().lower()
embedding_token = {"lpm": "l1000", "l1000": "l1000", "ecfp": "ecfp", "onehot": "onehot"}.get(
    embedding
)
if embedding_token is None:
    raise SystemExit("Unknown embedding. Expected one of: lpm, l1000, ecfp, onehot")

input_path = Path(input_arg)
candidates = []
if input_path.exists():
    candidates.append(input_path)
if not input_path.is_absolute():
    candidates.extend([Path("results") / input_arg, Path("results/sciplex3_cv") / input_arg])

seen = set()
resolved = []
for candidate in candidates:
    try:
        key = candidate.resolve()
    except FileNotFoundError:
        key = candidate
    if candidate.exists() and key not in seen:
        resolved.append(candidate)
        seen.add(key)

if not resolved:
    raise SystemExit(
        f"Input folder not found: {input_arg}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )

if len(resolved) > 1:
    print(
        "Multiple matching input folders found; using the first. "
        f"Pass an explicit path to choose another: {', '.join(str(p) for p in resolved)}",
        file=sys.stderr,
    )
base_dir = resolved[0]

models = ["linear", "latent", "decoder", "cpa", "cpa_noadv"]
folds = list(range(n_folds))
folder_label = base_dir.name
output_dir = Path(output_arg)
safe_embedding = "lpm" if embedding in {"lpm", "l1000"} else embedding_token
prefix = output_dir / f"sciplex3_cv_{folder_label}_{safe_embedding}"

per_fold_rows = []
missing_rows = []

for model in models:
    experiment = f"{model}_{embedding_token}"
    for fold in folds:
        fold_dir = base_dir / experiment / f"fold{fold}"
        summary_path = fold_dir / "evaluation" / "summary.csv"
        ckpt_dir = fold_dir / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
        last_ckpt = ckpt_dir / "last.ckpt"

        if summary_path.exists():
            status = "done"
            try:
                df = pd.read_csv(summary_path)
            except Exception as exc:
                status = "bad_summary"
                missing_rows.append(
                    {
                        "input_folder": str(base_dir),
                        "embedding": safe_embedding,
                        "model": model,
                        "experiment": experiment,
                        "fold": fold,
                        "status": status,
                        "summary_path": str(summary_path),
                        "checkpoint_path": str(last_ckpt) if last_ckpt.exists() else "",
                        "note": str(exc),
                    }
                )
                continue

            if df.shape[1] < 2 or "metric" not in df.columns:
                status = "bad_summary"
                missing_rows.append(
                    {
                        "input_folder": str(base_dir),
                        "embedding": safe_embedding,
                        "model": model,
                        "experiment": experiment,
                        "fold": fold,
                        "status": status,
                        "summary_path": str(summary_path),
                        "checkpoint_path": str(last_ckpt) if last_ckpt.exists() else "",
                        "note": "summary.csv must contain metric plus one value column",
                    }
                )
                continue

            value_col = [c for c in df.columns if c != "metric"][0]
            for _, row in df.iterrows():
                per_fold_rows.append(
                    {
                        "input_folder": str(base_dir),
                        "embedding": safe_embedding,
                        "model": model,
                        "experiment": experiment,
                        "fold": fold,
                        "metric": row["metric"],
                        "value": row[value_col],
                        "model_class": value_col,
                        "summary_path": str(summary_path),
                    }
                )
        elif last_ckpt.exists() or ckpts:
            status = "checkpoint_only"
        elif fold_dir.exists():
            status = "directory_only"
        else:
            status = "missing"

        if status != "done":
            checkpoint_path = ""
            if last_ckpt.exists():
                checkpoint_path = str(last_ckpt)
            elif ckpts:
                checkpoint_path = str(ckpts[-1])
            missing_rows.append(
                {
                    "input_folder": str(base_dir),
                    "embedding": safe_embedding,
                    "model": model,
                    "experiment": experiment,
                    "fold": fold,
                    "status": status,
                    "summary_path": str(summary_path),
                    "checkpoint_path": checkpoint_path,
                    "note": "",
                }
            )

per_fold = pd.DataFrame(per_fold_rows)
missing = pd.DataFrame(missing_rows)

per_fold_out = prefix.with_name(prefix.name + "_per_fold.csv")
summary_out = prefix.with_name(prefix.name + "_summary.csv")
missing_out = prefix.with_name(prefix.name + "_missing.csv")

per_fold.to_csv(per_fold_out, index=False)
missing.to_csv(missing_out, index=False)

if per_fold.empty:
    summary = pd.DataFrame(
        columns=[
            "input_folder",
            "embedding",
            "model",
            "experiment",
            "completed_folds",
            "missing_folds",
            "missing_count",
        ]
    )
else:
    grouped = (
        per_fold.groupby(["input_folder", "embedding", "model", "experiment", "metric"])[
            "value"
        ]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    wide = grouped.pivot(
        index=["input_folder", "embedding", "model", "experiment"],
        columns="metric",
        values=["mean", "std", "n"],
    )
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    summary = wide.reset_index()

    completed = (
        per_fold[["input_folder", "embedding", "model", "experiment", "fold"]]
        .drop_duplicates()
        .groupby(["input_folder", "embedding", "model", "experiment"])["fold"]
        .apply(lambda s: ",".join(str(x) for x in sorted(s)))
        .reset_index(name="completed_folds")
    )
    missing_folds = (
        missing.groupby(["input_folder", "embedding", "model", "experiment"])["fold"]
        .apply(lambda s: ",".join(str(x) for x in sorted(s)))
        .reset_index(name="missing_folds")
        if not missing.empty
        else pd.DataFrame(
            columns=["input_folder", "embedding", "model", "experiment", "missing_folds"]
        )
    )
    missing_counts = (
        missing.groupby(["input_folder", "embedding", "model", "experiment"])
        .size()
        .reset_index(name="missing_count")
        if not missing.empty
        else pd.DataFrame(
            columns=["input_folder", "embedding", "model", "experiment", "missing_count"]
        )
    )
    summary = summary.merge(
        completed, on=["input_folder", "embedding", "model", "experiment"], how="left"
    )
    summary = summary.merge(
        missing_folds,
        on=["input_folder", "embedding", "model", "experiment"],
        how="left",
    )
    summary = summary.merge(
        missing_counts,
        on=["input_folder", "embedding", "model", "experiment"],
        how="left",
    )
    summary["missing_folds"] = summary["missing_folds"].fillna("")
    summary["missing_count"] = summary["missing_count"].fillna(0).astype(int)

    ordered = [
        "input_folder",
        "embedding",
        "model",
        "experiment",
        "completed_folds",
        "missing_folds",
        "missing_count",
    ]
    summary = summary[ordered + sorted(c for c in summary.columns if c not in ordered)]

summary.to_csv(summary_out, index=False)

done_count = len(per_fold[["model", "fold"]].drop_duplicates()) if not per_fold.empty else 0
expected = len(models) * n_folds
print(f"Input:       {base_dir}")
print(f"Embedding:   {safe_embedding} ({embedding_token})")
print(f"Completed:   {done_count}/{expected} model-fold evaluations")
print(f"Missing:     {expected - done_count}/{expected} model-fold evaluations")
print(f"Wrote:       {per_fold_out}")
print(f"Wrote:       {summary_out}")
print(f"Wrote:       {missing_out}")
if not missing.empty:
    print("\nMissing or partial:")
    for row in missing.sort_values(["model", "fold"]).itertuples(index=False):
        print(f"  {row.experiment} fold{row.fold}: {row.status}")
PY
