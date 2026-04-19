"""Aggregate per-fold evaluation summaries into a CV results table.

Expects results layout:
    <results_dir>/<experiment>/fold<k>/evaluation/summary.csv

Each summary.csv has two columns (metric, <ModelClass>). This script collects
all metrics across folds for each experiment, computes mean / std, and writes:

  <results_dir>/cv_per_fold.csv   -- long-format with a row per (exp, fold, metric)
  <results_dir>/cv_summary.csv    -- wide-format with mean±std across folds
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FOLD_RE = re.compile(r"fold(\d+)$")


def parse_experiment_name(exp_name: str) -> tuple[str, str]:
    """Split an experiment name into (model, embedding).

    Handles patterns like:
        linear_onehot              -> ("linear",     "onehot")
        latent_ecfp_learnable      -> ("latent",     "ecfp_learnable")
        cpa_noadv_lpm              -> ("cpa_noadv",  "lpm")
        cpa_noadv_ecfp_learnable   -> ("cpa_noadv",  "ecfp_learnable")
    """
    embed_tokens = {"onehot", "ecfp", "lpm"}
    parts = exp_name.split("_")
    for i, part in enumerate(parts):
        if part in embed_tokens:
            model = "_".join(parts[:i]) or part
            embedding = "_".join(parts[i:])
            return model, embedding
    return exp_name, ""


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for fold_dir in sorted(exp_dir.iterdir()):
            if not fold_dir.is_dir():
                continue
            m = FOLD_RE.match(fold_dir.name)
            if not m:
                continue
            fold = int(m.group(1))
            summary = fold_dir / "evaluation" / "summary.csv"
            if not summary.exists():
                continue
            df = pd.read_csv(summary)
            if df.shape[1] != 2:
                continue
            metric_col, value_col = df.columns
            model_class = value_col
            for _, row in df.iterrows():
                rows.append(
                    {
                        "experiment": exp_dir.name,
                        "fold": fold,
                        "metric": row[metric_col],
                        "value": row[value_col],
                        "model_class": model_class,
                    }
                )
    return pd.DataFrame(rows)


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df

    meta = long_df[["experiment"]].drop_duplicates().copy()
    meta[["model", "embedding"]] = meta["experiment"].apply(
        lambda x: pd.Series(parse_experiment_name(x))
    )

    summary = (
        long_df.groupby(["experiment", "metric"])["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary = summary.merge(meta, on="experiment", how="left")

    wide_mean = summary.pivot(index="experiment", columns="metric", values="mean")
    wide_std = summary.pivot(index="experiment", columns="metric", values="std")
    wide_n = summary.pivot(index="experiment", columns="metric", values="n")
    wide_mean = wide_mean.add_suffix("_mean")
    wide_std = wide_std.add_suffix("_std")
    wide_n = wide_n.add_suffix("_n")
    wide = wide_mean.join(wide_std).join(wide_n).reset_index()
    wide = wide.merge(meta, on="experiment", how="left")

    first_cols = ["experiment", "model", "embedding"]
    other = [c for c in wide.columns if c not in first_cols]
    return wide[first_cols + sorted(other)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing <experiment>/fold<k>/evaluation/summary.csv",
    )
    parser.add_argument(
        "--per-fold-out",
        default=None,
        help="Output path for the per-fold long-format CSV "
        "(default: <results-dir>/cv_per_fold.csv)",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output path for the cross-fold summary CSV "
        "(default: <results-dir>/cv_summary.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    long_df = collect(results_dir)
    if long_df.empty:
        print(f"No evaluation summaries found under {results_dir}")
        return

    per_fold_out = Path(args.per_fold_out or (results_dir / "cv_per_fold.csv"))
    summary_out = Path(args.summary_out or (results_dir / "cv_summary.csv"))

    long_df.to_csv(per_fold_out, index=False)
    summary = summarize(long_df)
    summary.to_csv(summary_out, index=False)

    # Small human-readable summary on stdout for the core metrics.
    core_metrics = [
        "rmse_average",
        "rmse_rank_average",
        "cosine_pca_average",
        "cosine_logfc",
        "r2_score_scores",
    ]
    print(f"\nWrote {per_fold_out}")
    print(f"Wrote {summary_out}")
    print(f"\nAcross-fold mean ± std (n folds reported) for {len(summary)} experiments:\n")
    for _, row in summary.sort_values(["model", "embedding"]).iterrows():
        parts = [f"{row['experiment']:<40s}"]
        for m in core_metrics:
            mean = row.get(f"{m}_mean", np.nan)
            std = row.get(f"{m}_std", np.nan)
            n = row.get(f"{m}_n", np.nan)
            if pd.notna(mean):
                parts.append(f"{m}={mean:.4f}±{std:.4f} (n={int(n)})")
        print(" | ".join(parts))


if __name__ == "__main__":
    main()
