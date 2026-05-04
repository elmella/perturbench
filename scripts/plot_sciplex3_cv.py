"""Plot sciplex3 CV / single-split results.

Produces two layers of output, both with identical sub-directory structure
(so they're easy to diff/compare across runs):

  <results_dir>/<experiment>/fold<k>/plots/<metric>.png   # per-fold bar charts
                                       /metrics.csv       # tidied per-fold metrics
  <results_dir>/_combined/plots/<metric>.png              # mean ± std across folds
  <results_dir>/_combined/cv_summary.csv                  # wide summary
  <results_dir>/_combined/cv_per_fold.csv                 # long per-fold

Per-fold plot titles do NOT mention the fold number (the directory does).
The combined plot titles do not mention any specific fold either; they say
"mean ± std across folds".

Works for the single-split runner too: a "split" with a single non-fold
subdirectory (e.g. fold0) produces the same files but the combined plot
just has zero-width error bars.

Usage:
  uv run python scripts/plot_sciplex3_cv.py --results-dir results/sciplex3_cv/<tag>
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Each entry is (metric_name, axis_label, higher_is_better)
METRICS = [
    ("rmse_average", "RMSE (lower is better)", False),
    ("rmse_rank_average", "RMSE rank (lower is better)", False),
    ("cosine_logfc", "Cosine logFC (higher is better)", True),
    ("cosine_rank_logfc", "Cosine logFC rank (lower is better)", False),
]

MODEL_ORDER = ["linear", "latent", "decoder", "cpa", "cpa_noadv"]
# Embedding categories (legend labels). l1000 is not its own category — it's
# an LPM model trained on the L1000 gene panel. The specific LPM checkpoint
# is encoded in the run tag (the parent directory).
EMBED_ORDER = ["onehot", "ecfp", "lpm"]
EMBED_COLORS = {
    "onehot": "#9E9E9E",
    "ecfp": "#1f77b4",
    "lpm": "#d62728",
}

# Tokens that may appear in an experiment filename and indicate the embedding.
# "l1000" → "lpm" (different gene panel, same embedding type).
EMBED_TOKEN_NORMALIZE = {
    "onehot": "onehot",
    "ecfp": "ecfp",
    "lpm": "lpm",
    "l1000": "lpm",
}

FOLD_RE = re.compile(r"^fold(\d+)$")


def parse_experiment_name(exp_name: str) -> tuple[str, str]:
    """Return (model, embedding_category). 'embedding_category' is normalized
    to one of EMBED_ORDER even when the filename uses 'l1000' or trailing
    qualifiers like 'ecfp_learnable'."""
    parts = exp_name.split("_")
    for i, part in enumerate(parts):
        if part in EMBED_TOKEN_NORMALIZE:
            model = "_".join(parts[:i]) or part
            return model, EMBED_TOKEN_NORMALIZE[part]
    return exp_name, ""


def load_summary(summary_path: Path) -> pd.Series | None:
    """Read a per-run summary.csv and return a Series indexed by metric name."""
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if df.shape[1] != 2:
        return None
    metric_col, value_col = df.columns
    return pd.Series(df[value_col].values, index=df[metric_col].astype(str).values)


def collect_runs(results_dir: Path) -> pd.DataFrame:
    """Walk results into a long DataFrame. Supports both layouts:
      <exp>/fold<k>/evaluation/summary.csv   # CV runner
      <exp>/evaluation/summary.csv           # single-split runner (treated as 'single')
    """
    rows = []
    for exp_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if exp_dir.name.startswith("_"):
            continue
        model, embedding = parse_experiment_name(exp_dir.name)

        # Single-split layout: <exp>/evaluation/summary.csv
        single_summary = exp_dir / "evaluation" / "summary.csv"
        if single_summary.exists():
            metrics = load_summary(single_summary)
            if metrics is not None:
                for metric_name, value in metrics.items():
                    rows.append({
                        "experiment": exp_dir.name,
                        "model": model,
                        "embedding": embedding,
                        "fold": "single",
                        "metric": metric_name,
                        "value": float(value),
                    })
            continue  # don't also descend into evaluation/ as a fold dir

        # CV layout: <exp>/fold<k>/evaluation/summary.csv
        for split_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            m = FOLD_RE.match(split_dir.name)
            fold = int(m.group(1)) if m else split_dir.name
            metrics = load_summary(split_dir / "evaluation" / "summary.csv")
            if metrics is None:
                continue
            for metric_name, value in metrics.items():
                rows.append({
                    "experiment": exp_dir.name,
                    "model": model,
                    "embedding": embedding,
                    "fold": fold,
                    "metric": metric_name,
                    "value": float(value),
                })
    return pd.DataFrame(rows)


def plot_grouped_bars(
    ax,
    metric: str,
    means_by_embed: dict[str, list[float]],
    stds_by_embed: dict[str, list[float]] | None,
    models: list[str],
    embed_order: list[str],
):
    x = np.arange(len(models))
    n_embed = len(embed_order)
    bar_w = 0.8 / max(n_embed, 1)
    for j, embed in enumerate(embed_order):
        means = means_by_embed.get(embed, [np.nan] * len(models))
        stds = (stds_by_embed or {}).get(embed) if stds_by_embed else None
        offset = (j - (n_embed - 1) / 2.0) * bar_w
        ax.bar(
            x + offset,
            means,
            bar_w,
            yerr=stds,
            capsize=3,
            label=embed.upper(),
            color=EMBED_COLORS.get(embed, None),
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def write_per_fold_plots(long_df: pd.DataFrame, results_dir: Path) -> None:
    """One plot per (experiment, fold) — but only models from the SAME fold are
    grouped together. Each fold's directory ends up with one PNG per metric and
    a metrics.csv. Plot titles only name the metric."""
    available_models = [m for m in MODEL_ORDER if m in long_df["model"].unique()]
    available_embeds = [e for e in EMBED_ORDER if e in long_df["embedding"].unique()]
    if not available_models or not available_embeds:
        print("No (model, embedding) combinations found.")
        return

    for fold, fold_df in long_df.groupby("fold"):
        for metric, ylabel, higher_better in METRICS:
            sub = fold_df[fold_df["metric"] == metric]
            if sub.empty:
                continue
            means = {
                embed: [
                    sub.loc[
                        (sub["model"] == m) & (sub["embedding"] == embed), "value"
                    ].mean()
                    if not sub.loc[
                        (sub["model"] == m) & (sub["embedding"] == embed)
                    ].empty
                    else np.nan
                    for m in available_models
                ]
                for embed in available_embeds
            }
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_grouped_bars(
                ax, metric, means, stds_by_embed=None,
                models=available_models, embed_order=available_embeds,
            )
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend(title="Embedding", loc="best")
            fig.tight_layout()

            # Write PNG into each (exp, fold)/plots directory
            for exp_name in sub["experiment"].unique():
                model, _ = parse_experiment_name(exp_name)
                exp_fold_dir = results_dir / exp_name / f"fold{fold}" if isinstance(fold, int) else results_dir / exp_name / str(fold)
                plot_dir = exp_fold_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                # Filter to just this experiment's row in the bar (still keep
                # the cross-experiment context: same chart, all models/embeds).
                fig.savefig(plot_dir / f"{metric}.png", dpi=140)

            plt.close(fig)

        # Write a per-fold tidy CSV in each (exp, fold) dir
        for exp_name, exp_fold_df in fold_df.groupby("experiment"):
            split_dir_name = f"fold{fold}" if isinstance(fold, int) else str(fold)
            exp_fold_dir = results_dir / exp_name / split_dir_name
            tidy = exp_fold_df[["metric", "value"]].copy()
            tidy.to_csv(exp_fold_dir / "metrics.csv", index=False)


def write_combined(long_df: pd.DataFrame, results_dir: Path) -> None:
    combined = results_dir / "_combined"
    combined.mkdir(parents=True, exist_ok=True)
    plot_dir = combined / "plots"
    plot_dir.mkdir(exist_ok=True)

    # CSV outputs: long-format (per fold) and wide-format (mean ± std)
    long_df.to_csv(combined / "cv_per_fold.csv", index=False)
    grouped = (
        long_df.groupby(["experiment", "model", "embedding", "metric"])["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    wide_mean = grouped.pivot(index=["experiment", "model", "embedding"], columns="metric", values="mean").add_suffix("_mean")
    wide_std = grouped.pivot(index=["experiment", "model", "embedding"], columns="metric", values="std").add_suffix("_std")
    wide_n = grouped.pivot(index=["experiment", "model", "embedding"], columns="metric", values="n").add_suffix("_n")
    wide = wide_mean.join(wide_std).join(wide_n).reset_index()
    wide.to_csv(combined / "cv_summary.csv", index=False)

    available_models = [m for m in MODEL_ORDER if m in long_df["model"].unique()]
    available_embeds = [e for e in EMBED_ORDER if e in long_df["embedding"].unique()]

    for metric, ylabel, higher_better in METRICS:
        sub = grouped[grouped["metric"] == metric]
        if sub.empty:
            continue
        means = {
            embed: [
                sub.loc[(sub["model"] == m) & (sub["embedding"] == embed), "mean"].iloc[0]
                if not sub.loc[(sub["model"] == m) & (sub["embedding"] == embed)].empty
                else np.nan
                for m in available_models
            ]
            for embed in available_embeds
        }
        stds = {
            embed: [
                sub.loc[(sub["model"] == m) & (sub["embedding"] == embed), "std"].iloc[0]
                if not sub.loc[(sub["model"] == m) & (sub["embedding"] == embed)].empty
                else 0.0
                for m in available_models
            ]
            for embed in available_embeds
        }
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_grouped_bars(ax, metric, means, stds, available_models, available_embeds)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\nmean ± std across folds")
        ax.legend(title="Embedding", loc="best")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}.png", dpi=140)
        plt.close(fig)

    print(f"Wrote {combined / 'cv_per_fold.csv'}")
    print(f"Wrote {combined / 'cv_summary.csv'}")
    print(f"Wrote {len(METRICS)} combined plots to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Top-level CV results directory")
    parser.add_argument(
        "--skip-per-fold",
        action="store_true",
        help="Only write the _combined/ output (skip per-fold plots and CSVs)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Not found: {results_dir}")

    long_df = collect_runs(results_dir)
    if long_df.empty:
        raise SystemExit(f"No evaluation summaries under {results_dir}")

    print(f"Loaded {len(long_df)} (run, metric) rows from {results_dir}")
    print(f"Experiments: {sorted(long_df['experiment'].unique())}")
    print(f"Folds:       {sorted(long_df['fold'].unique())}")

    if not args.skip_per_fold:
        write_per_fold_plots(long_df, results_dir)
    write_combined(long_df, results_dir)


if __name__ == "__main__":
    main()
