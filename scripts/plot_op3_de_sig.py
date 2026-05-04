"""Plot op3 de-genes single-split (sig) results: bar charts grouped by
(model, embedding). Single value per run since this is a fixed train/test
split (no CV), so no error bars.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("rmse_average", "RMSE (lower better)"),
    ("rmse_rank_average", "RMSE rank (lower better)"),
    ("cosine_pca_average", "Cosine PCA (higher better)"),
    ("cosine_logfc", "Cosine logFC (higher better)"),
]

MODEL_ORDER = ["linear", "latent", "decoder", "cpa", "cpa_noadv"]
# The embedding column covers fixed encodings plus learnable-init variants.
EMBED_ORDER = ["onehot", "ecfp", "lpm", "ecfp_learnable", "lpm_learnable"]
EMBED_COLORS = {
    "onehot": "#9E9E9E",
    "ecfp": "#1f77b4",
    "lpm": "#d62728",
    "ecfp_learnable": "#6baed6",
    "lpm_learnable": "#fcae91",
}
EMBED_LABELS = {
    "onehot": "ONEHOT",
    "ecfp": "ECFP (fixed)",
    "lpm": "LPM (fixed)",
    "ecfp_learnable": "ECFP (learnable)",
    "lpm_learnable": "LPM (learnable)",
}

EMBEDDING_TOKENS = {"onehot", "ecfp", "lpm"}


def parse_experiment_name(exp_name: str) -> tuple[str, str]:
    parts = exp_name.split("_")
    for i, p in enumerate(parts):
        if p in EMBEDDING_TOKENS:
            return "_".join(parts[:i]) or p, "_".join(parts[i:])
    return exp_name, ""


def collect(results_dir: Path, keep_filter=None) -> pd.DataFrame:
    """Walk a results dir and return long-format metric rows.

    keep_filter: optional callable(exp_name) -> bool to include/exclude
    individual experiment directories.
    """
    rows = []
    if not results_dir.exists():
        return pd.DataFrame(rows)
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if keep_filter is not None and not keep_filter(exp_dir.name):
            continue
        summary = exp_dir / "evaluation" / "summary.csv"
        if not summary.exists():
            continue
        df = pd.read_csv(summary)
        if df.shape[1] != 2:
            continue
        metric_col, value_col = df.columns
        model, embedding = parse_experiment_name(exp_dir.name)
        for _, row in df.iterrows():
            rows.append({
                "experiment": exp_dir.name,
                "model": model,
                "embedding": embedding,
                "model_class": value_col,
                "metric": row[metric_col],
                "value": row[value_col],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/op3_de_sig_comparison",
        help="Directory with <experiment>/evaluation/summary.csv files",
    )
    parser.add_argument(
        "--extra-results-dir",
        action="append",
        default=[],
        help="Additional directories to merge (e.g. pick up learnable runs that live elsewhere). Repeatable. "
        "An extra dir contributes an experiment only if the primary dir doesn't already have it.",
    )
    parser.add_argument(
        "--title",
        default="OP3 (de-genes) single-split — 35 test compounds (op3_signatures seed=42)",
    )
    parser.add_argument(
        "--out-name",
        default="sig_summary",
        help="Basename for the output csv and png (default: sig_summary)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    long_df = collect(results_dir)
    primary_experiments = set(long_df["experiment"].unique()) if not long_df.empty else set()

    # Merge extra dirs, but only for experiments not already collected from the primary dir.
    for extra in args.extra_results_dir:
        extra_path = Path(extra)
        extra_df = collect(extra_path, keep_filter=lambda name: name not in primary_experiments)
        if extra_df.empty:
            continue
        long_df = pd.concat([long_df, extra_df], ignore_index=True)
        primary_experiments |= set(extra_df["experiment"].unique())

    if long_df.empty:
        print(f"No evaluation summaries found under {results_dir} (or any extra dirs)")
        return

    out_csv = results_dir / f"{args.out_name}.csv"
    out_png = results_dir / f"{args.out_name}.png"
    wide = long_df.pivot(index="experiment", columns="metric", values="value").reset_index()
    wide[["model", "embedding"]] = wide["experiment"].apply(
        lambda x: pd.Series(parse_experiment_name(x))
    )
    first = ["experiment", "model", "embedding"]
    other = sorted(c for c in wide.columns if c not in first)
    wide = wide[first + other]
    wide.to_csv(out_csv, index=False)

    # Include only embedding types that actually show up in the data (keeps the
    # chart compact when a learnable sweep isn't present).
    present_embeds = set(long_df["embedding"].unique())
    active_embeds = [e for e in EMBED_ORDER if e in present_embeds]
    n_groups = len(active_embeds)
    bar_w = min(0.9 / n_groups, 0.22)
    center_offset = (n_groups - 1) / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    x_positions = np.arange(len(MODEL_ORDER))
    for ax, (metric, ylabel) in zip(axes.flat, METRICS):
        for j, embed in enumerate(active_embeds):
            values = []
            for model in MODEL_ORDER:
                sub = long_df[
                    (long_df["model"] == model)
                    & (long_df["embedding"] == embed)
                    & (long_df["metric"] == metric)
                ]
                values.append(sub["value"].iloc[0] if len(sub) else np.nan)
            offset = (j - center_offset) * bar_w
            ax.bar(
                x_positions + offset,
                values,
                bar_w,
                label=EMBED_LABELS.get(embed, embed.upper()),
                color=EMBED_COLORS.get(embed, "#888888"),
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_ORDER, rotation=15)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0, 0].legend(title="Embedding", loc="upper right", fontsize=9)
    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
