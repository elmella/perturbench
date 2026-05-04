"""Plot op3 de-genes CV results: bar charts of mean metrics across folds
with error bars (std across folds), grouped by (model, embedding)."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [
    ("rmse_average", "RMSE (lower better)", False),
    ("rmse_rank_average", "RMSE rank (lower better)", False),
    ("cosine_pca_average", "Cosine PCA (higher better)", True),
    ("cosine_logfc", "Cosine logFC (higher better)", True),
]

MODEL_ORDER = ["linear", "latent", "decoder", "cpa", "cpa_noadv"]
EMBED_ORDER = ["onehot", "ecfp", "lpm"]
EMBED_COLORS = {"onehot": "#9E9E9E", "ecfp": "#1f77b4", "lpm": "#d62728"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/op3_de_cv_comparison",
        help="Directory containing cv_summary.csv (produced by aggregate_cv_metrics.py)",
    )
    parser.add_argument(
        "--title",
        default="OP3 (de-genes) 4-fold CV over unseen perturbations",
        help="Chart title prefix",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_csv = results_dir / "cv_summary.csv"
    out_png = results_dir / "cv_summary.png"
    df = pd.read_csv(summary_csv)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    x_positions = np.arange(len(MODEL_ORDER))
    bar_w = 0.26

    for ax, (metric, ylabel, _higher_better) in zip(axes.flat, METRICS):
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        for j, embed in enumerate(EMBED_ORDER):
            means, stds = [], []
            for model in MODEL_ORDER:
                row = df[(df["model"] == model) & (df["embedding"] == embed)]
                if len(row) == 0:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(row[mean_col].iloc[0])
                    stds.append(row[std_col].iloc[0])
            offset = (j - 1) * bar_w
            ax.bar(
                x_positions + offset,
                means,
                bar_w,
                yerr=stds,
                capsize=3,
                label=embed.upper(),
                color=EMBED_COLORS[embed],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MODEL_ORDER, rotation=15)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0, 0].legend(title="Embedding", loc="upper right")
    fig.suptitle(f"{args.title} — mean ± std across folds", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
