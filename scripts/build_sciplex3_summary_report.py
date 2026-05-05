"""Build comparison tables and plots from Sciplex3 CV summary CSVs.

Input is the set of files produced by aggregate_sciplex3_embedding_results.sh:
    <summary-dir>/sciplex3_cv_<tag>_<embedding>_summary.csv
    <summary-dir>/sciplex3_cv_<tag>_<embedding>_per_fold.csv

Outputs are written to:
    <summary-dir>/report/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "rmse_average": "lower",
    "cosine_pca_average": "higher",
    "cosine_logfc": "higher",
    "r2_score_scores": "higher",
    "mmd_pca": "lower",
    "top_k_recall_scores": "higher",
}
DISPLAY_METRICS = [
    "rmse_average",
    "cosine_pca_average",
    "cosine_logfc",
    "mmd_pca",
    "r2_score_scores",
    "top_k_recall_scores",
]
MODEL_ORDER = ["linear", "latent", "decoder", "cpa", "cpa_noadv"]
EMBEDDING_ORDER = ["lpm", "ecfp", "onehot"]
EVAL_CHECKPOINT_ORDER = ["final", "best_train_loss", "summary", "legacy"]
EMBEDDING_COLORS = {
    "lpm": "#2E86AB",
    "ecfp": "#D1495B",
    "onehot": "#4D9078",
}


def read_csvs(summary_dir: Path, suffix: str) -> pd.DataFrame:
    frames = []
    for path in sorted(summary_dir.glob(f"sciplex3_cv_*_{suffix}.csv")):
        if path.stat().st_size <= 1:
            continue
        df = pd.read_csv(path)
        if not df.empty:
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ordered_category(series: pd.Series, order: list[str]) -> pd.Categorical:
    return pd.Categorical(series, categories=order, ordered=True)


def fmt_mean_std(row: pd.Series, metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    n = row.get(f"{metric}_n")
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.4g} (n={int(n)})"
    return f"{mean:.4g} +/- {std:.3g} (n={int(n)})"


def build_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        out = {
            "embedding": row["embedding"],
            "model": row["model"],
            "eval_checkpoint": row.get("eval_checkpoint", "legacy"),
            "completed_folds": row.get("completed_folds", ""),
            "missing_folds": row.get("missing_folds", ""),
            "missing_count": int(row.get("missing_count", 0)),
        }
        for metric in DISPLAY_METRICS:
            out[metric] = fmt_mean_std(row, metric)
        rows.append(out)
    compact = pd.DataFrame(rows)
    compact["embedding"] = ordered_category(compact["embedding"], EMBEDDING_ORDER)
    compact["model"] = ordered_category(compact["model"], MODEL_ORDER)
    compact["eval_checkpoint"] = ordered_category(
        compact["eval_checkpoint"], EVAL_CHECKPOINT_ORDER
    )
    return compact.sort_values(["model", "embedding", "eval_checkpoint"]).reset_index(drop=True)


def build_best_by_metric(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    complete = summary[summary["missing_count"].fillna(0).eq(0)].copy()
    for metric, direction in METRICS.items():
        col = f"{metric}_mean"
        if col not in complete.columns:
            continue
        values = complete.dropna(subset=[col])
        if values.empty:
            continue
        idx = values[col].idxmin() if direction == "lower" else values[col].idxmax()
        row = values.loc[idx]
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "embedding": row["embedding"],
                "model": row["model"],
                "eval_checkpoint": row.get("eval_checkpoint", "legacy"),
                "mean": row[col],
                "std": row.get(f"{metric}_std"),
                "n": int(row.get(f"{metric}_n", 0)),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as GitHub-flavored Markdown without tabulate."""
    if df.empty:
        return "_No rows._\n"
    rendered = df.astype(str).replace({"nan": "", "NaT": ""})
    headers = list(rendered.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in rendered.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def plot_metric_bars(summary: pd.DataFrame, report_dir: Path, metric: str) -> None:
    col = f"{metric}_mean"
    err_col = f"{metric}_std"
    if col not in summary.columns:
        return

    data = summary.copy()
    data["model"] = ordered_category(data["model"], MODEL_ORDER)
    data["embedding"] = ordered_category(data["embedding"], EMBEDDING_ORDER)
    data = data.sort_values(["model", "embedding"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.24
    x = range(len(MODEL_ORDER))
    offsets = {
        "lpm": -width,
        "ecfp": 0,
        "onehot": width,
    }
    for embedding in EMBEDDING_ORDER:
        subset = data[data["embedding"].eq(embedding)].set_index("model")
        means = [subset[col].get(model, float("nan")) for model in MODEL_ORDER]
        errs = [subset[err_col].get(model, 0) for model in MODEL_ORDER]
        xpos = [i + offsets[embedding] for i in x]
        ax.bar(
            xpos,
            means,
            width=width,
            yerr=errs,
            capsize=3,
            label=embedding,
            color=EMBEDDING_COLORS[embedding],
            alpha=0.88,
        )

    direction = METRICS.get(metric, "")
    ax.set_title(f"{metric} by model and embedding ({direction} is better)")
    ax.set_ylabel(metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="embedding")
    fig.tight_layout()
    fig.savefig(report_dir / f"{metric}_by_model_embedding.png", dpi=180)
    plt.close(fig)


def plot_rmse_vs_cosine(summary: pd.DataFrame, report_dir: Path) -> None:
    if "rmse_average_mean" not in summary.columns or "cosine_pca_average_mean" not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for embedding in EMBEDDING_ORDER:
        subset = summary[summary["embedding"].eq(embedding)]
        ax.scatter(
            subset["rmse_average_mean"],
            subset["cosine_pca_average_mean"],
            s=90,
            label=embedding,
            color=EMBEDDING_COLORS[embedding],
            alpha=0.9,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["model"],
                (row["rmse_average_mean"], row["cosine_pca_average_mean"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title("RMSE vs PCA Cosine")
    ax.set_xlabel("rmse_average_mean (lower better)")
    ax.set_ylabel("cosine_pca_average_mean (higher better)")
    ax.grid(alpha=0.25)
    ax.legend(title="embedding")
    fig.tight_layout()
    fig.savefig(report_dir / "rmse_vs_cosine_pca.png", dpi=180)
    plt.close(fig)


def plot_metric_heatmap(summary: pd.DataFrame, report_dir: Path, metric: str) -> None:
    col = f"{metric}_mean"
    if col not in summary.columns:
        return
    pivot = summary.pivot(index="model", columns="embedding", values=col)
    pivot = pivot.reindex(index=MODEL_ORDER, columns=EMBEDDING_ORDER)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title(metric)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i, model in enumerate(pivot.index):
        for j, embedding in enumerate(pivot.columns):
            value = pivot.loc[model, embedding]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(report_dir / f"{metric}_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-dir",
        default="summary_tables/sciplex3_cv",
        help="Directory containing per-embedding summary/per-fold/missing CSVs.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Output report directory (default: <summary-dir>/report).",
    )
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    report_dir = Path(args.report_dir or summary_dir / "report")
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csvs(summary_dir, "summary")
    per_fold = read_csvs(summary_dir, "per_fold")
    missing = read_csvs(summary_dir, "missing")
    if summary.empty:
        raise SystemExit(f"No summary CSVs found in {summary_dir}")

    if "eval_checkpoint" not in summary.columns:
        summary["eval_checkpoint"] = "legacy"

    # Prefer the newest generated copy if duplicate rows exist.
    summary = summary.drop_duplicates(
        ["embedding", "model", "experiment", "eval_checkpoint"], keep="last"
    )
    summary["embedding"] = summary["embedding"].replace({"l1000": "lpm"})
    summary["embedding"] = ordered_category(summary["embedding"], EMBEDDING_ORDER)
    summary["model"] = ordered_category(summary["model"], MODEL_ORDER)
    summary["eval_checkpoint"] = ordered_category(
        summary["eval_checkpoint"], EVAL_CHECKPOINT_ORDER
    )
    summary = summary.sort_values(["model", "embedding", "eval_checkpoint"]).reset_index(drop=True)

    combined_out = report_dir / "combined_summary.csv"
    compact_out = report_dir / "comparison_table.csv"
    markdown_out = report_dir / "comparison_table.md"
    best_out = report_dir / "best_by_metric.csv"
    per_fold_out = report_dir / "combined_per_fold.csv"
    missing_out = report_dir / "combined_missing.csv"

    summary.to_csv(combined_out, index=False)
    if not per_fold.empty:
        per_fold.to_csv(per_fold_out, index=False)
    if not missing.empty:
        missing.to_csv(missing_out, index=False)
    else:
        pd.DataFrame(
            columns=[
                "input_folder",
                "embedding",
                "model",
                "experiment",
                "fold",
                "status",
                "summary_path",
                "checkpoint_path",
                "note",
            ]
        ).to_csv(missing_out, index=False)

    compact = build_compact_table(summary)
    compact.to_csv(compact_out, index=False)
    markdown_out.write_text(dataframe_to_markdown(compact))

    best = build_best_by_metric(summary)
    best.to_csv(best_out, index=False)

    for eval_checkpoint in summary["eval_checkpoint"].dropna().unique():
        eval_summary = summary[summary["eval_checkpoint"].eq(eval_checkpoint)]
        eval_report_dir = report_dir / str(eval_checkpoint)
        eval_report_dir.mkdir(parents=True, exist_ok=True)
        for metric in DISPLAY_METRICS:
            plot_metric_bars(eval_summary, eval_report_dir, metric)
        plot_rmse_vs_cosine(eval_summary, eval_report_dir)
        for metric in ["rmse_average", "cosine_pca_average", "mmd_pca"]:
            plot_metric_heatmap(eval_summary, eval_report_dir, metric)

    print(f"Wrote {combined_out}")
    print(f"Wrote {compact_out}")
    print(f"Wrote {markdown_out}")
    print(f"Wrote {best_out}")
    print(f"Wrote plots to {report_dir}")
    print("\nBest complete model/embedding by metric:")
    if best.empty:
        print("  No complete rows available.")
    else:
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()
