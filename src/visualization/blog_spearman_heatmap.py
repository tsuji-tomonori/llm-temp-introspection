"""ブログ用 Spearman ρ ヒートマップ

行=(prompt_type × target)、列=model の Spearman ρ をヒートマップで表示する。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROMPT_ORDER = ["FACTUAL", "NORMAL", "CRAZY"]


def plot_spearman_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Spearman ρ ヒートマップを描画する。"""
    df["row_label"] = df["prompt_type"] + " - " + df["target"]

    # Sort rows by prompt_type order, then target
    prompt_rank = {p: i for i, p in enumerate(PROMPT_ORDER)}
    df["_sort"] = df["prompt_type"].map(prompt_rank)
    df = df.sort_values(["_sort", "target"]).drop(columns=["_sort"])

    pivot = df.pivot(index="row_label", columns="model", values="spearman_rho")
    p_pivot = df.pivot(index="row_label", columns="model", values="p_value")

    # Preserve row order
    row_order = df["row_label"].unique().tolist()
    pivot = pivot.reindex(row_order)
    p_pivot = p_pivot.reindex(row_order)

    # Annotation: ρ value with * for p < 0.05, "const" for NaN
    annot = pivot.copy().astype(str)
    for r in pivot.index:
        for c in pivot.columns:
            val = pivot.loc[r, c]
            p_val = p_pivot.loc[r, c]
            if pd.isna(val):
                annot.loc[r, c] = "const"
            elif not pd.isna(p_val) and p_val < 0.05:
                annot.loc[r, c] = f"{val:.2f}*"
            else:
                annot.loc[r, c] = f"{val:.2f}"

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1.0,
        vmax=1.0,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Spearman ρ"},
        mask=False,
    )

    ax.set_title(
        "Study 1: Spearman ρ (Temperature vs HIGH Rate)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Condition (prompt_type - target)", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Spearman ρ heatmap for blog"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path.cwd() / "output" / "analysis" / "study1_spearman.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.cwd() / "output" / "figures" / "blog_spearman_heatmap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)
    plot_spearman_heatmap(df, args.output_path)


if __name__ == "__main__":
    main()
