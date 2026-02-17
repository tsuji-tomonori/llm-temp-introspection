"""ブログ用 Study2 Δ(self-within) フォレストプロット

モデルごとの Δ ポイント推定値 + 95% bootstrap CI を表示する。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_forest(df: pd.DataFrame, output_path: Path) -> None:
    """フォレストプロットを描画する。"""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(df))

    # Error bars
    xerr_lower = df["delta_self_within"] - df["ci_lower"]
    xerr_upper = df["ci_upper"] - df["delta_self_within"]
    xerr = np.array([xerr_lower.values, xerr_upper.values])

    ax.errorbar(
        df["delta_self_within"],
        y_pos,
        xerr=xerr,
        fmt="o",
        color="#2166AC",
        ecolor="#67A9CF",
        elinewidth=2.5,
        capsize=5,
        capthick=2,
        markersize=8,
    )

    # Reference line at Δ=0
    ax.axvline(0, color="#D6604D", linestyle="--", linewidth=1.5, label="Δ = 0")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["predictor_model"], fontsize=9)
    ax.set_xlabel("Δ Accuracy (self − within)", fontsize=11)
    ax.set_title(
        "Study 2: Self-Reflection Advantage (Δ with 95% Bootstrap CI)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=9)

    # Annotate values
    for i, row in df.iterrows():
        ax.annotate(
            f"  {row['delta_self_within']:+.3f}"
            f" [{row['ci_lower']:+.3f}, {row['ci_upper']:+.3f}]",
            xy=(row["ci_upper"], y_pos[i]),
            fontsize=8,
            va="center",
        )

    ax.invert_yaxis()
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
        description="Generate Study 2 forest plot for blog"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path.cwd() / "output" / "analysis" / "study2_bootstrap_ci.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.cwd() / "output" / "figures" / "blog_study2_delta",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)
    plot_forest(df, args.output_path)


if __name__ == "__main__":
    main()
