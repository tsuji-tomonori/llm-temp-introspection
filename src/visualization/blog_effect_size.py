"""ブログ用 温度効果 vs プロンプト効果の棒グラフ

モデルごとの確率差を並べて比較する。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_effect_size(df: pd.DataFrame, output_path: Path) -> None:
    """温度効果 vs プロンプト効果の並列棒グラフを描画する。"""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df["temp_effect"],
        width,
        label="Temperature effect\n(P[HIGH|t=0.9] − P[HIGH|t=0.1])",
        color="#4393C3",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        df["prompt_effect_crazy_vs_factual"],
        width,
        label="Prompt effect\n(P[HIGH|CRAZY] − P[HIGH|FACTUAL])",
        color="#D6604D",
        edgecolor="white",
    )

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Probability Difference", fontsize=11)
    ax.set_title(
        "Study 1: Temperature Effect vs Prompt Effect (Logistic Regression)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")

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
        description="Generate effect size bar chart for blog"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path.cwd() / "output" / "analysis" / "study1_glm_effects.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.cwd() / "output" / "figures" / "blog_effect_size",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)
    plot_effect_size(df, args.output_path)


if __name__ == "__main__":
    main()
