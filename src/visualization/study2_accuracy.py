"""Study 2結果の精度可視化スクリプト

Figure 2(b)相当として、モデルごとの条件別accuracyを棒グラフで表示する。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CONDITION_ORDER = ["self_reflection", "within_model", "across_model"]
CONDITION_LABEL = {
    "self_reflection": "self-reflect",
    "within_model": "within-model predict",
    "across_model": "across-model predict",
}
CONDITION_COLORS = {
    # Study 1のRdYlGn系トーンに合わせた配色（赤→黄→緑）
    "self_reflection": "#D73027",
    "within_model": "#FEE08B",
    "across_model": "#1A9850",
}


def load_summary(summary_path: Path) -> pd.DataFrame:
    """summary.csvを読み込み、可視化用DataFrameを返す。"""
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Study 2 summary file not found: {summary_path}. "
            "Run src/study/s2.py first."
        )

    df = pd.read_csv(summary_path)
    required = {"predictor_model", "condition_type", "accuracy", "n_samples"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary.csv: {sorted(missing)}")

    df = df[df["condition_type"].isin(CONDITION_ORDER)].copy()
    df["condition_type"] = pd.Categorical(
        df["condition_type"], categories=CONDITION_ORDER, ordered=True
    )
    df = df.sort_values(["predictor_model", "condition_type"]).reset_index(drop=True)
    return df


def plot_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    """モデル x 条件のaccuracyを棒グラフで描画する。"""
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = [CONDITION_COLORS[condition] for condition in CONDITION_ORDER]

    sns.barplot(
        data=df,
        x="predictor_model",
        y="accuracy",
        hue="condition_type",
        hue_order=CONDITION_ORDER,
        palette=palette,
        ax=ax,
    )

    ax.axhline(
        0.5,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="majority / random baseline",
    )
    ax.set_title("Study 2: Accuracy by condition", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Model", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([i / 10 for i in range(0, 11)])

    for container in ax.containers:
        labels = [
            f"{bar.get_height():.2f}" if bar.get_height() > 0 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, fontsize=8, padding=2)

    handles, labels = ax.get_legend_handles_labels()
    label_map = {**CONDITION_LABEL, "majority / random baseline": "majority / random baseline"}
    mapped_labels = [label_map.get(label, label) for label in labels]
    ax.legend(handles, mapped_labels, title="Condition", loc="upper right")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure files:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Study 2 accuracy as grouped bar chart"
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path.cwd() / "output" / "study2" / "summary.csv",
        help="Path to Study 2 summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path.cwd() / "output" / "figures" / "study2_accuracy",
        help="Output path prefix without extension",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of predictor_model values to include (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_summary(args.summary_path)
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
        df = df[df["predictor_model"].isin(model_list)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No rows available in summary.csv for expected Study 2 conditions"
        )
    plot_accuracy(df, args.output_path)


if __name__ == "__main__":
    main()
