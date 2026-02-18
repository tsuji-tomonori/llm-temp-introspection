"""追実験D可視化

1. グループ付き棒グラフ: balanced_accuracy × label_condition (Full/Blind/Wrong-label)
2. Wrong-label shift図: FACTUAL-as-CRAZY vs CRAZY-as-FACTUAL の P(HIGH) 比較
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CONDITION_ORDER = ["full", "blind", "wrong_label"]
CONDITION_LABEL = {
    "full": "Full (within)",
    "blind": "Blind",
    "wrong_label": "Wrong-label",
}
CONDITION_COLORS = {
    "full": "#4393C3",
    "blind": "#FDB863",
    "wrong_label": "#D6604D",
}


def plot_accuracy_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """label_condition別のbalanced_accuracyをグループ付き棒グラフで描画する。"""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    df = df[df["label_condition"].isin(CONDITION_ORDER)].copy()
    df["label_condition"] = pd.Categorical(
        df["label_condition"], categories=CONDITION_ORDER, ordered=True
    )
    df = df.sort_values(["predictor_model", "label_condition"]).reset_index(drop=True)

    palette = [CONDITION_COLORS[c] for c in CONDITION_ORDER]

    sns.barplot(
        data=df,
        x="predictor_model",
        y="balanced_accuracy",
        hue="label_condition",
        hue_order=CONDITION_ORDER,
        palette=palette,
        ax=ax,
    )

    ax.axhline(
        0.5,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="random baseline",
    )
    ax.set_title(
        "Experiment D: Balanced Accuracy by Label Condition",
        fontsize=14,
        pad=12,
    )
    ax.set_xlabel("Predictor Model", fontsize=11)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([i / 10 for i in range(0, 11)])

    for container in ax.containers:
        labels = [
            f"{bar.get_height():.2f}" if bar.get_height() > 0 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, fontsize=8, padding=2)

    handles, labels = ax.get_legend_handles_labels()
    label_map = {**CONDITION_LABEL, "random baseline": "random baseline"}
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

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_wrong_label_shift(df: pd.DataFrame, output_path: Path) -> None:
    """Wrong-label shift: P(HIGH) Full vs Wrong-label 比較。"""
    if df.empty:
        print("No wrong-label shift data to plot.")
        return

    sns.set_theme(style="whitegrid")

    predictors = sorted(df["predictor_model"].unique())
    swap_dirs = sorted(df["swap_direction"].unique())
    n_predictors = len(predictors)
    n_swaps = len(swap_dirs)

    fig, axes = plt.subplots(1, n_swaps, figsize=(6 * n_swaps, 5), sharey=True)
    if n_swaps == 1:
        axes = [axes]

    bar_width = 0.35
    x = np.arange(n_predictors)

    for ax_idx, swap_dir in enumerate(swap_dirs):
        ax = axes[ax_idx]
        sub = df[df["swap_direction"] == swap_dir].set_index("predictor_model")

        full_vals = [
            sub.loc[p, "p_high_full"] if p in sub.index else 0
            for p in predictors
        ]
        wl_vals = [
            sub.loc[p, "p_high_wrong_label"] if p in sub.index else 0
            for p in predictors
        ]

        ax.bar(
            x - bar_width / 2, full_vals, bar_width,
            label="Full", color="#4393C3",
        )
        ax.bar(
            x + bar_width / 2, wl_vals, bar_width,
            label="Wrong-label", color="#D6604D",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(predictors, rotation=20, ha="right", fontsize=9)
        ax.set_title(swap_dir, fontsize=12, fontweight="bold")
        ax.set_ylabel("P(HIGH)" if ax_idx == 0 else "")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=9)
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.0)

    fig.suptitle(
        "Experiment D: Wrong-label P(HIGH) Shift",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
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
        description="Generate Experiment D plots"
    )
    parser.add_argument(
        "--accuracy-input",
        type=Path,
        default=(
            Path.cwd() / "output" / "analysis"
            / "experiment_d_accuracy_by_label_condition.csv"
        ),
    )
    parser.add_argument(
        "--shift-input",
        type=Path,
        default=(
            Path.cwd() / "output" / "analysis"
            / "experiment_d_wrong_label_shift.csv"
        ),
    )
    parser.add_argument(
        "--accuracy-output",
        type=Path,
        default=Path.cwd() / "output" / "figures" / "experiment_d_accuracy",
    )
    parser.add_argument(
        "--shift-output",
        type=Path,
        default=Path.cwd() / "output" / "figures" / "experiment_d_shift",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    accuracy_df = pd.read_csv(args.accuracy_input)
    if not accuracy_df.empty:
        plot_accuracy_comparison(accuracy_df, args.accuracy_output)
    else:
        print("No accuracy data to plot.")

    shift_df = pd.read_csv(args.shift_input)
    if not shift_df.empty:
        plot_wrong_label_shift(shift_df, args.shift_output)
    else:
        print("No shift data to plot.")


if __name__ == "__main__":
    main()
