"""Study 1レポート向けヒートマップ生成スクリプト。

Study2完了モデル（NOVA_MICRO, NOVA_2_LITE）のみを1x2レイアウトで可視化する。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from visualization.study1_heatmap import aggregate_high_rate, load_study1_data

REPORT_MODELS = {"NOVA_MICRO", "NOVA_2_LITE"}


def plot_report_heatmap(output_path: Path) -> None:
    output_dir = Path.cwd() / "output"
    df = load_study1_data(output_dir=output_dir, allowed_models=REPORT_MODELS)
    EXCLUDE_TARGETS = {"ELEPHANT"}
    df = df[~df["target"].isin(EXCLUDE_TARGETS)]
    if df.empty:
        raise ValueError("No Study 1 records found for report models")

    pivots = aggregate_high_rate(df)
    models = [model for model in ("NOVA_MICRO", "NOVA_2_LITE") if model in pivots]
    if not models:
        raise ValueError("No pivot data available for report models")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(models), figsize=(16, 7), squeeze=False)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        sns.heatmap(
            pivots[model],
            ax=ax,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Predicted Temp\n(HIGH Rate)", "pad": 0.02},
            annot=False,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(model, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Temperature", fontsize=11, labelpad=8)
        ax.set_ylabel("Condition", fontsize=11, labelpad=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    plt.tight_layout(pad=1.8, w_pad=2.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    print("Saved report heatmap to:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")


def main() -> None:
    output_path = Path.cwd() / "output" / "figures" / "study1_heatmap_study2_models_report"
    plot_report_heatmap(output_path)


if __name__ == "__main__":
    main()
