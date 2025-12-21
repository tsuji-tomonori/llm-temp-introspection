"""Study 1結果のヒートマップ可視化スクリプト

Figure 2(a)スタイルのヒートマップを生成する。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_study1_data(output_dir: Path) -> pd.DataFrame:
    """全JSONファイルを読み込みDataFrameに変換

    Args:
        output_dir: outputディレクトリのパス

    Returns:
        全実験結果を含むDataFrame
    """
    records = []

    # モデルディレクトリを走査
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        model_name = model_dir.name

        # ターゲットディレクトリを走査
        for target_dir in model_dir.iterdir():
            if not target_dir.is_dir():
                continue

            target_name = target_dir.name

            # プロンプトタイプディレクトリを走査
            for prompt_type_dir in target_dir.iterdir():
                if not prompt_type_dir.is_dir():
                    continue

                prompt_type = prompt_type_dir.name

                # JSONファイルを走査
                for json_file in prompt_type_dir.glob("temp_*.json"):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # データを抽出
                        record = {
                            "model": model_name,
                            "target": target_name,
                            "prompt_type": prompt_type,
                            "temperature": data["condition"]["temperature"],
                            "judgment": data["response"]["judgment"],
                            "loop_times": data.get("loop_times", 0),
                        }
                        records.append(record)

                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {len(df['model'].unique())} models")
    return df


def aggregate_high_rate(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """HIGH率を計算してモデル別のピボットテーブルを作成

    Args:
        df: 実験結果DataFrame

    Returns:
        モデル名をキーとし、ピボットテーブルを値とする辞書
    """
    # 有効な判定（HIGH/LOW）のみをフィルタ
    df_valid = df[df["judgment"].isin(["HIGH", "LOW"])].copy()

    # HIGH率を計算
    df_valid["is_high"] = (df_valid["judgment"] == "HIGH").astype(int)

    # モデル別に処理
    pivots = {}
    for model in sorted(df_valid["model"].unique()):
        df_model = df_valid[df_valid["model"] == model]

        # グループ化して集計
        grouped = (
            df_model.groupby(["prompt_type", "target", "temperature"])
            .agg({"is_high": ["sum", "count"]})
            .reset_index()
        )

        # カラム名をフラット化
        grouped.columns = ["prompt_type", "target", "temperature", "high_count", "total_count"]

        # HIGH率を計算
        grouped["high_rate"] = grouped["high_count"] / grouped["total_count"]

        # 行ラベルを作成（例: "FACTUAL - ELEPHANT"）
        grouped["row_label"] = grouped["prompt_type"] + " - " + grouped["target"]

        # ピボットテーブルを作成
        pivot = grouped.pivot(index="row_label", columns="temperature", values="high_rate")

        pivots[model] = pivot

    return pivots


def plot_study1_heatmap(
    data: dict[str, pd.DataFrame],
    output_path: Path,
    figsize: tuple = (22, 15),
    cmap: str = "RdYlGn_r",
) -> None:
    """Figure 2(a)スタイルのヒートマップを描画

    Args:
        data: モデル名をキーとし、ピボットテーブルを値とする辞書
        output_path: 出力ファイルパス（拡張子なし）
        figsize: 図のサイズ
        cmap: カラーマップ（論文に合わせてRdYlGn_r: 赤=HIGH, 黄=中間, 緑=LOW）
    """
    models = sorted(data.keys())
    n_models = len(models)

    # サブプロットのレイアウトを決定（2x3グリッド）
    nrows = 2
    ncols = 3

    # サブプロット間のスペースを調整
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    # 各モデルのヒートマップを描画
    for idx, model in enumerate(models):
        ax = axes[idx]
        pivot = data[model]

        # ヒートマップを描画
        # cbar_kwsでカラーバーのパディングを調整
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            cbar_kws={
                "label": "Predicted Temp\n(HIGH Rate)",
                "pad": 0.02,  # カラーバーとグラフの間隔
            },
            annot=False,
            fmt=".2f",
            linewidths=0.5,
            linecolor="gray",
        )

        # タイトルとラベル
        ax.set_title(model, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Temperature", fontsize=11, labelpad=8)
        ax.set_ylabel("Condition", fontsize=11, labelpad=8)

        # Y軸ラベルのフォントサイズを調整
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

        # X軸のラベルを回転
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    # 余ったサブプロットを非表示
    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    # レイアウトを調整（サブプロット間のスペースを確保）
    plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=3.0)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")

    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")

    print(f"Saved heatmap to:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")

    plt.close(fig)


def main() -> None:
    """メイン処理"""
    # パスの設定
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "output"
    figures_dir = output_dir / "figures"

    print("=== Study 1 Heatmap Generation ===")
    print(f"Output directory: {output_dir}")

    # データを読み込み
    print("\n1. Loading data...")
    df = load_study1_data(output_dir)

    if df.empty:
        print("Error: No data found!")
        return

    # 統計情報を表示
    print("\n2. Data statistics:")
    print(f"   Models: {df['model'].unique().tolist()}")
    print(f"   Targets: {df['target'].unique().tolist()}")
    print(f"   Prompt types: {df['prompt_type'].unique().tolist()}")
    print(f"   Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}")
    print(f"   Total records: {len(df)}")

    # HIGH率を集計
    print("\n3. Aggregating HIGH rates...")
    pivots = aggregate_high_rate(df)

    # ヒートマップを描画
    print("\n4. Plotting heatmap...")
    output_path = figures_dir / "study1_heatmap"
    plot_study1_heatmap(pivots, output_path)

    print("\n=== Completed ===")


if __name__ == "__main__":
    main()
