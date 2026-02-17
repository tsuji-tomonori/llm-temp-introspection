"""Study1: 温度 vs HIGH率の Spearman ρ を算出するスクリプト

(model, prompt_type, target) ごとに温度と HIGH率の単調関係を検定する。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualization.study1_heatmap import load_study1_data

STUDY2_MODELS = {"NOVA_MICRO", "NOVA_2_LITE", "GEMMA_3N_E4B", "DEVSTRAL"}
EXCLUDE_TARGETS = {"ELEPHANT"}


def compute_spearman_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """(model, prompt_type, target) ごとに Spearman ρ を算出する。"""
    df_valid = df[df["judgment"].isin(["HIGH", "LOW"])].copy()
    df_valid["is_high"] = (df_valid["judgment"] == "HIGH").astype(int)

    grouped = (
        df_valid.groupby(["model", "prompt_type", "target", "temperature"])
        .agg(high_rate=("is_high", "mean"))
        .reset_index()
    )

    rows = []
    for (model, prompt_type, target), g in grouped.groupby(
        ["model", "prompt_type", "target"]
    ):
        temps = g["temperature"].values
        rates = g["high_rate"].values

        if len(temps) < 3 or rates.std() == 0:
            rows.append(
                {
                    "model": model,
                    "prompt_type": prompt_type,
                    "target": target,
                    "spearman_rho": float("nan"),
                    "p_value": float("nan"),
                }
            )
        else:
            rho, p = stats.spearmanr(temps, rates)
            rows.append(
                {
                    "model": model,
                    "prompt_type": prompt_type,
                    "target": target,
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )

    return pd.DataFrame(rows)


def summarize_by_prompt(detail_df: pd.DataFrame) -> pd.DataFrame:
    """(model, prompt_type) で集約した概要版を作成する。"""
    rows = []
    for (model, prompt_type), g in detail_df.groupby(["model", "prompt_type"]):
        valid = g.dropna(subset=["spearman_rho"])
        rows.append(
            {
                "model": model,
                "prompt_type": prompt_type,
                "mean_rho": (
                    valid["spearman_rho"].mean()
                    if len(valid) > 0
                    else float("nan")
                ),
                "significant_count": int((valid["p_value"] < 0.05).sum()),
                "total_count": len(g),
                "nan_count": int(g["spearman_rho"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Spearman ρ for temperature vs HIGH rate in Study 1"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "output",
        help="Study 1 output root directory",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=Path.cwd() / "output" / "analysis",
        help="Directory to save analysis CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.analysis_output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Study 1 Spearman Analysis ===")
    df = load_study1_data(args.output_dir, allowed_models=STUDY2_MODELS)
    df = df[~df["target"].isin(EXCLUDE_TARGETS)]
    print(f"After ELEPHANT exclusion: {len(df)} records")

    detail = compute_spearman_by_group(df)
    detail_path = args.analysis_output_dir / "study1_spearman.csv"
    detail.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    summary = summarize_by_prompt(detail)
    summary_path = args.analysis_output_dir / "study1_spearman_by_prompt.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\n--- Detail (head) ---")
    print(detail.to_string(index=False))
    print("\n--- Summary ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
