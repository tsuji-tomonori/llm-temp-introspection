"""追実験A分析: Info+/Info−の温度判定バイアスを定量化する

predictor_modelごとに P(HIGH|Info+) - P(HIGH|Info-) の
平均delta + bootstrap 95% CI を算出。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42


def load_prediction_rows(predictions_dir: Path) -> list[dict]:
    """predictions/{info_plus|info_minus}/.../*.json から結果を読み込む。"""
    rows: list[dict] = []
    for variant in ["info_plus", "info_minus"]:
        variant_dir = predictions_dir / variant
        if not variant_dir.exists():
            continue
        for json_file in variant_dir.glob("*/*/*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                condition = data["condition"]
                rows.append(
                    {
                        "variant": variant,
                        "predictor_model": condition["predictor_model_id"],
                        "generator_model": condition["generator_model_id"],
                        "source_unique_id": condition["source_unique_id"],
                        "expected_judgment": condition["expected_judgment"],
                        "predicted_judgment": data["predicted_judgment"],
                        "is_high": data["predicted_judgment"] == "HIGH",
                    }
                )
            except Exception:
                continue
    return rows


def compute_p_high_delta(df: pd.DataFrame) -> pd.DataFrame:
    """predictor_modelごとに P(HIGH|Info+) - P(HIGH|Info-) を算出する。"""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for predictor in sorted(df["predictor_model"].unique()):
        df_pred = df[df["predictor_model"] == predictor]

        info_plus = df_pred[df_pred["variant"] == "info_plus"]
        info_minus = df_pred[df_pred["variant"] == "info_minus"]

        if info_plus.empty or info_minus.empty:
            continue

        # Pair by source_unique_id
        plus_map = dict(
            zip(
                info_plus["source_unique_id"],
                info_plus["is_high"].astype(int),
                strict=False,
            )
        )
        minus_map = dict(
            zip(
                info_minus["source_unique_id"],
                info_minus["is_high"].astype(int),
                strict=False,
            )
        )

        common_ids = sorted(set(plus_map.keys()) & set(minus_map.keys()))
        if not common_ids:
            continue

        plus_arr = np.array([plus_map[uid] for uid in common_ids])
        minus_arr = np.array([minus_map[uid] for uid in common_ids])

        p_high_plus = float(plus_arr.mean())
        p_high_minus = float(minus_arr.mean())
        observed_delta = p_high_plus - p_high_minus

        # Bootstrap CI
        n = len(common_ids)
        deltas = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            deltas[i] = plus_arr[idx].mean() - minus_arr[idx].mean()

        ci_lower = float(np.percentile(deltas, 2.5))
        ci_upper = float(np.percentile(deltas, 97.5))

        rows.append(
            {
                "predictor_model": predictor,
                "p_high_info_plus": round(p_high_plus, 4),
                "p_high_info_minus": round(p_high_minus, 4),
                "delta": round(observed_delta, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "n_pairs": n,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A analysis: Info+/Info- P(HIGH) delta"
    )
    parser.add_argument(
        "--experiment-a-output-dir",
        type=Path,
        default=Path.cwd() / "output" / "experiment_a",
        help="Experiment A output directory",
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

    print("=== Experiment A Analysis ===")

    predictions_dir = args.experiment_a_output_dir / "predictions"
    raw_rows = load_prediction_rows(predictions_dir)
    print(f"Loaded {len(raw_rows)} prediction rows")

    if not raw_rows:
        print("No data to analyze.")
        return

    df = pd.DataFrame(raw_rows)

    delta_df = compute_p_high_delta(df)
    delta_path = args.analysis_output_dir / "experiment_a_p_high_delta.csv"
    delta_df.to_csv(delta_path, index=False)
    print(f"Saved: {delta_path}")
    print("\n--- P(HIGH) Delta (Info+ - Info-) ---")
    print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()
