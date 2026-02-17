"""Study2: balanced accuracy / macro-F1 / bootstrap CI を算出するスクリプト

predictor_model × condition_type ごとの詳細メトリクスと、
Δ(self - within) の bootstrap 95% CI を報告する。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from study.s2 import collect_result_rows

CONDITION_ORDER = ["self_reflection", "within_model", "across_model"]
N_BOOTSTRAP = 10_000
RANDOM_SEED = 42


def compute_detailed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """predictor_model × condition_type ごとのメトリクスを算出する。"""
    rows = []
    for (predictor, cond), g in df.groupby(["predictor_model", "condition_type"]):
        y_true = g["expected_judgment"].values
        y_pred = g["predicted_judgment"].values
        is_correct = g["is_correct"].values

        accuracy = float(is_correct.mean())
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Majority baseline: predict the most common class
        from collections import Counter

        counts = Counter(y_true)
        majority_class_count = max(counts.values())
        majority_baseline = majority_class_count / len(y_true)

        rows.append(
            {
                "predictor_model": predictor,
                "condition_type": cond,
                "accuracy": round(accuracy, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "macro_f1": round(macro_f1, 4),
                "majority_baseline": round(majority_baseline, 4),
                "n_samples": len(g),
            }
        )

    result = pd.DataFrame(rows)
    # Sort by condition order
    cond_order_map = {c: i for i, c in enumerate(CONDITION_ORDER)}
    result["_sort"] = result["condition_type"].map(cond_order_map)
    result = result.sort_values(["predictor_model", "_sort"]).drop(columns=["_sort"])
    return result.reset_index(drop=True)


def compute_bootstrap_ci(df: pd.DataFrame) -> pd.DataFrame:
    """predictor_model ごとに Δ(self - within) の bootstrap 95% CI を算出する。"""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for predictor in sorted(df["predictor_model"].unique()):
        df_pred = df[df["predictor_model"] == predictor]

        self_data = df_pred[df_pred["condition_type"] == "self_reflection"]
        within_data = df_pred[df_pred["condition_type"] == "within_model"]

        if self_data.empty or within_data.empty:
            continue

        # Pair by source_unique_id
        self_map = dict(
            zip(self_data["source_unique_id"], self_data["is_correct"], strict=False)
        )
        within_map = dict(
            zip(
                within_data["source_unique_id"],
                within_data["is_correct"],
                strict=False,
            )
        )

        common_ids = sorted(set(self_map.keys()) & set(within_map.keys()))
        if not common_ids:
            continue

        self_arr = np.array([int(self_map[uid]) for uid in common_ids])
        within_arr = np.array([int(within_map[uid]) for uid in common_ids])

        observed_delta = float(self_arr.mean() - within_arr.mean())

        # Bootstrap
        n = len(common_ids)
        deltas = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            deltas[i] = self_arr[idx].mean() - within_arr[idx].mean()

        ci_lower = float(np.percentile(deltas, 2.5))
        ci_upper = float(np.percentile(deltas, 97.5))

        rows.append(
            {
                "predictor_model": predictor,
                "delta_self_within": round(observed_delta, 4),
                "ci_lower": round(ci_lower, 4),
                "ci_upper": round(ci_upper, 4),
                "n_paired": n,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute detailed Study 2 metrics with bootstrap CI"
    )
    parser.add_argument(
        "--study2-output-dir",
        type=Path,
        default=Path.cwd() / "output" / "study2",
        help="Study 2 output directory",
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

    print("=== Study 2 Detailed Analysis ===")
    raw_rows = collect_result_rows(args.study2_output_dir)
    print(f"Loaded {len(raw_rows)} result rows")

    df = pd.DataFrame(raw_rows)
    df = df[df["condition_type"].isin(CONDITION_ORDER)]

    # For bootstrap, we need source_unique_id. Re-read from JSON files.
    print("Loading source_unique_id for bootstrap pairing...")
    import json

    enriched_rows = []
    for json_file in args.study2_output_dir.glob("*/*/*/*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            condition = data["condition"]
            enriched_rows.append(
                {
                    "condition_type": condition["condition_type"],
                    "generator_model": condition["generator_model_id"],
                    "predictor_model": condition["predictor_model_id"],
                    "expected_judgment": condition["expected_judgment"],
                    "predicted_judgment": data["predicted_judgment"],
                    "is_correct": bool(data["is_correct"]),
                    "source_unique_id": condition.get("source_unique_id", ""),
                }
            )
        except Exception:
            continue

    df_full = pd.DataFrame(enriched_rows)
    df_full = df_full[df_full["condition_type"].isin(CONDITION_ORDER)]
    print(f"Enriched rows with source_unique_id: {len(df_full)}")

    # Detailed metrics
    metrics = compute_detailed_metrics(df_full)
    metrics_path = args.analysis_output_dir / "study2_detailed_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    # Bootstrap CI
    bootstrap = compute_bootstrap_ci(df_full)
    bootstrap_path = args.analysis_output_dir / "study2_bootstrap_ci.csv"
    bootstrap.to_csv(bootstrap_path, index=False)
    print(f"Saved: {bootstrap_path}")

    print("\n--- Detailed Metrics ---")
    print(metrics.to_string(index=False))
    print("\n--- Bootstrap CI ---")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
