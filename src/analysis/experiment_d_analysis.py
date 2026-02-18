"""追実験D分析: Blind/Wrong-label結果とStudy2 Full条件を比較する

- (predictor_model, label_condition) ごとの accuracy, balanced_accuracy, macro_f1
- Wrong-label shift分析: P(HIGH) の変化量
- Bootstrap CI付き
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_BOOTSTRAP = 10_000
RANDOM_SEED = 42


def load_result_rows(result_dir: Path) -> list[dict]:
    """JSON結果ファイルからフラットな辞書リストを読み込む。"""
    rows: list[dict] = []
    for json_file in result_dir.glob("*/*/*/*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            condition = data["condition"]
            rows.append(
                {
                    "condition_type": condition["condition_type"],
                    "generator_model": condition["generator_model_id"],
                    "predictor_model": condition["predictor_model_id"],
                    "expected_judgment": condition["expected_judgment"],
                    "predicted_judgment": data["predicted_judgment"],
                    "is_correct": bool(data["is_correct"]),
                    "source_unique_id": condition.get("source_unique_id", ""),
                    "prompt_type": condition.get("prompt_type", ""),
                }
            )
        except Exception:
            continue
    return rows


def compute_accuracy_by_label_condition(df: pd.DataFrame) -> pd.DataFrame:
    """(predictor_model, label_condition) ごとのメトリクスを算出する。"""
    rows = []
    for (predictor, cond), g in df.groupby(["predictor_model", "label_condition"]):
        y_true = g["expected_judgment"].values
        y_pred = g["predicted_judgment"].values
        is_correct = g["is_correct"].values

        accuracy = float(is_correct.mean())
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        counts = Counter(y_true)
        majority_class_count = max(counts.values())
        majority_baseline = majority_class_count / len(y_true)

        rows.append(
            {
                "predictor_model": predictor,
                "label_condition": cond,
                "accuracy": round(accuracy, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "macro_f1": round(macro_f1, 4),
                "majority_baseline": round(majority_baseline, 4),
                "n_samples": len(g),
            }
        )

    result = pd.DataFrame(rows)
    cond_order = {"full": 0, "blind": 1, "wrong_label": 2}
    result["_sort"] = result["label_condition"].map(cond_order)
    result = result.sort_values(["predictor_model", "_sort"]).drop(columns=["_sort"])
    return result.reset_index(drop=True)


def compute_wrong_label_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Wrong-label shift分析: FACTUAL-as-CRAZY / CRAZY-as-FACTUAL の P(HIGH) 変化。

    Study2 within_model (Full条件) と experiment_d wrong_label を比較する。
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for predictor in sorted(df["predictor_model"].unique()):
        df_pred = df[df["predictor_model"] == predictor]

        # Full条件 (within_model) のFACTUAL/CRAZYサンプル
        full_factual = df_pred[
            (df_pred["label_condition"] == "full")
            & (df_pred["prompt_type"] == "事実に基づいた")
        ]
        full_crazy = df_pred[
            (df_pred["label_condition"] == "full")
            & (df_pred["prompt_type"] == "クレイジーな")
        ]

        # Wrong-label条件のFACTUAL/CRAZYサンプル（元のprompt_typeで分類）
        wl_factual = df_pred[
            (df_pred["label_condition"] == "wrong_label")
            & (df_pred["prompt_type"] == "事実に基づいた")
        ]
        wl_crazy = df_pred[
            (df_pred["label_condition"] == "wrong_label")
            & (df_pred["prompt_type"] == "クレイジーな")
        ]

        for original_type, full_df, wl_df, swap_desc in [
            ("FACTUAL", full_factual, wl_factual, "FACTUAL→CRAZY"),
            ("CRAZY", full_crazy, wl_crazy, "CRAZY→FACTUAL"),
        ]:
            if full_df.empty or wl_df.empty:
                continue

            # P(HIGH) in each condition
            p_high_full = float(
                (full_df["predicted_judgment"] == "HIGH").mean()
            )
            p_high_wl = float(
                (wl_df["predicted_judgment"] == "HIGH").mean()
            )
            delta = p_high_wl - p_high_full

            # Bootstrap CI for the delta
            common_ids = sorted(
                set(full_df["source_unique_id"]) & set(wl_df["source_unique_id"])
            )
            if not common_ids:
                ci_lower, ci_upper = float("nan"), float("nan")
                n_paired = 0
            else:
                full_map = dict(
                    zip(
                        full_df["source_unique_id"],
                        (full_df["predicted_judgment"] == "HIGH").astype(int),
                        strict=False,
                    )
                )
                wl_map = dict(
                    zip(
                        wl_df["source_unique_id"],
                        (wl_df["predicted_judgment"] == "HIGH").astype(int),
                        strict=False,
                    )
                )
                full_arr = np.array([full_map[uid] for uid in common_ids])
                wl_arr = np.array([wl_map[uid] for uid in common_ids])

                n_paired = len(common_ids)
                deltas = np.empty(N_BOOTSTRAP)
                for i in range(N_BOOTSTRAP):
                    idx = rng.integers(0, n_paired, size=n_paired)
                    deltas[i] = wl_arr[idx].mean() - full_arr[idx].mean()

                ci_lower = float(np.percentile(deltas, 2.5))
                ci_upper = float(np.percentile(deltas, 97.5))

            rows.append(
                {
                    "predictor_model": predictor,
                    "swap_direction": swap_desc,
                    "original_prompt_type": original_type,
                    "p_high_full": round(p_high_full, 4),
                    "p_high_wrong_label": round(p_high_wl, 4),
                    "delta_p_high": round(delta, 4),
                    "ci_lower": round(ci_lower, 4) if not np.isnan(ci_lower) else None,
                    "ci_upper": round(ci_upper, 4) if not np.isnan(ci_upper) else None,
                    "n_full": len(full_df),
                    "n_wrong_label": len(wl_df),
                    "n_paired": n_paired,
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment D analysis: Blind/Wrong-label vs Full comparison"
    )
    parser.add_argument(
        "--study2-output-dir",
        type=Path,
        default=Path.cwd() / "output" / "study2",
        help="Study 2 output directory (for Full/within_model results)",
    )
    parser.add_argument(
        "--experiment-d-output-dir",
        type=Path,
        default=Path.cwd() / "output" / "experiment_d",
        help="Experiment D output directory",
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

    print("=== Experiment D Analysis ===")

    # Load Study2 within_model results as "full" condition
    full_rows = load_result_rows(args.study2_output_dir)
    full_rows = [r for r in full_rows if r["condition_type"] == "within_model"]
    for r in full_rows:
        r["label_condition"] = "full"
    print(f"Full (within_model) rows: {len(full_rows)}")

    # Load Experiment D results
    exp_d_rows = load_result_rows(args.experiment_d_output_dir)
    for r in exp_d_rows:
        r["label_condition"] = r["condition_type"]
    print(f"Experiment D rows: {len(exp_d_rows)}")

    all_rows = full_rows + exp_d_rows
    if not all_rows:
        print("No data to analyze.")
        return

    df = pd.DataFrame(all_rows)

    # Accuracy by label condition
    accuracy_df = compute_accuracy_by_label_condition(df)
    accuracy_path = (
        args.analysis_output_dir / "experiment_d_accuracy_by_label_condition.csv"
    )
    accuracy_df.to_csv(accuracy_path, index=False)
    print(f"Saved: {accuracy_path}")
    print("\n--- Accuracy by Label Condition ---")
    print(accuracy_df.to_string(index=False))

    # Wrong-label shift
    shift_df = compute_wrong_label_shift(df)
    shift_path = args.analysis_output_dir / "experiment_d_wrong_label_shift.csv"
    shift_df.to_csv(shift_path, index=False)
    print(f"\nSaved: {shift_path}")
    print("\n--- Wrong-label Shift ---")
    print(shift_df.to_string(index=False))


if __name__ == "__main__":
    main()
