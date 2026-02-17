"""Study1: ロジスティック回帰によるプロンプト種別 vs 温度の効果比較

ネストモデル比較と尤度比検定で、どちらが支配的かを定量化する。
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visualization.study1_heatmap import load_study1_data

STUDY2_MODELS = {"NOVA_MICRO", "NOVA_2_LITE", "GEMMA_3N_E4B", "DEVSTRAL"}
EXCLUDE_TARGETS = {"ELEPHANT"}


def _fit_glm(
    formula: str, data: pd.DataFrame
) -> tuple[object, pd.DataFrame]:
    """GLM を当てはめ、(result, X) を返す。"""
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    y_vals = y.iloc[:, 0]
    glm = sm.GLM(y_vals, X, family=sm.families.Binomial())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = glm.fit()
    except Exception:
        result = glm.fit_regularized(alpha=0.1, L1_wt=0.0)
    return result, X


def compute_effects(data: pd.DataFrame) -> dict:
    """温度効果とプロンプト効果の確率差を算出する。"""
    formula = "is_high ~ temp + C(prompt_type) + C(target)"
    result, X = _fit_glm(formula, data)

    # 温度効果: P(HIGH|temp=0.9) - P(HIGH|temp=0.1), prompt_type=NORMAL固定
    base_row = pd.DataFrame({col: [0.0] for col in X.columns})
    base_row["Intercept"] = 1.0

    row_low = base_row.copy()
    row_low["temp"] = 0.1
    p_low = result.predict(row_low).iloc[0]

    row_high = base_row.copy()
    row_high["temp"] = 0.9
    p_high = result.predict(row_high).iloc[0]

    temp_effect = float(p_high - p_low)

    # prompt効果: P(HIGH|CRAZY) - P(HIGH|FACTUAL), temp=0.5固定
    # Reference category is CRAZY (alphabetically first)
    row_crazy = base_row.copy()
    row_crazy["temp"] = 0.5
    p_crazy = result.predict(row_crazy).iloc[0]

    row_factual = base_row.copy()
    row_factual["temp"] = 0.5
    factual_col = [c for c in X.columns if "FACTUAL" in c]
    if factual_col:
        row_factual[factual_col[0]] = 1.0
    p_factual = result.predict(row_factual).iloc[0]

    prompt_effect = float(p_crazy - p_factual)

    return {
        "temp_effect": temp_effect,
        "prompt_effect_crazy_vs_factual": prompt_effect,
    }


def run_nested_comparison(
    data: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """4つのネストモデルを当てはめ、AIC比較とLRTを行う。"""
    formulas = {
        "M_temp": "is_high ~ temp + C(target)",
        "M_prompt": "is_high ~ C(prompt_type) + C(target)",
        "M_both": "is_high ~ temp + C(prompt_type) + C(target)",
        "M_int": "is_high ~ temp * C(prompt_type) + C(target)",
    }

    results = {}
    model_rows = []
    for name, formula in formulas.items():
        res, X = _fit_glm(formula, data)
        results[name] = res
        model_rows.append(
            {
                "model_name": name,
                "aic": getattr(res, "aic", float("nan")),
                "llf": getattr(res, "llf", float("nan")),
                "df": X.shape[1],
            }
        )

    # Build a lookup for df values
    df_lookup = {r["model_name"]: r["df"] for r in model_rows}

    # LRT comparisons
    lrt_rows = []
    comparisons = [
        ("M_both vs M_prompt", "M_both", "M_prompt"),
        ("M_both vs M_temp", "M_both", "M_temp"),
    ]
    for label, full_name, reduced_name in comparisons:
        full = results[full_name]
        reduced = results[reduced_name]
        full_llf = getattr(full, "llf", None)
        reduced_llf = getattr(reduced, "llf", None)
        df_diff = df_lookup[full_name] - df_lookup[reduced_name]

        if full_llf is not None and reduced_llf is not None:
            lr_stat = 2 * (full_llf - reduced_llf)
            if df_diff > 0:
                p_value = float(
                    1 - stats.chi2.cdf(lr_stat, df_diff)
                )
            else:
                p_value = float("nan")
        else:
            lr_stat = float("nan")
            p_value = float("nan")

        lrt_rows.append(
            {
                "comparison": label,
                "LR_stat": lr_stat,
                "df": df_diff,
                "p_value": p_value,
            }
        )

    return model_rows, lrt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logistic regression analysis for Study 1"
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

    print("=== Study 1 Logistic Regression Analysis ===")
    df = load_study1_data(args.output_dir, allowed_models=STUDY2_MODELS)
    df = df[~df["target"].isin(EXCLUDE_TARGETS)]

    df_valid = df[df["judgment"].isin(["HIGH", "LOW"])].copy()
    df_valid["is_high"] = (df_valid["judgment"] == "HIGH").astype(int)
    df_valid["temp"] = df_valid["temperature"]
    print(f"Valid records: {len(df_valid)}")

    all_glm_rows: list[dict] = []
    all_lrt_rows: list[dict] = []
    all_effect_rows: list[dict] = []

    for model_name in sorted(df_valid["model"].unique()):
        print(f"\nProcessing model: {model_name}")
        model_data = df_valid[df_valid["model"] == model_name].copy()

        glm_rows, lrt_rows = run_nested_comparison(model_data)
        for r in glm_rows:
            r["model"] = model_name
        for r in lrt_rows:
            r["model"] = model_name

        all_glm_rows.extend(glm_rows)
        all_lrt_rows.extend(lrt_rows)

        effects = compute_effects(model_data)
        effects["model"] = model_name
        all_effect_rows.append(effects)

    # Save outputs
    glm_df = pd.DataFrame(all_glm_rows)
    glm_path = args.analysis_output_dir / "study1_glm_comparison.csv"
    glm_df.to_csv(glm_path, index=False)
    print(f"\nSaved: {glm_path}")

    lrt_df = pd.DataFrame(all_lrt_rows)
    lrt_path = args.analysis_output_dir / "study1_glm_lrt.csv"
    lrt_df.to_csv(lrt_path, index=False)
    print(f"Saved: {lrt_path}")

    effects_df = pd.DataFrame(all_effect_rows)
    effects_path = args.analysis_output_dir / "study1_glm_effects.csv"
    effects_df.to_csv(effects_path, index=False)
    print(f"Saved: {effects_path}")

    print("\n--- GLM Comparison ---")
    print(glm_df.to_string(index=False))
    print("\n--- LRT ---")
    print(lrt_df.to_string(index=False))
    print("\n--- Effects ---")
    print(effects_df.to_string(index=False))


if __name__ == "__main__":
    main()
