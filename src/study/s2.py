import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from core.llm import LlmExecution
from models.llm import ModelId
from models.temperature_introspection import (
    LLMConfig,
    PromptType,
    Study2ConditionType,
    Study2ExperimentalCondition,
    Study2ExperimentalResult,
    Study2PromptVariables,
    Target,
    TemperatureJudgment,
    TemperaturePredictionResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_BY_VALUE = {model.value: model for model in ModelId}
PROMPT_TYPE_BY_VALUE = {prompt_type.value: prompt_type for prompt_type in PromptType}
TARGET_BY_VALUE = {target.value: target for target in Target}


def parse_model_list(value: str) -> list[ModelId]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    models: list[ModelId] = []
    for item in values:
        try:
            models.append(ModelId[item])
        except KeyError as exc:
            available = ", ".join(model.name for model in ModelId)
            raise ValueError(
                f"Unknown model name: {item}. Available: {available}"
            ) from exc
    return models


def expected_judgment_from_temperature(
    temperature: float, low_max: float, high_min: float
) -> TemperatureJudgment | None:
    if temperature <= low_max:
        return TemperatureJudgment.LOW
    if temperature >= high_min:
        return TemperatureJudgment.HIGH
    return None


def load_study1_candidates(
    output_dir: Path,
    low_max: float,
    high_min: float,
    generator_models: list[ModelId] | None,
) -> list[dict]:
    records: list[dict] = []
    allow_generators = (
        {model.value for model in generator_models} if generator_models else None
    )
    for json_file in output_dir.glob("*/*/*/temp_*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            condition = data["condition"]
            response = data["response"]
            model_value = condition["model_id"]
            if allow_generators is not None and model_value not in allow_generators:
                continue

            expected_judgment = expected_judgment_from_temperature(
                float(condition["temperature"]), low_max=low_max, high_min=high_min
            )
            if expected_judgment is None:
                continue

            model_id = MODEL_BY_VALUE.get(model_value)
            prompt_type = PROMPT_TYPE_BY_VALUE.get(condition["prompt_type"])
            target = TARGET_BY_VALUE.get(condition["target"])
            if model_id is None or prompt_type is None or target is None:
                continue

            source_judgment = response.get("judgment")
            if source_judgment not in {
                TemperatureJudgment.HIGH.value,
                TemperatureJudgment.LOW.value,
            }:
                continue

            records.append(
                {
                    "source_path": str(json_file),
                    "source_unique_id": data.get("unique_id") or json_file.stem,
                    "generator_model": model_id,
                    "prompt_type": prompt_type,
                    "target": target,
                    "temperature": float(condition["temperature"]),
                    "loop_times": int(data.get("loop_times", 0)),
                    "generated_sentence": response["generated_sentence"],
                    "source_reasoning": response.get("reasoning", ""),
                    "source_judgment": TemperatureJudgment(source_judgment),
                    "expected_judgment": expected_judgment,
                }
            )
        except Exception:
            logger.exception(f"Failed to load {json_file}")
            continue
    return records


def build_result(
    *,
    condition_type: Study2ConditionType,
    sample: dict,
    predictor_model: ModelId,
    reasoning: str,
    predicted_judgment: TemperatureJudgment,
    processing_time_ms: int,
) -> Study2ExperimentalResult:
    condition = Study2ExperimentalCondition(
        condition_type=condition_type,
        generator_model_id=sample["generator_model"],
        predictor_model_id=predictor_model,
        temperature=sample["temperature"],
        expected_judgment=sample["expected_judgment"],
        prompt_type=sample["prompt_type"],
        target=sample["target"],
        source_loop_times=sample["loop_times"],
        source_unique_id=sample["source_unique_id"],
    )
    return Study2ExperimentalResult(
        condition=condition,
        generated_sentence=sample["generated_sentence"],
        reasoning=reasoning,
        predicted_judgment=predicted_judgment,
        is_correct=(predicted_judgment == sample["expected_judgment"]),
        procession_time_ms=processing_time_ms,
    )


def result_output_path(output_dir: Path, result: Study2ExperimentalResult) -> Path:
    condition = result.condition
    return (
        output_dir
        / condition.condition_type.value
        / condition.generator_model_id.name
        / condition.predictor_model_id.name
        / f"{condition.source_unique_id}.json"
    )


def save_result(
    output_dir: Path,
    result: Study2ExperimentalResult,
    skip_existing: bool,
) -> bool:
    out_file = result_output_path(output_dir, result)
    if skip_existing and out_file.exists():
        return False
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    return True


def run_self_reflection(
    samples: list[dict],
    output_dir: Path,
    skip_existing: bool,
) -> tuple[int, int]:
    saved = 0
    skipped = 0
    for sample in samples:
        start = time.time()
        result = build_result(
            condition_type=Study2ConditionType.SELF_REFLECTION,
            sample=sample,
            predictor_model=sample["generator_model"],
            reasoning=sample["source_reasoning"],
            predicted_judgment=sample["source_judgment"],
            processing_time_ms=int((time.time() - start) * 1000),
        )
        if save_result(output_dir, result, skip_existing=skip_existing):
            saved += 1
        else:
            skipped += 1
    return saved, skipped


def run_prediction(
    samples: list[dict],
    output_dir: Path,
    predictor_models: list[ModelId],
    condition_type: Study2ConditionType,
    skip_existing: bool,
) -> tuple[int, int, int]:
    saved = 0
    skipped = 0
    failed = 0

    for sample in samples:
        generator = sample["generator_model"]
        if condition_type == Study2ConditionType.WITHIN_MODEL:
            predictors = [generator]
        else:
            predictors = [model for model in predictor_models if model != generator]

        for predictor in predictors:
            preliminary_result = build_result(
                condition_type=condition_type,
                sample=sample,
                predictor_model=predictor,
                reasoning="",
                predicted_judgment=TemperatureJudgment.LOW,
                processing_time_ms=0,
            )
            out_file = result_output_path(output_dir, preliminary_result)
            if skip_existing and out_file.exists():
                skipped += 1
                continue

            start = time.time()
            try:
                model = LlmExecution(
                    config=LLMConfig(model_id=predictor, temperature=0.0),
                )
                response = model.execute(
                    model_type=TemperaturePredictionResponse,
                    prompt_name="study2_prediction",
                    kwargs=Study2PromptVariables(
                        generated_sentence=sample["generated_sentence"],
                        prompt_type=sample["prompt_type"].value,
                        target=sample["target"].value,
                    ),
                )
                result = build_result(
                    condition_type=condition_type,
                    sample=sample,
                    predictor_model=predictor,
                    reasoning=response.reasoning,
                    predicted_judgment=response.judgment,
                    processing_time_ms=int((time.time() - start) * 1000),
                )
                save_result(output_dir, result, skip_existing=False)
                saved += 1
            except Exception:
                failed += 1
                logger.exception(
                    (
                        "Failed prediction: condition=%s generator=%s "
                        "predictor=%s source=%s"
                    ),
                    condition_type.value,
                    generator.name,
                    predictor.name,
                    sample["source_unique_id"],
                )

    return saved, skipped, failed


def collect_result_rows(study2_output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for json_file in study2_output_dir.glob("*/*/*/*.json"):
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
                }
            )
        except Exception:
            logger.exception(f"Failed to read result file: {json_file}")
    return rows


def build_summary(study2_output_dir: Path) -> pd.DataFrame:
    rows = collect_result_rows(study2_output_dir)
    if not rows:
        return pd.DataFrame(
            columns=["predictor_model", "condition_type", "accuracy", "n_samples"]
        )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["predictor_model", "condition_type"])
        .agg(accuracy=("is_correct", "mean"), n_samples=("is_correct", "count"))
        .reset_index()
        .sort_values(["predictor_model", "condition_type"])
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study 2: privileged self-access experiment runner"
    )
    parser.add_argument(
        "--study1-output-dir",
        type=Path,
        default=Path.cwd() / "output",
        help="Study 1 results root directory",
    )
    parser.add_argument(
        "--study2-output-dir",
        type=Path,
        default=Path.cwd() / "output" / "study2",
        help="Study 2 output directory",
    )
    parser.add_argument(
        "--low-max",
        type=float,
        default=0.5,
        help="LOW label threshold: temperature <= low_max",
    )
    parser.add_argument(
        "--high-min",
        type=float,
        default=0.8,
        help="HIGH label threshold: temperature >= high_min",
    )
    parser.add_argument(
        "--generator-models",
        type=parse_model_list,
        default=None,
        help=(
            "Comma-separated generator model enum names "
            "(e.g. QWEN3_CODER_30B,NOVA_MICRO)"
        ),
    )
    parser.add_argument(
        "--predictor-models",
        type=parse_model_list,
        default=None,
        help=(
            "Comma-separated predictor model enum names. "
            "Default: same set as generators"
        ),
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Use only first N candidate samples for quick checks",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Study 2 outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.low_max >= args.high_min:
        raise ValueError("low-max must be smaller than high-min")

    samples = load_study1_candidates(
        output_dir=args.study1_output_dir,
        low_max=args.low_max,
        high_min=args.high_min,
        generator_models=args.generator_models,
    )
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]

    if not samples:
        logger.info("No eligible Study 1 samples found.")
        return

    generator_models = sorted(
        {sample["generator_model"] for sample in samples},
        key=lambda x: x.name,
    )
    predictor_models = args.predictor_models or generator_models
    skip_existing = not args.force

    logger.info("=== Study 2 execution start ===")
    logger.info(f"Study 1 input: {args.study1_output_dir}")
    logger.info(f"Study 2 output: {args.study2_output_dir}")
    logger.info(f"Thresholds: LOW<= {args.low_max}, HIGH>= {args.high_min}")
    logger.info(f"Candidate samples: {len(samples)}")
    logger.info(f"Generator models: {[model.name for model in generator_models]}")
    logger.info(f"Predictor models: {[model.name for model in predictor_models]}")

    self_saved, self_skipped = run_self_reflection(
        samples=samples,
        output_dir=args.study2_output_dir,
        skip_existing=skip_existing,
    )
    logger.info(f"self_reflection saved={self_saved} skipped={self_skipped}")

    within_saved, within_skipped, within_failed = run_prediction(
        samples=samples,
        output_dir=args.study2_output_dir,
        predictor_models=predictor_models,
        condition_type=Study2ConditionType.WITHIN_MODEL,
        skip_existing=skip_existing,
    )
    logger.info(
        "within_model saved=%s skipped=%s failed=%s",
        within_saved,
        within_skipped,
        within_failed,
    )

    across_saved, across_skipped, across_failed = run_prediction(
        samples=samples,
        output_dir=args.study2_output_dir,
        predictor_models=predictor_models,
        condition_type=Study2ConditionType.ACROSS_MODEL,
        skip_existing=skip_existing,
    )
    logger.info(
        "across_model saved=%s skipped=%s failed=%s",
        across_saved,
        across_skipped,
        across_failed,
    )

    summary = build_summary(args.study2_output_dir)
    summary_file = args.study2_output_dir / "summary.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)

    logger.info("Saved summary: %s", summary_file)
    if not summary.empty:
        logger.info("\n%s", summary.to_string(index=False))
    logger.info("=== Study 2 execution completed ===")


if __name__ == "__main__":
    main()
