"""追実験D: Blind / Wrong-label 予測実験ランナー

Study2の予測プロンプトからラベル情報を隠す（Blind）/入れ替える（Wrong-label）ことで、
ラベル依存度を定量化する。
"""

import argparse
import logging
import time
from pathlib import Path

from core.llm import LlmExecution
from models.llm import ModelId
from models.temperature_introspection import (
    LLMConfig,
    PromptType,
    Study2BlindPromptVariables,
    Study2ConditionType,
    Study2PromptVariables,
    TemperatureJudgment,
    TemperaturePredictionResponse,
)
from study.s2 import (
    build_result,
    load_study1_candidates,
    parse_model_list,
    result_output_path,
    save_result,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROMPT_TYPE_SWAP: dict[PromptType, PromptType] = {
    PromptType.FACTUAL: PromptType.CRAZY,
    PromptType.CRAZY: PromptType.FACTUAL,
    PromptType.NORMAL: PromptType.NORMAL,
}


def run_blind_prediction(
    samples: list[dict],
    output_dir: Path,
    predictor_models: list[ModelId],
    skip_existing: bool,
) -> tuple[int, int, int]:
    """Blind条件: prompt_type/targetを隠して予測を実行する。"""
    saved = 0
    skipped = 0
    failed = 0

    for sample in samples:
        for predictor in predictor_models:
            preliminary_result = build_result(
                condition_type=Study2ConditionType.BLIND,
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
                    prompt_name="study2_prediction_blind",
                    kwargs=Study2BlindPromptVariables(
                        generated_sentence=sample["generated_sentence"],
                    ),
                )
                result = build_result(
                    condition_type=Study2ConditionType.BLIND,
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
                    "Failed blind prediction: generator=%s predictor=%s source=%s",
                    sample["generator_model"].name,
                    predictor.name,
                    sample["source_unique_id"],
                )

    return saved, skipped, failed


def run_wrong_label_prediction(
    samples: list[dict],
    output_dir: Path,
    predictor_models: list[ModelId],
    skip_existing: bool,
) -> tuple[int, int, int]:
    """Wrong-label条件: prompt_typeを入れ替えて予測を実行する。"""
    saved = 0
    skipped = 0
    failed = 0

    for sample in samples:
        swapped_prompt_type = PROMPT_TYPE_SWAP[sample["prompt_type"]]
        if swapped_prompt_type == sample["prompt_type"]:
            # NORMALはswap対象外
            continue

        for predictor in predictor_models:
            preliminary_result = build_result(
                condition_type=Study2ConditionType.WRONG_LABEL,
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
                        prompt_type=swapped_prompt_type.value,
                        target=sample["target"].value,
                    ),
                )
                result = build_result(
                    condition_type=Study2ConditionType.WRONG_LABEL,
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
                    "Failed wrong_label prediction: "
                    "generator=%s predictor=%s source=%s",
                    sample["generator_model"].name,
                    predictor.name,
                    sample["source_unique_id"],
                )

    return saved, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment D: Blind / Wrong-label prediction runner"
    )
    parser.add_argument(
        "--study1-output-dir",
        type=Path,
        default=Path.cwd() / "output",
        help="Study 1 results root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "output" / "experiment_d",
        help="Experiment D output directory",
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
        help="Comma-separated generator model enum names",
    )
    parser.add_argument(
        "--predictor-models",
        type=parse_model_list,
        default=None,
        help="Comma-separated predictor model enum names",
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
        help="Overwrite existing outputs",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
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

    logger.info("=== Experiment D execution start ===")
    logger.info(f"Candidate samples: {len(samples)}")
    logger.info(f"Generator models: {[m.name for m in generator_models]}")
    logger.info(f"Predictor models: {[m.name for m in predictor_models]}")
    logger.info(f"Output dir: {args.output_dir}")

    blind_saved, blind_skipped, blind_failed = run_blind_prediction(
        samples=samples,
        output_dir=args.output_dir,
        predictor_models=predictor_models,
        skip_existing=skip_existing,
    )
    logger.info(
        "blind saved=%s skipped=%s failed=%s",
        blind_saved,
        blind_skipped,
        blind_failed,
    )

    wl_saved, wl_skipped, wl_failed = run_wrong_label_prediction(
        samples=samples,
        output_dir=args.output_dir,
        predictor_models=predictor_models,
        skip_existing=skip_existing,
    )
    logger.info(
        "wrong_label saved=%s skipped=%s failed=%s",
        wl_saved,
        wl_skipped,
        wl_failed,
    )

    logger.info("=== Experiment D execution completed ===")


if __name__ == "__main__":
    main()
