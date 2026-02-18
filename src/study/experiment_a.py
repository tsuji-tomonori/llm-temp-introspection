"""追実験A: Info+/Info−（情報密度操作）実験ランナー

同一内容の文をInfo+（詳細版）/Info−（簡略版）に編集し、
情報密度だけで温度判定が動くかを検証する。

Step 4a: 編集ペア生成（editor_modelでInfo+/Info−を生成）
Step 4b: 予測実行（各predictor_modelで温度予測）
"""

import argparse
import json
import logging
import time
from pathlib import Path

from core.llm import LlmExecution
from models.llm import ModelId
from models.temperature_introspection import (
    ExperimentAEditedPair,
    ExperimentAEditPromptVariables,
    LLMConfig,
    PromptType,
    SentenceEditingResponse,
    Study2ConditionType,
    Study2ExperimentalCondition,
    Study2ExperimentalResult,
    Study2PromptVariables,
    TemperaturePredictionResponse,
)
from study.s2 import load_study1_candidates, parse_model_list

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_edited_pairs(
    samples: list[dict],
    output_dir: Path,
    editor_model: ModelId,
    skip_existing: bool,
) -> tuple[int, int, int]:
    """Step 4a: NORMALプロンプトのサンプルからInfo+/Info−編集ペアを生成する。"""
    saved = 0
    skipped = 0
    failed = 0

    normal_samples = [s for s in samples if s["prompt_type"] == PromptType.NORMAL]
    logger.info(f"NORMAL samples for editing: {len(normal_samples)}")

    for sample in normal_samples:
        out_file = (
            output_dir
            / "edited"
            / sample["generator_model"].name
            / f"{sample['source_unique_id']}.json"
        )
        if skip_existing and out_file.exists():
            skipped += 1
            continue

        start = time.time()
        try:
            model = LlmExecution(
                config=LLMConfig(model_id=editor_model, temperature=0.0),
            )
            response = model.execute(
                model_type=SentenceEditingResponse,
                prompt_name="experiment_a_edit",
                kwargs=ExperimentAEditPromptVariables(
                    generated_sentence=sample["generated_sentence"],
                ),
            )
            pair = ExperimentAEditedPair(
                source_unique_id=sample["source_unique_id"],
                generator_model=sample["generator_model"],
                prompt_type=sample["prompt_type"],
                target=sample["target"],
                temperature=sample["temperature"],
                expected_judgment=sample["expected_judgment"],
                original_sentence=sample["generated_sentence"],
                info_plus=response.info_plus,
                info_minus=response.info_minus,
                loop_times=sample["loop_times"],
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(pair.model_dump_json(indent=2))
            saved += 1
            logger.info(
                "Edited pair saved: generator=%s source=%s (%.1fs)",
                sample["generator_model"].name,
                sample["source_unique_id"],
                time.time() - start,
            )
        except Exception:
            failed += 1
            logger.exception(
                "Failed editing: generator=%s source=%s",
                sample["generator_model"].name,
                sample["source_unique_id"],
            )

    return saved, skipped, failed


def load_edited_pairs(edited_dir: Path) -> list[ExperimentAEditedPair]:
    """編集済みペアをディレクトリから読み込む。"""
    pairs: list[ExperimentAEditedPair] = []
    for json_file in edited_dir.glob("*/*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            pairs.append(ExperimentAEditedPair(**data))
        except Exception:
            logger.exception(f"Failed to load edited pair: {json_file}")
    return pairs


def run_predictions(
    pairs: list[ExperimentAEditedPair],
    output_dir: Path,
    predictor_models: list[ModelId],
    skip_existing: bool,
) -> tuple[int, int, int]:
    """Step 4b: Info+/Info−それぞれに対してpredictor_modelsで温度予測を実行する。"""
    saved = 0
    skipped = 0
    failed = 0

    variants: list[tuple[str, Study2ConditionType]] = [
        ("info_plus", Study2ConditionType.INFO_PLUS),
        ("info_minus", Study2ConditionType.INFO_MINUS),
    ]

    for pair in pairs:
        for variant_key, condition_type in variants:
            sentence = pair.info_plus if variant_key == "info_plus" else pair.info_minus

            for predictor in predictor_models:
                out_file = (
                    output_dir
                    / "predictions"
                    / variant_key
                    / pair.generator_model.name
                    / predictor.name
                    / f"{pair.source_unique_id}.json"
                )
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
                            generated_sentence=sentence,
                            prompt_type=pair.prompt_type.value,
                            target=pair.target.value,
                        ),
                    )
                    condition = Study2ExperimentalCondition(
                        condition_type=condition_type,
                        generator_model_id=pair.generator_model,
                        predictor_model_id=predictor,
                        temperature=pair.temperature,
                        expected_judgment=pair.expected_judgment,
                        prompt_type=pair.prompt_type,
                        target=pair.target,
                        source_loop_times=pair.loop_times,
                        source_unique_id=pair.source_unique_id,
                    )
                    result = Study2ExperimentalResult(
                        condition=condition,
                        generated_sentence=sentence,
                        reasoning=response.reasoning,
                        predicted_judgment=response.judgment,
                        is_correct=(
                            response.judgment == pair.expected_judgment
                        ),
                        procession_time_ms=int((time.time() - start) * 1000),
                    )
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(result.model_dump_json(indent=2))
                    saved += 1
                except Exception:
                    failed += 1
                    logger.exception(
                        "Failed prediction: variant=%s generator=%s "
                        "predictor=%s source=%s",
                        variant_key,
                        pair.generator_model.name,
                        predictor.name,
                        pair.source_unique_id,
                    )

    return saved, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A: Info+/Info- information density experiment"
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
        default=Path.cwd() / "output" / "experiment_a",
        help="Experiment A output directory",
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
        "--editor-model",
        type=lambda v: ModelId[v],
        default=ModelId.NOVA_2_LITE,
        help="Model to use for editing sentences (default: NOVA_2_LITE)",
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
        "--skip-edit",
        action="store_true",
        help="Skip editing step and only run predictions on existing pairs",
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

    skip_existing = not args.force

    if not args.skip_edit:
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

        logger.info("=== Experiment A: Step 4a - Editing ===")
        logger.info(f"Candidate samples: {len(samples)}")
        logger.info(f"Editor model: {args.editor_model.name}")

        edit_saved, edit_skipped, edit_failed = generate_edited_pairs(
            samples=samples,
            output_dir=args.output_dir,
            editor_model=args.editor_model,
            skip_existing=skip_existing,
        )
        logger.info(
            "editing saved=%s skipped=%s failed=%s",
            edit_saved,
            edit_skipped,
            edit_failed,
        )

    # Step 4b: Predictions
    edited_dir = args.output_dir / "edited"
    if not edited_dir.exists():
        logger.info("No edited pairs directory found. Run editing step first.")
        return

    pairs = load_edited_pairs(edited_dir)
    if not pairs:
        logger.info("No edited pairs found.")
        return

    generator_models = sorted(
        {pair.generator_model for pair in pairs},
        key=lambda x: x.name,
    )
    predictor_models = args.predictor_models or generator_models

    logger.info("=== Experiment A: Step 4b - Predictions ===")
    logger.info(f"Edited pairs: {len(pairs)}")
    logger.info(f"Predictor models: {[m.name for m in predictor_models]}")

    pred_saved, pred_skipped, pred_failed = run_predictions(
        pairs=pairs,
        output_dir=args.output_dir,
        predictor_models=predictor_models,
        skip_existing=skip_existing,
    )
    logger.info(
        "predictions saved=%s skipped=%s failed=%s",
        pred_saved,
        pred_skipped,
        pred_failed,
    )

    logger.info("=== Experiment A execution completed ===")


if __name__ == "__main__":
    main()
