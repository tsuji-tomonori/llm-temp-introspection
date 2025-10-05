import logging
import time
from itertools import product
from pathlib import Path

from core.llm import LlmExecution
from models.llm import ModelId
from models.temperature_introspection import (
    PromptType,
    Study1ExperimentalCondition,
    Study1ExperimentalResult,
    Study1PromptVariables,
    Target,
    TemperatureIntrospectionResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
loop_times = range(3)  # 各条件でのループ回数
temperatures = (round(i * 0.1, 1) for i in range(0, 10 + 1))
models = (ModelId.QWEN3_CODER_30B, ModelId.MAGISTRAL_SAMLL)
output_root_dir = Path.cwd() / "output"

for items in product(models, temperatures, PromptType, Target, loop_times):
    condition = Study1ExperimentalCondition(
        model_id=items[0],
        temperature=items[1],
        prompt_type=items[2],
        target=items[3],
    )
    logger.info(f"Executing with condition: {condition}")
    model = LlmExecution(config=condition)
    output_dir = (
        output_root_dir
        / condition.model_id.name
        / condition.target.name
        / condition.prompt_type.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"temp_{condition.temperature}_loop_{items[4]}.json"
    if output_file.exists():
        continue
    start_time = time.time()
    response = model.execute(
        model_type=TemperatureIntrospectionResponse,
        prompt_name="study1",
        kwargs=Study1PromptVariables(
            target=condition.target.value,
            prompt_type=condition.prompt_type.value,
        ),
    )
    end_time = time.time()
    processing_time = end_time - start_time
    result = Study1ExperimentalResult(
        condition=condition,
        response=response,  # type: ignore
        loop_times=items[4],
        procession_time_ms=int(processing_time * 1000),  # ミリ秒単位に変換
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))  # type: ignore
    logger.info(f"Saved result to {output_file} elapsed_time: {processing_time:.2f}s")
