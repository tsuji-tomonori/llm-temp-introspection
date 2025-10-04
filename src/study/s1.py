import logging
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
models = (ModelId.QWEN3_CODER_30B, ModelId.GEMMA_3N_E4B, ModelId.GPT_OSS_20B)
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
    response = model.execute(
        model_type=TemperatureIntrospectionResponse,
        prompt_name="study1",
        kwargs=Study1PromptVariables(
            target=condition.target.value,
            prompt_type=condition.prompt_type.value,
        ),
    )
    result = Study1ExperimentalResult(
        condition=condition,
        response=response,  # type: ignore
        loop_times=items[4],
    )
    output_dir = (
        output_root_dir
        / condition.model_id.name
        / condition.target.name
        / condition.prompt_type.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir / f"temp_{condition.temperature}_loop_{result.loop_times}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2, ensure_ascii=False))  # type: ignore
    logger.info(f"Saved result to {output_file}")
