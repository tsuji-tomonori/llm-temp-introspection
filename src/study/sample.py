from core.llm import LlmExecution
from models.llm import ModelId
from models.temperature_introspection import LLMConfig, TemperatureIntrospectionResponse

model = LlmExecution(
    config=LLMConfig(
        model_id=ModelId.QWEN3_CODER_30B,
        temperature=0.0,
    )
)
response = model.execute(
    model_type=TemperatureIntrospectionResponse,
    prompt_name="study1",
    target="猫",
    mode="カジュアルな",
)
print(response.model_dump_json())  # type: ignore
