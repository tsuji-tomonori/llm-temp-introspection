from core.llm import LlmExecution
from models.llm import ModelId
from models.temperature_introspection import (
    LLMConfig,
    PromptType,
    Study1PromptVariables,
    Target,
    TemperatureIntrospectionResponse,
)

model = LlmExecution(
    config=LLMConfig(
        model_id=ModelId.GEMMA_3N_E4B,
        temperature=0.0,
    )
)
response = model.execute(
    model_type=TemperatureIntrospectionResponse,
    prompt_name="study1",
    kwargs=Study1PromptVariables(
        target=Target.UNICORN.value,
        prompt_type=PromptType.FACTUAL.value,
    ),
)
print(response.model_dump_json())  # type: ignore
