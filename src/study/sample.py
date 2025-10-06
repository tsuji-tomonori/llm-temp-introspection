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
        model_id=ModelId.NOVA_MICRO,
        temperature=0.2,
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
# response = model.sample(
#     prompt_name="study1",
#     kwargs=Study1PromptVariables(
#         target=Target.UNICORN.value,
#         prompt_type=PromptType.FACTUAL.value,
#     ),
# )
# print(response)
