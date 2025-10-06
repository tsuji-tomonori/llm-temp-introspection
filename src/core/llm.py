import logging
from pathlib import Path
from typing import Final

from jinja2 import Template
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from models.env import EnvConfig
from models.llm import ModelType
from models.temperature_introspection import LLMConfig

PROMPT_PATH: Final = Path.cwd() / "resources" / "prompts"
ENV: Final = EnvConfig.from_env()
logger = logging.getLogger(__name__)


def load_prompt(prompt_name: str, kwargs: BaseModel) -> str:
    """プロンプトを読み込み・レンダリング"""
    prompt_file = PROMPT_PATH / f"{prompt_name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"prompt file is not exists: {prompt_file}")
    try:
        with open(prompt_file, encoding="utf-8") as f:
            content = f.read()
        result = Template(content).render(kwargs.model_dump())
        logger.debug(f"Loaded prompt '{prompt_name}': {result}")
        return result
    except Exception:
        logger.exception(f"Failed to load or render prompt '{prompt_name}'")
        raise


class LlmExecution:
    """LLM実行クラス"""

    def __init__(self, config: LLMConfig) -> None:
        if config.model_id.model_type() == ModelType.LM_STUDIO:
            self.llm = ChatOpenAI(
                base_url=ENV.base_url,
                api_key=SecretStr(ENV.api_key),
                model=config.model_id.value,
                temperature=config.temperature,
                max_retries=ENV.max_retries,
                timeout=ENV.timeout,
            )
        elif config.model_id.model_type() == ModelType.AWS_BEDROCK:
            self.llm = ChatBedrock(  # type: ignore
                model=config.model_id.value,
                temperature=config.temperature,
            )
        else:
            raise ValueError("モデルの種類がLM_STUDIO, AWS_BEDROCK")

    def execute[T](self, model_type: T, prompt_name: str, kwargs: BaseModel) -> T:
        """LLMを実行し、結果を返す"""
        structured_llm = self.llm.with_structured_output(model_type)  # type: ignore
        response = structured_llm.invoke(load_prompt(prompt_name, kwargs))  # type: ignore
        return response  # type: ignore
