from enum import Enum


class ModelType(Enum):
    """モデルタイプ.

    LLMのモデルタイプを示す列挙型。
    これにより、異なるLLMプロバイダーやモデルに対応した処理を行うことができる。
    例えば、LM Studio系モデルとAmazon Bedrock系モデルで異なる設定やAPI呼び出しが必要な場合に使用する。
    """

    LM_STUDIO = "LM_STUDIO"
    AWS_BEDROCK = "AWS_BEDROCK"


class ModelId(Enum):
    """モデルID.

    使用する具体的なLLMモデルを識別するための列挙型。

    **LM Studio系モデル:**
    - qwen/qwen3-coder-30b
    - google/gemma-3n-e4b
    - openai/gpt-oss-20b

    **Amazon Bedrock系モデル:**
    - amazon/nova-micro
    - amazon/titan-text-lite
    - anthropic/claude-3
    """

    QWEN3_CODER_30B = "qwen/qwen3-coder-30b"
    GEMMA_3N_E4B = "google/gemma-3n-e4b"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    MAGISTRAL_SAMLL = "mistralai/magistral-small-2509"
    IBM_GRANITE4_TINY = "ibm/granite-4-h-tiny"
    NOVA_MICRO = "amazon.nova-micro-v1:0"
    DEVSTRAL = "mistralai/devstral-small-2507"
    NOVA_2_LITE = "global.amazon.nova-2-lite-v1:0"
    CLAUDE_CODE_HAIKU_4_5 = "anthropic.claude-haiku-4-5-20251001-v1:0"

    def model_type(self) -> ModelType:
        """モデルタイプに応じたモデルIDを返す"""
        match self:
            case (
                ModelId.QWEN3_CODER_30B
                | ModelId.GEMMA_3N_E4B
                | ModelId.GPT_OSS_20B
                | ModelId.MAGISTRAL_SAMLL
                | ModelId.IBM_GRANITE4_TINY
                | ModelId.DEVSTRAL
            ):
                return ModelType.LM_STUDIO
            case (
                ModelId.NOVA_MICRO | ModelId.NOVA_2_LITE | ModelId.CLAUDE_CODE_HAIKU_4_5
            ):
                return ModelType.AWS_BEDROCK
            case _:
                raise ValueError(f"Unsupported model ID: {self.value}")
