"""温度パラメータ推測実験のレスポンスモデル"""

from enum import Enum

from pydantic import BaseModel, Field

from models.llm import ModelId


class PromptType(str, Enum):
    """プロンプトタイプ（モード）"""

    FACTUAL = "factual"  # 事実に基づいて
    NORMAL = "normal"  # 通常
    ABSURD = "absurd"  # 常軌を逸した


class Target(str, Enum):
    """文生成の対象"""

    ELEPHANT = "elephant"  # 象
    UNICORN = "unicorn"  # ユニコーン
    MURLOC = "murloc"  # マーロック
    ADELE_DOKODOKO_YATTAZE_PENGUIN = (
        "adele-dokodoko-yattaze-penguin"  # アイレット・ドコドコ・ヤッタゼ・ペンギン
    )


class TemperatureJudgment(str, Enum):
    """温度パラメータの判定結果"""

    HIGH = "HIGH"
    LOW = "LOW"


class TemperatureIntrospectionResponse(BaseModel):
    """温度パラメータ推測実験のレスポンスモデル

    プロンプトに対するLLMの応答を構造化したモデル。
    LLMが生成した文、温度パラメータの考察、最終判定を含む。
    """

    generated_sentence: str = Field(
        description="指定されたターゲットとモードに基づいてLLMが生成した文"
    )
    reasoning: str = Field(
        description="生成した文を踏まえた温度パラメータ（HIGH/LOW）の考察内容"
    )
    judgment: TemperatureJudgment = Field(
        description="温度パラメータに関する最終判断（HIGH または LOW）"
    )


class LLMConfig(BaseModel):
    """LLMに与える基本設定

    モデルIDと温度パラメータを含む、LLM実行に必要な最小限の設定。
    """

    model_id: ModelId = Field(description="使用するLLMモデル")
    temperature: float = Field(
        description="生成時の温度パラメータ（0.0〜2.0）", ge=0.0, le=2.0
    )


class Study1ExperimentalCondition(LLMConfig):
    """Study 1: プロンプトスタイルと対象による温度推測の実験条件

    LLMが生成した文から自身の温度を推測できるかを検証する実験の条件を定義。
    モデル、温度、プロンプトタイプ、対象の4つの変数を組み合わせて実験を実施。
    """

    prompt_type: PromptType = Field(description="プロンプトのタイプ（モード）")
    target: Target = Field(description="文生成の対象")
