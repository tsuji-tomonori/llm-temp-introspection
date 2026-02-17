"""温度パラメータ推測実験のレスポンスモデル"""

import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from models.llm import ModelId


class PromptType(Enum):
    """プロンプトタイプ（モード）"""

    FACTUAL = "事実に基づいた"
    NORMAL = ""
    CRAZY = "クレイジーな"


class Target(Enum):
    """文生成の対象"""

    ELEPHANT = "像"
    ELEPHANT_KANA = "ゾウ"
    UNICORN = "ユニコーン"
    MURLOC = "マーロック"
    IRET_DOKODOKO_YATTAZE_PENGUIN = "アイレット・ドコドコ・ヤッタゼ・ペンギン"


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
        ..., description="指定されたターゲットとモードに基づいてLLMが生成した文"
    )
    reasoning: str = Field(
        ..., description="生成した文を踏まえた温度パラメータ（HIGH/LOW）の考察内容"
    )
    judgment: TemperatureJudgment = Field(
        ..., description="温度パラメータに関する最終判断（HIGH または LOW）"
    )


class TemperaturePredictionResponse(BaseModel):
    """温度予測タスク（Study 2）用レスポンスモデル"""

    reasoning: str = Field(..., description="温度パラメータ（HIGH/LOW）の考察内容")
    judgment: TemperatureJudgment = Field(
        ..., description="温度パラメータに関する最終判断（HIGH または LOW）"
    )


class LLMConfig(BaseModel):
    """LLMに与える基本設定

    モデルIDと温度パラメータを含む、LLM実行に必要な最小限の設定。
    """

    model_id: ModelId = Field(..., description="使用するLLMモデル")
    temperature: float = Field(
        ..., description="生成時の温度パラメータ（0.0〜2.0）", ge=0.0, le=2.0
    )


class Study1ExperimentalCondition(LLMConfig):
    """Study 1: プロンプトスタイルと対象による温度推測の実験条件

    LLMが生成した文から自身の温度を推測できるかを検証する実験の条件を定義。
    モデル、温度、プロンプトタイプ、対象の4つの変数を組み合わせて実験を実施。
    """

    prompt_type: PromptType = Field(..., description="プロンプトのタイプ（モード）")
    target: Target = Field(..., description="文生成の対象")


class Study1PromptVariables(BaseModel):
    """Study 1: プロンプト変数モデル

    プロンプトテンプレートに埋め込む変数を定義。
    実験条件に基づいてプロンプトを動的に生成するために使用する。
    """

    target: str = Field(..., description="文生成の対象")
    prompt_type: str = Field(..., description="プロンプトのタイプ（モード）")


class Study2PromptVariables(BaseModel):
    """Study 2: 温度予測用プロンプト変数モデル"""

    generated_sentence: str = Field(..., description="対象モデルが生成した文")
    prompt_type: str = Field(..., description="生成時のプロンプトタイプ")
    target: str = Field(..., description="生成時の対象")


class Study1ExperimentalResult(BaseModel):
    """Study 1: 実験結果モデル

    各実験条件に対するLLMの応答と、正解温度パラメータを含む。
    """

    condition: Study1ExperimentalCondition = Field(..., description="実験条件")
    response: TemperatureIntrospectionResponse = Field(..., description="LLMの応答")
    loop_times: int = Field(..., description="実験のループ回数")
    unique_id: str = Field(
        default_factory=lambda: str(uuid4()), description="実験の一意な識別子"
    )
    procession_time_ms: int = Field(..., description="実験の処理時間（ミリ秒単位）")
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="実験結果の作成日時（ISO 8601形式）",
    )


class Study2ConditionType(str, Enum):
    """Study 2の実験条件タイプ"""

    SELF_REFLECTION = "self_reflection"
    WITHIN_MODEL = "within_model"
    ACROSS_MODEL = "across_model"


class Study2ExperimentalCondition(BaseModel):
    """Study 2: 実験条件モデル"""

    condition_type: Study2ConditionType = Field(..., description="Study 2の条件タイプ")
    generator_model_id: ModelId = Field(..., description="文を生成したモデル")
    predictor_model_id: ModelId = Field(..., description="温度を推定するモデル")
    temperature: float = Field(..., description="生成時温度")
    expected_judgment: TemperatureJudgment = Field(..., description="正解ラベル")
    prompt_type: PromptType = Field(..., description="生成時プロンプトタイプ")
    target: Target = Field(..., description="生成時ターゲット")
    source_loop_times: int = Field(..., description="Study 1側のloop回数")
    source_unique_id: str = Field(..., description="Study 1側の一意ID")


class Study2ExperimentalResult(BaseModel):
    """Study 2: 実験結果モデル"""

    condition: Study2ExperimentalCondition = Field(..., description="実験条件")
    generated_sentence: str = Field(..., description="判定対象の生成文")
    reasoning: str = Field(..., description="推定理由")
    predicted_judgment: TemperatureJudgment = Field(..., description="推定結果")
    is_correct: bool = Field(..., description="推定が正解かどうか")
    unique_id: str = Field(
        default_factory=lambda: str(uuid4()), description="実験の一意な識別子"
    )
    procession_time_ms: int = Field(..., description="実験の処理時間（ミリ秒単位）")
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(),
        description="実験結果の作成日時（ISO 8601形式）",
    )
