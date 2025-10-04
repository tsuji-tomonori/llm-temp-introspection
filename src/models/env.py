import os

from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """環境変数の設定を管理するモデル"""

    model_config = {
        "extra": "ignore",  # 未定義フィールドを無視
        "frozen": True,  # インスタンスを不変に
    }

    base_url: str = Field(
        default="http://192.168.1.3:1234/v1",
        description="APIのベースURL (例: http://192.168.1.3:1234/v1)",
    )
    api_key: str = Field(
        default="your_api_key",
        description="API認証用のキー",
    )
    timeout: int = Field(
        default=30,
        description="APIリクエストのタイムアウト時間（秒）",
    )
    max_retries: int = Field(
        default=3,
        description="API呼び出し失敗時の最大リトライ回数",
    )

    @classmethod
    def from_env(cls) -> "EnvConfig":
        """環境変数から設定を読み込む"""
        return cls.model_validate(os.environ)
