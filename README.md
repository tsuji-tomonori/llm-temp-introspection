# llm-temp-introspection

論文「Privileged Self-Access Matters for Introspection in AI」の日本語追実験プロジェクト

## 概要

このプロジェクトは、LLMが自身の温度パラメータを正確に内省できるかを検証する実験です。元論文の実験を日本語で再現し、以下のモデルでテストします。

**LM Studio系モデル:**
- Qwen3 Coder 30B
- Gemma 3n E4B
- OpenAI GPT-OSS 20B
- Magistral Small 2509 (Mistral AI)
- IBM Granite 4 H Tiny
- Devstral Small 2507 (Mistral AI)

**Amazon Bedrock系モデル:**
- Amazon Nova Micro
- Amazon Titan Text Lite
- Claude 3 Haiku

## 実験内容

### Study 1: プロンプトスタイルと対象による温度推測
- **目的**: モデルが生成した文から自身の温度を推測できるか
- **変数**:
  - プロンプトタイプ
    - 事実に基づいて
    - 通常
    - 常軌を逸した
  - 対象:
    - 象
    - ユニコーン
    - マーロック
    - アイレット・ドコドコ・ヤッタゼ・ペンギン
  - 温度: 0.0〜2.0

### Study 2: 特権的自己アクセスの検証
- **目的**: モデルが他のモデルより自分の温度を正確に推測できるか
- **条件**:
  - self-reflection: 自分で生成→自分で推測
  - within-model: 同じモデルが別呼び出しで推測
  - across-model: 別のモデルが推測

## 実行方法

### Study 1の実行
```bash
PYTHONPATH=src uv run python src/study/s1.py
```

実験結果は `output/` ディレクトリに保存されます。

### Study 2の実行
```bash
PYTHONPATH=src uv run python src/study/s2.py
```

最短の実行手順（Study 1結果の集計 → Figure 2b相当の可視化）:

```bash
# 1) Study 2集計を作成
PYTHONPATH=src uv run python src/study/s2.py

# 2) 棒グラフを出力
PYTHONPATH=src uv run python src/visualization/study2_accuracy.py
```

※ `src/study/s2.py` は Study 1 の出力JSON（`output/*/*/*/temp_*.json`）を入力として使います。

既定では `output/` のStudy 1結果から、`LOW <= 0.5` と `HIGH >= 0.8` の条件を使って
`self-reflection` / `within-model` / `across-model` を実行します。

主なオプション:

```bash
PYTHONPATH=src uv run python src/study/s2.py \
  --low-max 0.5 \
  --high-min 0.8 \
  --generator-models QWEN3_CODER_30B,NOVA_MICRO \
  --predictor-models QWEN3_CODER_30B,NOVA_MICRO \
  --limit-samples 100
```

AWS上で実行できるモデルだけで実行する例:

```bash
PYTHONPATH=src uv run python src/study/s2.py \
  --generator-models NOVA_MICRO,NOVA_2_LITE,CLAUDE_CODE_HAIKU_4_5 \
  --predictor-models NOVA_MICRO,NOVA_2_LITE,CLAUDE_CODE_HAIKU_4_5
```

出力:
- 生データ: `output/study2/{self_reflection|within_model|across_model}/.../*.json`
- 集計: `output/study2/summary.csv`

### ヒートマップ可視化
Study 1の実験結果をヒートマップで可視化できます：

```bash
PYTHONPATH=src uv run python src/visualization/study1_heatmap.py
```

可視化結果は `output/figures/` ディレクトリに以下の形式で保存されます：
- `study1_heatmap.png` - PNG形式（高解像度、300dpi）
- `study1_heatmap.pdf` - PDF形式（論文用）

**可視化内容:**
- 各モデルのHIGH率（温度が高いと判定した割合）をヒートマップで表示
- 横軸: 実際の温度設定（0.0〜2.0）
- 縦軸: 実験条件（プロンプトタイプ × 対象）
- カラーマップ: 赤=HIGH率高、緑=LOW率高

### Study 2精度の可視化（Figure 2b相当）
Study 2の `summary.csv` をもとに、条件別accuracyを棒グラフで可視化できます：

```bash
PYTHONPATH=src uv run python src/visualization/study2_accuracy.py
```

可視化結果は `output/figures/` ディレクトリに以下の形式で保存されます：
- `study2_accuracy.png` - PNG形式（高解像度、300dpi）
- `study2_accuracy.pdf` - PDF形式（論文用）
