# ADR-001: Study 1結果のヒートマップ可視化

## ステータス
提案中

## コンテキスト

論文 "Privileged Self-Access Matters for Introspection in AI" (arXiv:2508.14802v1) のFigure 2(a)と同様の可視化を、本プロジェクトの実験結果データから作成する必要がある。

### 参照する図の仕様（Figure 2a）
- **グラフタイプ**: ヒートマップ（モデル別サブプロット）
- **X軸**: 実際の温度（0.0〜2.0、論文では0.1刻み）
- **Y軸**: プロンプト条件（prompt_type × target の組み合わせ、9行）
  - factual - elephants, factual - unicorns, factual - murlocs
  - normal - elephants, normal - unicorns, normal - murlocs
  - crazy - elephants, crazy - unicorns, crazy - murlocs
- **色**: 予測された温度（HIGH率: 0.0〜1.0）
- **レイアウト**: 2x2グリッドで4モデル表示

### 本プロジェクトのデータ構造

```
output/
├── GEMMA_3N_E4B/
├── IBM_GRANITE4_TINY/
├── MAGISTRAL_SAMLL/
├── NOVA_MICRO/
└── QWEN3_CODER_30B/
    ├── ELEPHANT/
    ├── ELEPHANT_KANA/
    ├── IRET_DOKODOKO_YATTAZE_PENGUIN/
    ├── MURLOC/
    └── UNICORN/
        ├── FACTUAL/
        ├── NORMAL/
        └── CRAZY/
            └── temp_{0.0-1.0}_loop_{0-2}.json
```

### JSONデータスキーマ
```json
{
  "condition": {
    "model_id": "google/gemma-3n-e4b",
    "temperature": 0.3,
    "prompt_type": "事実に基づいた",  // 日本語
    "target": "像"                    // 日本語
  },
  "response": {
    "generated_sentence": "...",
    "reasoning": "...",
    "judgment": "LOW"  // or "HIGH"
  },
  "loop_times": 2,
  "unique_id": "...",
  "procession_time_ms": 19381,
  "created_at": "..."
}
```

### 本プロジェクトの特徴（論文との差異）
| 項目 | 論文 | 本プロジェクト |
|------|------|----------------|
| 温度範囲 | 0.0〜2.0 | 0.0〜1.0 |
| モデル数 | 4 (GPT-4o, GPT-4.1, Gemini-2.0/2.5-flash) | 5 (Gemma, Granite, Magistral, Nova, Qwen) |
| Target | 3 (elephants, unicorns, murlocs) | 5 (ELEPHANT, ELEPHANT_KANA, IRET..., MURLOC, UNICORN) |
| 各条件の試行回数 | 3 | 3 (loop_times: 0, 1, 2) |
| 言語 | 英語 | 日本語 |

## 決定事項

### 1. 実装アプローチ

**Pythonスクリプト** (`src/visualization/study1_heatmap.py`) として実装する。

### 2. 使用ライブラリ
```
matplotlib >= 3.7.0    # ヒートマップ描画
seaborn >= 0.12.0      # ヒートマップスタイリング
pandas >= 2.0.0        # データ集計
numpy >= 1.24.0        # 数値計算
```

### 3. データ処理フロー

```
1. データ読み込み
   output/{MODEL}/{TARGET}/{PROMPT_TYPE}/temp_{T}_loop_{N}.json

2. 集計
   - グループキー: (model, target, prompt_type, temperature)
   - 集計値: HIGH率 = count(judgment=="HIGH") / total_count

3. ピボット変換
   - 行: "{prompt_type} - {target}" (例: "FACTUAL - ELEPHANT")
   - 列: temperature (0.0, 0.1, ..., 1.0)
   - 値: HIGH率 (0.0〜1.0)

4. 可視化
   - seaborn.heatmap() でヒートマップ描画
   - matplotlib.pyplot.subplots() でモデル別サブプロット
```

### 4. 出力仕様

| 項目 | 仕様 |
|------|------|
| ファイル形式 | PNG, PDF |
| 出力先 | `output/figures/study1_heatmap.{png,pdf}` |
| 解像度 | 300 DPI |
| サイズ | 16 x 12 インチ（調整可能） |
| カラーマップ | `RdYlGn_r` または `coolwarm`（低=青/緑、高=赤） |

### 5. サブプロットレイアウト

5モデルの場合、以下のレイアウトを検討：
- **案A**: 2x3グリッド（1つ空き）
- **案B**: 3x2グリッド（1つ空き）
- **案C**: 1x5横並び（幅広）

→ **案A (2x3)** を採用。論文のスタイルに近く、視認性が良い。

### 6. Y軸ラベルの順序

論文に合わせ、以下の順序で表示：
1. FACTUAL - ELEPHANT
2. FACTUAL - UNICORN
3. FACTUAL - MURLOC
4. NORMAL - ELEPHANT
5. NORMAL - UNICORN
6. NORMAL - MURLOC
7. CRAZY - ELEPHANT
8. CRAZY - UNICORN
9. CRAZY - MURLOC

※ ELEPHANT_KANA, IRET_DOKODOKO_YATTAZE_PENGUIN は追加行として表示

### 7. 欠損データの扱い

- 一部の温度/条件でデータが欠損している場合、そのセルはグレー表示
- 集計時に有効な回答（HIGH/LOW）のみカウント

## 実装計画

### ステップ1: データローダー作成
```python
def load_study1_data(output_dir: Path) -> pd.DataFrame:
    """全JSONファイルを読み込みDataFrameに変換"""
    pass
```

### ステップ2: 集計関数作成
```python
def aggregate_high_rate(df: pd.DataFrame) -> pd.DataFrame:
    """HIGH率を計算してピボットテーブル作成"""
    pass
```

### ステップ3: 可視化関数作成
```python
def plot_study1_heatmap(
    data: dict[str, pd.DataFrame],  # model_name -> pivot_table
    output_path: Path,
    figsize: tuple = (16, 12),
    cmap: str = "RdYlGn_r"
) -> None:
    """Figure 2(a)スタイルのヒートマップを描画"""
    pass
```

### ステップ4: メインスクリプト
```python
def main():
    output_dir = Path("output")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    df = load_study1_data(output_dir)
    pivots = aggregate_high_rate(df)
    plot_study1_heatmap(pivots, figures_dir / "study1_heatmap")
```

## 代替案

### 代替案1: Jupyter Notebook での実装
- **メリット**: インタラクティブな調整が容易
- **デメリット**: 再現性・自動化に不向き
- **却下理由**: CI/CDパイプラインでの自動生成を想定

### 代替案2: Plotly による対話的可視化
- **メリット**: ズーム・ホバー等の対話機能
- **デメリット**: 静的PDF出力時の品質、依存関係増加
- **却下理由**: 論文投稿用の静的図として十分

### 代替案3: R + ggplot2
- **メリット**: 学術論文での標準的なツール
- **デメリット**: プロジェクトの言語統一性が損なわれる
- **却下理由**: 既存のPython環境を活用

## 影響

- `pyproject.toml` に `matplotlib`, `seaborn`, `pandas` の依存追加が必要
- `src/visualization/` ディレクトリの新規作成

## 参考資料

- 論文: arXiv:2508.14802v1 Figure 2(a)
- seaborn heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html
- matplotlib subplots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
