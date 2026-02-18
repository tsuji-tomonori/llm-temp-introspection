# 追実験A（Info+/Info−）・追実験D（Blind/Wrong-label）使い方

## 概要

| 実験 | 目的 | 操作変数 |
|---|---|---|
| 追実験A | 情報密度だけで温度判定が動くかを検証 | 同一文のInfo+（詳細版）/Info−（簡略版） |
| 追実験D | ラベル（prompt_type/target）への依存度を定量化 | Blind（ラベル非開示）/ Wrong-label（ラベル入替） |

## 前提条件

- Study 1 の結果が `output/` 配下に存在すること（`output/{MODEL}/{TARGET}/{PROMPT_TYPE}/temp_*.json`）
- Study 2 の within_model 結果が `output/study2/` に存在すること（追実験D分析で Full 条件として使用）

## ディレクトリ構成

```
output/
├── experiment_a/
│   ├── edited/{generator_model}/{source_unique_id}.json
│   └── predictions/{info_plus|info_minus}/{generator_model}/{predictor_model}/{source_unique_id}.json
├── experiment_d/
│   ├── blind/{generator_model}/{predictor_model}/{source_unique_id}.json
│   └── wrong_label/{generator_model}/{predictor_model}/{source_unique_id}.json
├── analysis/
│   ├── experiment_a_p_high_delta.csv
│   ├── experiment_d_accuracy_by_label_condition.csv
│   └── experiment_d_wrong_label_shift.csv
└── figures/
    ├── experiment_a_delta.{png,pdf}
    ├── experiment_d_accuracy.{png,pdf}
    └── experiment_d_shift.{png,pdf}
```

## 実行手順

すべてのコマンドは `src/` ディレクトリから実行する。

```bash
cd src
```

### 1. 追実験D（Blind/Wrong-label）

```bash
# dry run（5サンプル）
python -m study.experiment_d --limit-samples 5

# 全データ実行
python -m study.experiment_d

# モデルを限定する場合
python -m study.experiment_d \
  --generator-models NOVA_2_LITE,NOVA_MICRO \
  --predictor-models NOVA_2_LITE,NOVA_MICRO
```

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--study1-output-dir` | `output/` | Study 1 結果ディレクトリ |
| `--output-dir` | `output/experiment_d/` | 出力先 |
| `--low-max` | `0.5` | LOW ラベル閾値（temperature <= この値） |
| `--high-min` | `0.8` | HIGH ラベル閾値（temperature >= この値） |
| `--generator-models` | 全モデル | カンマ区切りのモデル enum 名 |
| `--predictor-models` | generator と同一 | カンマ区切りのモデル enum 名 |
| `--limit-samples` | なし | 先頭 N サンプルのみ使用 |
| `--force` | `false` | 既存ファイルを上書き |

### 2. 追実験A（Info+/Info−）

追実験Aは2段階で動作する:
1. **編集ステップ**: Study 1 の NORMAL プロンプトサンプルから Info+/Info− ペアを生成
2. **予測ステップ**: 編集済みペアに対して各モデルで温度予測

```bash
# dry run（2サンプル）
python -m study.experiment_a --limit-samples 2 --editor-model NOVA_2_LITE

# 全データ実行
python -m study.experiment_a --editor-model NOVA_2_LITE

# 予測のみ再実行（編集済みペアを再利用）
python -m study.experiment_a --skip-edit
```

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--editor-model` | `NOVA_2_LITE` | 文編集に使うモデル（temp=0.0 固定） |
| `--skip-edit` | `false` | 編集ステップをスキップし予測のみ実行 |
| `--generator-models` | 全モデル | カンマ区切りのモデル enum 名 |
| `--predictor-models` | generator と同一 | カンマ区切りのモデル enum 名 |
| `--limit-samples` | なし | 先頭 N サンプルのみ使用 |
| `--force` | `false` | 既存ファイルを上書き |

### 3. 分析

```bash
# 追実験D分析
python -m analysis.experiment_d_analysis

# 追実験A分析
python -m analysis.experiment_a_analysis
```

分析オプション（共通パターン）:

| オプション | 説明 |
|---|---|
| `--experiment-d-output-dir` / `--experiment-a-output-dir` | 実験出力ディレクトリ |
| `--study2-output-dir`（Dのみ） | Study 2 結果ディレクトリ（Full 条件用） |
| `--analysis-output-dir` | CSV 出力先（デフォルト: `output/analysis/`） |

### 4. 可視化

```bash
# 追実験A: フォレストプロット
python -m visualization.experiment_a_plot

# 追実験D: balanced accuracy 棒グラフ + wrong-label shift 図
python -m visualization.experiment_d_plot
```

## 出力 CSV の説明

### `experiment_a_p_high_delta.csv`

| カラム | 説明 |
|---|---|
| `predictor_model` | 予測モデル |
| `p_high_info_plus` | Info+ に対する P(HIGH) |
| `p_high_info_minus` | Info− に対する P(HIGH) |
| `delta` | P(HIGH\|Info+) − P(HIGH\|Info−) |
| `ci_lower` / `ci_upper` | Bootstrap 95% CI |
| `n_pairs` | ペア数 |

### `experiment_d_accuracy_by_label_condition.csv`

| カラム | 説明 |
|---|---|
| `predictor_model` | 予測モデル |
| `label_condition` | `full` / `blind` / `wrong_label` |
| `accuracy` / `balanced_accuracy` / `macro_f1` | 各メトリクス |
| `majority_baseline` | 多数決ベースライン |
| `n_samples` | サンプル数 |

### `experiment_d_wrong_label_shift.csv`

| カラム | 説明 |
|---|---|
| `predictor_model` | 予測モデル |
| `swap_direction` | `FACTUAL→CRAZY` または `CRAZY→FACTUAL` |
| `p_high_full` / `p_high_wrong_label` | 各条件の P(HIGH) |
| `delta_p_high` | P(HIGH) の変化量 |
| `ci_lower` / `ci_upper` | Bootstrap 95% CI |

## skip-existing の動作

すべてのランナーは `--force` を指定しない限り既存ファイルをスキップする。
再実行時にログに `saved=0, skipped=N` と表示されればスキップが正しく動作している。

## Wrong-label の swap ルール

| 元の prompt_type | swap 先 |
|---|---|
| FACTUAL（事実に基づいた） | CRAZY（クレイジーな） |
| CRAZY（クレイジーな） | FACTUAL（事実に基づいた） |
| NORMAL（空文字列） | swap しない（実験から除外） |
