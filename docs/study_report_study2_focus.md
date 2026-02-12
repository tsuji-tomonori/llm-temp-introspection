# 実験レポート（Study2完了モデルフォーカス）

## 目次

- [背景](#背景)
- [目的](#目的)
- [実験の内容](#実験の内容)
- [結果](#結果)
- [考察](#考察)
- [まとめ](#まとめ)

## 背景

本プロジェクトは、LLMが生成文から温度設定（LOW/HIGH）をどの程度内省できるかを検証する追実験である。元論文の「self-reflection（自己アクセス優位）」仮説を日本語条件で再確認することに加え、今回追加した対象「アイレット・ドコドコ・ヤッタゼ・ペンギン」と、実在性の低い生成（CRAZY条件）が推定精度に与える影響を確認した。

## 目的

1. Study2が完了しているモデルに限定して、Study1可視化を再作成する。
2. Study2の条件別精度（self/within/across）を整理し、自己優位の有無を評価する。
3. 追加対象（アイレット・ドコドコ・ヤッタゼ・ペンギン）と「絶対にないもの」寄り条件（CRAZY）を実データで考察し、今後の扱い方針を定める。

## 実験の内容

### 対象モデル

- `NOVA_MICRO`（`amazon.nova-micro-v1:0`）
- `NOVA_2_LITE`（`global.amazon.nova-2-lite-v1:0`）

### データと判定

- Study1: `output/<MODEL>/<TARGET>/<PROMPT>/temp_*.json`
- Study2集計: `output/study2/summary.csv`
- 温度ラベル閾値: `LOW <= 0.5`, `HIGH >= 0.8`

### 可視化（再生成）

- Study1（対象モデル限定）
  - `output/figures/study1_heatmap_study2_models_report.png`
  - `output/figures/study1_heatmap_study2_models_report.pdf`
- Study2（色調整済み）
  - `output/figures/study2_accuracy.png`
  - `output/figures/study2_accuracy.pdf`

Study1（Study2完了モデル限定）:

![Study1 heatmap (study2 models)](../output/figures/study1_heatmap_study2_models_report.png)

Study2 条件別精度:

![Study2 accuracy](../output/figures/study2_accuracy.png)

## 結果

### 1) Study2 条件別精度

`output/study2/summary.csv` より:

| predictor_model | self_reflection | within_model | across_model |
| --- | ---: | ---: | ---: |
| amazon.nova-micro-v1:0 | 0.548 (n=405) | 0.494 (n=401) | 0.434 (n=76) |
| global.amazon.nova-2-lite-v1:0 | 0.481 (n=77) | 0.675 (n=77) | 0.531 (n=405) |

図（Study2条件別精度）:

![Study2 accuracy (results)](../output/figures/study2_accuracy.png)

### 2) Study1（対象モデル限定）の全体傾向

- `NOVA_MICRO`: 全体精度 0.548（n=405）
- `NOVA_2_LITE`: 全体精度 0.481（n=77）

Prompt別（Study1閾値判定ベース）:

| model | FACTUAL | NORMAL | CRAZY |
| --- | ---: | ---: | ---: |
| NOVA_MICRO | 0.681 (n=135) | 0.615 (n=135) | 0.348 (n=135) |
| NOVA_2_LITE | 1.000 (n=30) | 0.233 (n=30) | 0.000 (n=17) |

温度カバレッジ補足:

- `NOVA_MICRO` は 0.0〜1.0（0.1刻み）を実施済み。
- `NOVA_2_LITE` は多くが 0.0/0.1 のみ、`CRAZY` の一部は 0.0 のみで、途中段階データを含む。

図（Study1ヒートマップ、対象モデル限定）:

![Study1 heatmap (results)](../output/figures/study1_heatmap_study2_models_report.png)

### 3) 追加対象「アイレット・ドコドコ・ヤッタゼ・ペンギン」の挙動

Study1（対象ターゲット限定、閾値判定精度）:

| model | FACTUAL | NORMAL | CRAZY |
| --- | ---: | ---: | ---: |
| NOVA_MICRO | 0.704 (n=27) | 0.407 (n=27) | 0.333 (n=27) |
| NOVA_2_LITE | 1.000 (n=6) | 0.000 (n=6) | 0.000 (n=3) |

代表的な実データ:

- `output/NOVA_MICRO/IRET_DOKODOKO_YATTAZE_PENGUIN/FACTUAL/temp_0.0_loop_0.json`: 「実在しません」→ `LOW`（妥当）
- `output/NOVA_MICRO/IRET_DOKODOKO_YATTAZE_PENGUIN/CRAZY/temp_0.0_loop_0.json`: 奇抜文を根拠に `HIGH`（低温でもHIGH寄り）
- `output/NOVA_2_LITE/IRET_DOKODOKO_YATTAZE_PENGUIN/NORMAL/temp_0.0_loop_0.json`: 通常条件でも `HIGH`

## 考察

### 1) self優位はモデル依存で一貫しない

- `NOVA_MICRO` は self > within > across で自己優位寄り。
- `NOVA_2_LITE` は within が最良で self を上回る。
- よって「privileged self-access」が常に成立するとは言いにくく、モデル固有の推定戦略差が大きい。

### 2) 「絶対にないもの」寄り条件（CRAZY）は温度推定を歪める

- 両モデルで CRAZY 条件のHIGH判定率がほぼ飽和し、低温でもHIGHに寄る。
- その結果、CRAZY の閾値精度が大きく悪化（例: `NOVA_2_LITE` で 0.000）。
- これは「温度」より「文体の奇抜さ」を手がかりにしている可能性を示す。

### 3) 追加対象（アイレット・ドコドコ・ヤッタゼ・ペンギン）の影響

- 対象そのものの未知性より、`NORMAL/CRAZY` の文体バイアスとの組み合わせで誤判定が増える傾向。
- `FACTUAL` ではむしろ安定して正解するケースがあり、未知語だけが主因ではない。

### 4) 「未知語・実在しない対象（ご要望の“道”に相当する扱い）」の運用方針

温度推定実験としては、次の扱いが妥当:

1. 主評価セットは「事実対象 + 中立文体（FACTUAL/NORMAL中心）」で構成する。
2. CRAZY は別枠のストレス評価として分離報告する。
3. 未知対象は「実在/非実在フラグ」を付与し、既知対象と分けて集計する。
4. 指標は accuracy 単独でなく、条件別サンプル数と信頼区間を併記する。
5. 可能なら判断根拠テキストを分類し、「文体依存」と「温度依存」を切り分ける。

## まとめ

- Study2完了モデルに限定した再可視化を実施し、比較対象を統一した。
- 結果として、自己優位はモデル依存で一貫せず、`within-model` 優位なモデルも確認された。
- 追加対象「アイレット・ドコドコ・ヤッタゼ・ペンギン」および CRAZY 条件は、温度推定より文体バイアスを強く誘発する。
- 今後は「主評価（中立条件）」と「ストレス評価（CRAZY/未知対象）」を分離する設計が必要である。
