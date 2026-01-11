---
description: データの追加・整形・除外フィルタリングの一連フロー
---

# データ処理ワークフロー

データの追加から、クリーニング（重複・不正値排除）、解析用データの抽出（除外フィルタ）までの一連の流れです。

## 1. データの追加

1. `add_data.py` を開き、 `input_file` 変数を追加したいCSVファイルのパスに変更して保存してください。
2. 以下のコマンドを実行してデータを追加します。

```powershell
python add_data.py
```

## 2. データのチェックと整形

データの重複削除、不正値除外、`Start=0` の修正を行います。

まずチェックのみ実行：
// turbo
```powershell
python check_data.py check
```

問題があれば整形を実行（上書き保存されます）：
// turbo
```powershell
python check_data.py clean
```

## 3. 解析用データの抽出（除外フィルタ）

`Start + 出玉/4 < 2000` の除外条件を適用し、 `jyogai...csv` を生成します。

// turbo
```powershell
python filter_duo_data.py
```
