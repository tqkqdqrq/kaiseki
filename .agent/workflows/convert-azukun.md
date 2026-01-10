---
description: motoデータをazukun形式に変換して追加する
---

# motoデータをAzukun形式に変換

このワークフローは、`motoデータ`フォルダのCSVファイルを`20251223_duo2_azukun.csv`形式に変換して追加します。

## 前提条件

- 入力ファイル: `C:\Users\ilove\Desktop\解析\motoデータ\<ファイル名>.csv`
- 出力ファイル: `C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv`
- Pythonがインストールされていること

## ステップ

1. 変換したいファイルを指定して実行:

```powershell
python "C:\Users\ilove\Desktop\解析\convert_to_azukun.py" "C:\Users\ilove\Desktop\解析\motoデータ\<ファイル名>.csv"
```

例:
```powershell
# 20251226のデータを変換
python "C:\Users\ilove\Desktop\解析\convert_to_azukun.py" "C:\Users\ilove\Desktop\解析\motoデータ\20251226_duo2.csv"

# 20251227のデータを変換
python "C:\Users\ilove\Desktop\解析\convert_to_azukun.py" "C:\Users\ilove\Desktop\解析\motoデータ\20251227_duo2.csv"
```

## 処理内容

スクリプトは以下の処理を行います：

1. **ID生成**: `Date`, `Machine_URL`(dai_hall_id), `Machine_No`から `MM_DD_HallCode_MachineNo` 形式のIDを作成
2. **バグデータ除外**: `Start`, `Dedama`, `Status`が空のデータを除外
3. **二重データ除外**: 同一ファイル内および既存データとの重複を除外
4. **追加**: 新規データを既存のazukunファイルに追加

## 出力

変換結果の統計が表示されます：
- 処理行数
- 無効データ（Start/Dedama/Status欠損）の数
- ソース内重複の数
- 既存データと重複の数
- 新規追加の数
- 出力ファイル合計件数
