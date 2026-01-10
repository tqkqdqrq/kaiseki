---
description: motoデータをazukun形式に変換して追加する
---

# motoデータをAzukun形式に追加

`motoデータ`フォルダのCSVファイルを`20251223_duo2_azukun.csv`に追加します。

## 使い方

1. `add_data.py` の `input_file` を変換したいファイルに変更
2. 実行:

// turbo
```powershell
python add_data.py
```

## 処理内容

- **ID生成**: `月_日_Hall_Name_Machine_No` 形式
- **除外**: Start/Dedama/Status欠損、重複データ
- **追加先**: `20251223_duo2_azukun.csv`
