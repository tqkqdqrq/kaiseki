---
description: azukunファイルの重複・バグデータをチェック＆削除
---

# データチェック＆クリーンアップ

`20251223_duo2_azukun.csv`の重複・バグデータをチェック・削除します。

## チェックのみ

// turbo
```powershell
python check_data.py check
```

## チェック後に削除

// turbo
```powershell
python check_data.py clean
```

## チェック内容

- **バグデータ**: Start/Dedama/Status欠損、Statusが BB/RB 以外
- **重複データ**: ID+Status+Start+Dedama+Time が同じ行
