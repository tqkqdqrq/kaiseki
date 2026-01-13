import pandas as pd

# CSVファイルを読み込む
import shutil
import os

# CSVファイルを読み込む
input_file = r"C:\Users\ilove\Desktop\解析\jyogai20251223_duo2_azukun.csv"

# CSVを読み込む
df = pd.read_csv(input_file, encoding='cp932')

print(f"元データ件数: {len(df)}")
print(f"ユニークID数: {df['ID'].nunique()}")

# 各IDの最初の行を取得
first_rows = df.groupby('ID').first().reset_index()

# 最初のStartが1のIDリストを取得
ids_to_exclude = first_rows[first_rows['Start'] == 1]['ID'].tolist()

print(f"\n除外対象ID数 (最初のStartが1): {len(ids_to_exclude)}")
if len(ids_to_exclude) > 0:
    print("\n除外対象ID一覧:")
    for id_val in ids_to_exclude:
        count = len(df[df['ID'] == id_val])
        print(f"  - {id_val} ({count}件)")

    # 除外対象IDを除外したデータ
    df_filtered = df[~df['ID'].isin(ids_to_exclude)]

    print(f"\n除外後データ件数: {len(df_filtered)}")
    print(f"除外後ユニークID数: {df_filtered['ID'].nunique()}")

    # バックアップを作成
    backup_file = input_file + ".bak"
    shutil.copy(input_file, backup_file)
    print(f"\nバックアップを作成しました: {backup_file}")

    # 結果を上書き保存
    df_filtered.to_csv(input_file, index=False, encoding='utf-8')
    print(f"ファイルを上書き更新しました: {input_file}")
else:
    print("\n除外対象はありませんでした。ファイルは更新されません。")
