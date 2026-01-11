import pandas as pd
import os

INPUT_FILE = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'
OUTPUT_FILE = r'C:\Users\ilove\Desktop\解析\jyogai20251223_duo2_azukun.csv'

def main():
    print(f"Reading from {INPUT_FILE}...")
    try:
        # encodingはcp932を試行、失敗ならutf-8
        df = pd.read_csv(INPUT_FILE, encoding='cp932')
        print("Encoding: cp932")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        print("Encoding: utf-8")

    print(f"Total rows: {len(df)}")

    # IDごとに集計
    # StartとDedamaの合計を計算
    grouped = df.groupby('ID').agg({
        'Start': 'sum',
        'Dedama': 'sum'
    })

    # 条件計算
    # Startの合計 + Dedama合計/4
    # 条件: 2000未満を除外 (>= 2000 を残す)
    grouped['Criteria'] = grouped['Start'] + (grouped['Dedama'] / 4.0)
    
    valid_ids = grouped[grouped['Criteria'] >= 2000].index
    
    print(f"Total IDs: {len(grouped)}")
    print(f"Valid IDs: {len(valid_ids)} (Criteria >= 2000)")
    print(f"Excluded IDs: {len(grouped) - len(valid_ids)}")
    
    # フィルタリング
    filtered_df = df[df['ID'].isin(valid_ids)]
    
    print(f"Filtered rows: {len(filtered_df)}")
    
    # 出力
    filtered_df.to_csv(OUTPUT_FILE, index=False, encoding='cp932')
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
