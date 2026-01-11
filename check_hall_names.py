import pandas as pd
from difflib import SequenceMatcher
import unicodedata

# ファイル読み込み
code_unique = pd.read_csv(r'C:\Users\ilove\Desktop\解析\code_unique.csv', header=None, names=['ホールコード'])
online_halls = pd.read_csv(r'C:\Users\ilove\Desktop\解析\データオンライン地域 - 対応可能ホール.csv', encoding='utf-8')
duo2 = pd.read_csv(r'C:\Users\ilove\Desktop\解析\20251223_duo2.csv', encoding='utf-8')

# code_unique.csvのホールコードでオンラインホールをフィルタリング
codes_list = code_unique['ホールコード'].astype(str).tolist()
filtered_online_halls = online_halls[online_halls['ホールコード'].astype(str).isin(codes_list)]

# 対象ホール名リスト
target_hall_names = filtered_online_halls['ホール名'].tolist()
print(f"code_unique.csvから抽出したホール数: {len(target_hall_names)}")

# duo2のユニークなホール名
duo2_hall_names = duo2['Hall_Name'].unique().tolist()
print(f"duo2のユニークホール数: {len(duo2_hall_names)}")

# 文字列正規化関数
def normalize_name(name):
    """全角・半角・大文字小文字を正規化"""
    name = str(name)
    # Unicode正規化 (NFKC: 全角→半角、互換文字の統一)
    name = unicodedata.normalize('NFKC', name)
    # 小文字に統一
    name = name.lower()
    # スペース削除
    name = name.replace(' ', '').replace('　', '')
    return name

# 類似度計算関数
def similarity_ratio(a, b):
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()

# マッチング結果を格納
results = []

for target_name in target_hall_names:
    target_normalized = normalize_name(target_name)
    
    for duo2_name in duo2_hall_names:
        duo2_normalized = normalize_name(duo2_name)
        
        # 完全一致
        if target_normalized == duo2_normalized:
            results.append({
                'オンラインホール名': target_name,
                'duo2ホール名': duo2_name,
                'マッチタイプ': '完全一致',
                '類似度': 1.0
            })
        else:
            # 部分一致（片方がもう片方を含む）
            if target_normalized in duo2_normalized or duo2_normalized in target_normalized:
                sim = similarity_ratio(target_name, duo2_name)
                results.append({
                    'オンラインホール名': target_name,
                    'duo2ホール名': duo2_name,
                    'マッチタイプ': '部分一致',
                    '類似度': round(sim, 3)
                })
            else:
                # 類似度が高い場合（0.7以上）
                sim = similarity_ratio(target_name, duo2_name)
                if sim >= 0.6:
                    results.append({
                        'オンラインホール名': target_name,
                        'duo2ホール名': duo2_name,
                        'マッチタイプ': '類似',
                        '類似度': round(sim, 3)
                    })

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# 類似度で降順ソート
results_df = results_df.sort_values(by=['マッチタイプ', '類似度'], ascending=[True, False])

# 重複削除（同じオンラインホールに対して複数のマッチがある場合は最も類似度が高いものを残す）
results_df = results_df.drop_duplicates(subset=['オンラインホール名', 'duo2ホール名'])

# CSVに出力
output_path = r'C:\Users\ilove\Desktop\解析\matching_halls.csv'
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n結果を {output_path} に保存しました")
print(f"マッチング結果: {len(results_df)} 件")
print("\nマッチタイプ別件数:")
print(results_df['マッチタイプ'].value_counts())
print("\n最初の20件:")
print(results_df.head(20).to_string())
