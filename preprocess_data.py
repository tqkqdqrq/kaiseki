"""
デュオ解析 - データ前処理スクリプト
大きなCSVを事前に解析してキャッシュファイルに保存
"""

import pandas as pd
import pickle
from datetime import datetime

# ===== 設定 =====
HEAVEN_THRESHOLD = 35
COIN_HOLD = 25.3
INPUT_FILE = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'
CACHE_FILE = r'C:\Users\ilove\Desktop\解析\duo_cache.pkl'


def load_data(filepath):
    """azukunフォーマットCSVを読み込み"""
    print(f"Loading: {filepath}")
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except:
        df = pd.read_csv(filepath, encoding='cp932')
    
    df['Original_Order'] = range(len(df))
    
    id_parts = df['ID'].str.rsplit('_', n=2, expand=True)
    df['Hall_Name'] = id_parts[0]
    df['Machine_No'] = pd.to_numeric(id_parts[1], errors='coerce').fillna(0).astype(int)
    df['Date'] = id_parts[2].str.replace(r'(\d{4})(\d{2})(\d{2})', r'\1-\2-\3', regex=True)
    
    df['Start'] = pd.to_numeric(df['Start'], errors='coerce').fillna(0).astype(int)
    df['Dedama'] = pd.to_numeric(df['Dedama'], errors='coerce').fillna(0).astype(int)
    
    df_sorted = df.sort_values(by=['Hall_Name', 'Date', 'Machine_No', 'Time'])
    df_sorted['Count'] = df_sorted.groupby(['Hall_Name', 'Date', 'Machine_No']).cumcount() + 1
    df = df_sorted.sort_values(by='Original_Order').reset_index(drop=True)
    
    print(f"  Loaded: {len(df):,} rows")
    return df


def analyze_chains(df, heaven_threshold=35):
    """連チャン判定と天国移行率を計算"""
    print("Analyzing chains...")
    df = df.sort_values(by='Original_Order').reset_index(drop=True)
    
    all_chains = []
    chain_id = 0
    chain_number_per_id = {}
    
    df['Chain_ID'] = 0
    df['Chain_Position'] = 0
    
    current_id = None
    current_chain_hits = []
    prev_chain_len = 0
    
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        if idx % 50000 == 0:
            print(f"  Processing: {idx:,}/{total_rows:,} ({100*idx/total_rows:.1f}%)")
        
        row_id = row['ID']
        start_g = row['Start']
        
        if current_id != row_id or start_g > heaven_threshold:
            if current_chain_hits:
                chain_id += 1
                chain_len = len(current_chain_hits)
                is_heaven = chain_len >= 2
                
                for pos, hit_idx in enumerate(current_chain_hits, start=1):
                    df.at[hit_idx, 'Chain_ID'] = chain_id
                    df.at[hit_idx, 'Chain_Position'] = pos
                
                first_start = df.at[current_chain_hits[0], 'Start']
                total_dedama = df.loc[current_chain_hits, 'Dedama'].sum()
                total_invest = df.loc[current_chain_hits, 'Start'].sum() / COIN_HOLD * 50
                net_diff = total_dedama - total_invest
                last_dedama = df.at[current_chain_hits[-1], 'Dedama']
                first_invest = first_start / COIN_HOLD * 50
                special_judge = net_diff - last_dedama + first_invest
                
                hit_games = df.loc[current_chain_hits, 'Start'].tolist()
                hit_statuses = df.loc[current_chain_hits, 'Status'].tolist()
                
                machine_id = df.at[current_chain_hits[0], 'ID']
                if machine_id not in chain_number_per_id:
                    chain_number_per_id[machine_id] = 0
                chain_number_per_id[machine_id] += 1
                
                all_chains.append({
                    'ID': machine_id,
                    'Chain_ID': chain_id,
                    'Chain_Number': chain_number_per_id[machine_id],
                    'Chain_Length': chain_len,
                    'Hit_Games': hit_games,
                    'Hit_Statuses': hit_statuses,
                    'Is_Heaven': is_heaven,
                    'First_G': first_start,
                    'Through_Before': 0,
                    'Prev_Chain_Length': prev_chain_len,
                    'Total_Dedama': total_dedama,
                    'Total_Invest': total_invest,
                    'Net_Diff': net_diff,
                    'Special_Judge': special_judge,
                    'Prev_Special_Judge': 0.0
                })
                
                if is_heaven:
                    prev_chain_len = chain_len
            
            if current_id != row_id:
                prev_chain_len = 0
            
            current_id = row_id
            current_chain_hits = [idx]
        else:
            current_chain_hits.append(idx)
    
    # 最後のチェーン
    if current_chain_hits:
        chain_id += 1
        chain_len = len(current_chain_hits)
        is_heaven = chain_len >= 2
        
        for pos, hit_idx in enumerate(current_chain_hits, start=1):
            df.at[hit_idx, 'Chain_ID'] = chain_id
            df.at[hit_idx, 'Chain_Position'] = pos
        
        first_start = df.at[current_chain_hits[0], 'Start']
        total_dedama = df.loc[current_chain_hits, 'Dedama'].sum()
        total_invest = df.loc[current_chain_hits, 'Start'].sum() / COIN_HOLD * 50
        net_diff = total_dedama - total_invest
        last_dedama = df.at[current_chain_hits[-1], 'Dedama']
        first_invest = first_start / COIN_HOLD * 50
        special_judge = net_diff - last_dedama + first_invest
        
        hit_games = df.loc[current_chain_hits, 'Start'].tolist()
        hit_statuses = df.loc[current_chain_hits, 'Status'].tolist()
        
        machine_id = df.at[current_chain_hits[0], 'ID']
        if machine_id not in chain_number_per_id:
            chain_number_per_id[machine_id] = 0
        chain_number_per_id[machine_id] += 1
        
        all_chains.append({
            'ID': machine_id,
            'Chain_ID': chain_id,
            'Chain_Number': chain_number_per_id[machine_id],
            'Chain_Length': chain_len,
            'Hit_Games': hit_games,
            'Hit_Statuses': hit_statuses,
            'Is_Heaven': is_heaven,
            'First_G': first_start,
            'Through_Before': 0,
            'Prev_Chain_Length': prev_chain_len,
            'Total_Dedama': total_dedama,
            'Total_Invest': total_invest,
            'Net_Diff': net_diff,
            'Special_Judge': special_judge,
            'Prev_Special_Judge': 0.0
        })
    
    chains_df = pd.DataFrame(all_chains)
    print(f"  Created: {len(chains_df):,} chains")
    
    # Through_Before, Prev_Chain_Length 計算
    if not chains_df.empty:
        print("  Calculating through counts...")
        chains_df['Through_Before'] = 0
        chains_df['Prev_Chain_Length'] = 0
        chains_df['Prev_Special_Judge'] = 0.0
        
        prev_id = None
        through_count = 0
        prev_heaven_len = 0
        prev_special = 0.0
        
        for idx in chains_df.index:
            current_id = chains_df.at[idx, 'ID']
            is_heaven = chains_df.at[idx, 'Is_Heaven']
            chain_len = chains_df.at[idx, 'Chain_Length']
            special_judge = chains_df.at[idx, 'Special_Judge']
            
            if current_id != prev_id:
                through_count = 0
                prev_heaven_len = 0
                prev_special = 0.0
            
            chains_df.at[idx, 'Through_Before'] = through_count
            chains_df.at[idx, 'Prev_Chain_Length'] = prev_heaven_len
            chains_df.at[idx, 'Prev_Special_Judge'] = prev_special
            
            if is_heaven:
                through_count = 0
                prev_heaven_len = chain_len
            else:
                through_count += 1
            
            prev_special = special_judge
            prev_id = current_id
    
    print("  Done!")
    return df, chains_df


def main():
    print("=" * 50)
    print("デュオ解析 - 前処理スクリプト")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # データ読み込み
    df = load_data(INPUT_FILE)
    
    # 解析
    df, chains_df = analyze_chains(df, HEAVEN_THRESHOLD)
    
    # キャッシュに保存
    print(f"\nSaving cache to: {CACHE_FILE}")
    cache_data = {
        'df': df,
        'chains_df': chains_df,
        'heaven_threshold': HEAVEN_THRESHOLD,
        'created_at': datetime.now().isoformat()
    }
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n完了! (処理時間: {elapsed:.1f}秒)")
    print(f"キャッシュファイル: {CACHE_FILE}")
    print(f"\n次回からダッシュボードでこのキャッシュを読み込めば数秒で起動します")


if __name__ == "__main__":
    main()
