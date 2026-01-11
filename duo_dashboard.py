"""
デュオ天国移行率分析 - Streamlit ダッシュボード
インタラクティブな解析ツール
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="デュオ天国移行率分析",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== 設定 =====
HEAVEN_THRESHOLD = 35  # 天国連チャンとみなすG数
COIN_HOLD = 25.3  # コイン持ち (G/50枚)
DEFAULT_INPUT_FILE = r'C:\Users\ilove\Desktop\解析\20251223_duo2_azukun.csv'
CACHE_FILE = r'C:\Users\ilove\Desktop\解析\duo_cache.pkl'

import os
import pickle


# ===== キャッシュ読み込み =====
@st.cache_data
def load_cache():
    """キャッシュファイルから解析済みデータを読み込み（高速）"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['df'], cache_data['chains_df'], cache_data.get('created_at', 'unknown')
    return None, None, None


# ===== データ読み込み・キャッシュ =====
@st.cache_data
def load_data(filepath):
    """azukunフォーマットCSVを読み込み"""
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
    
    return df


@st.cache_data
def analyze_chains(df, heaven_threshold=35):
    """連チャン判定と天国移行率を計算"""
    df = df.sort_values(by='Original_Order').reset_index(drop=True)
    
    all_chains = []
    chain_id = 0
    chain_number_per_id = {}
    
    df['Chain_ID'] = 0
    df['Chain_Position'] = 0
    
    current_id = None
    current_chain_hits = []
    prev_chain_len = 0
    
    for idx, row in df.iterrows():
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
    
    # Through_Before, Prev_Chain_Length 計算
    if not chains_df.empty:
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
    
    return df, chains_df


# ===== 分析関数 =====
def calculate_heaven_rate_by_through(chains_df, max_through=10):
    """スルー回数別の天国移行率を計算"""
    results = []
    
    for thr in range(max_through + 1):
        subset = chains_df[chains_df['Through_Before'] == thr]
        total = len(subset)
        heaven = len(subset[subset['Is_Heaven'] == True])
        rate = heaven / total if total > 0 else 0
        
        results.append({
            'スルー回数': f'{thr}スルー',
            '天国移行数': heaven,
            'サンプル数': total,
            '天国移行率': rate
        })
    
    # 11スルー以上
    subset = chains_df[chains_df['Through_Before'] >= max_through + 1]
    total = len(subset)
    heaven = len(subset[subset['Is_Heaven'] == True])
    rate = heaven / total if total > 0 else 0
    
    results.append({
        'スルー回数': f'{max_through + 1}スルー以上',
        '天国移行数': heaven,
        'サンプル数': total,
        '天国移行率': rate
    })
    
    # 合計
    total_all = len(chains_df)
    heaven_all = len(chains_df[chains_df['Is_Heaven'] == True])
    rate_all = heaven_all / total_all if total_all > 0 else 0
    
    results.append({
        'スルー回数': '合計',
        '天国移行数': heaven_all,
        'サンプル数': total_all,
        '天国移行率': rate_all
    })
    
    return pd.DataFrame(results)


def calculate_heaven_rate_by_chain(chains_df, min_chain=2, max_chain=20):
    """前回連チャン長別の天国移行率を計算"""
    results = []
    
    for chain_len in range(min_chain, max_chain):
        subset = chains_df[chains_df['Prev_Chain_Length'] == chain_len]
        total = len(subset)
        heaven = len(subset[subset['Is_Heaven'] == True])
        rate = heaven / total if total > 0 else 0
        
        results.append({
            '前回連チャン': f'{chain_len}連後',
            '天国移行数': heaven,
            'サンプル数': total,
            '天国移行率': rate
        })
    
    # 20連以上
    subset = chains_df[chains_df['Prev_Chain_Length'] >= max_chain]
    total = len(subset)
    heaven = len(subset[subset['Is_Heaven'] == True])
    rate = heaven / total if total > 0 else 0
    
    results.append({
        '前回連チャン': f'{max_chain}連以上後',
        '天国移行数': heaven,
        'サンプル数': total,
        '天国移行率': rate
    })
    
    # 合計
    subset_all = chains_df[chains_df['Prev_Chain_Length'] >= min_chain]
    total_all = len(subset_all)
    heaven_all = len(subset_all[subset_all['Is_Heaven'] == True])
    rate_all = heaven_all / total_all if total_all > 0 else 0
    
    results.append({
        '前回連チャン': '合計',
        '天国移行数': heaven_all,
        'サンプル数': total_all,
        '天国移行率': rate_all
    })
    
    return pd.DataFrame(results)


def calculate_crosstab(chains_df, prev_chain_filter=None, through_filter=None):
    """前回連チャン長 × スルー回数 のクロス集計"""
    # 朝イチを除外
    filtered = chains_df[chains_df['Chain_Number'] > 1].copy()
    
    if prev_chain_filter:
        filtered = filtered[filtered['Prev_Chain_Length'].isin(prev_chain_filter)]
    
    if through_filter is not None:
        filtered = filtered[filtered['Through_Before'].isin(through_filter)]
    
    # クロス集計
    pivot_rate = pd.pivot_table(
        filtered,
        values='Is_Heaven',
        index='Prev_Chain_Length',
        columns='Through_Before',
        aggfunc='mean',
        fill_value=None
    )
    
    pivot_count = pd.pivot_table(
        filtered,
        values='Is_Heaven',
        index='Prev_Chain_Length',
        columns='Through_Before',
        aggfunc='count',
        fill_value=0
    )
    
    return pivot_rate, pivot_count


def calculate_3chain_rule_analysis(chains_df, loc_filter=None, relative_through_range=10):
    """
    3連基準2連スルー分析
    - 3連以上を「真の天国」とする
    - 2連は「隠れスルー」として2回分のスルーとしてカウント
    - 2連発生位置からの追跡を行う
    - カテゴリ別に集計（全体、2連目の条件別）
    """
    # カテゴリ定義
    categories = [
        '全体',
        '2連目1G_RB', '2連目1G_BB',
        '2連目[2-10,32]', '2連目[11-15]', '2連目[16-31]',
        '2連目BB_[2-10,32]', '2連目BB_[11-15]', '2連目BB_[16-31]',
        '2連目RB_[2-10,32]', '2連目RB_[11-15]', '2連目RB_[16-31]'
    ]
    
    # 3連以上天国の後からのみ追跡
    chains_df = chains_df.copy()
    chains_df['R3_Through'] = 0  # 3連基準スルー回数
    chains_df['R3_Heaven'] = False  # 3連基準天国判定
    chains_df['R3_2Chain_Loc'] = None  # 2連発生位置
    chains_df['R3_2nd_G'] = 0  # 2連目のG数
    chains_df['R3_2nd_Status'] = None  # 2連目のステータス
    
    prev_id = None
    r3_through = 0
    r3_active = False
    first_2chain_pos = None
    first_2chain_2nd_g = 0
    first_2chain_2nd_status = None
    
    for idx in chains_df.index:
        current_id = chains_df.at[idx, 'ID']
        chain_len = chains_df.at[idx, 'Chain_Length']
        is_3chain_heaven = chain_len >= 3
        is_2chain = chain_len == 2
        
        # 2連目のG数とステータスを取得
        hit_games = chains_df.at[idx, 'Hit_Games']
        hit_statuses = chains_df.at[idx, 'Hit_Statuses']
        second_g = hit_games[1] if len(hit_games) >= 2 else 0
        second_status = hit_statuses[1] if len(hit_statuses) >= 2 else None
        
        # 台が変わったらリセット
        if current_id != prev_id:
            r3_through = 0
            r3_active = is_3chain_heaven
            first_2chain_pos = None
            first_2chain_2nd_g = 0
            first_2chain_2nd_status = None
            prev_id = current_id
            
            if is_3chain_heaven:
                chains_df.at[idx, 'R3_Heaven'] = True
                chains_df.at[idx, 'R3_Through'] = 0
                chains_df.at[idx, 'R3_2Chain_Loc'] = '2連なし'
            continue
        
        if not r3_active:
            if is_3chain_heaven:
                r3_active = True
                r3_through = 0
                first_2chain_pos = None
                chains_df.at[idx, 'R3_Heaven'] = True
                chains_df.at[idx, 'R3_Through'] = 0
                chains_df.at[idx, 'R3_2Chain_Loc'] = '2連なし'
            continue
        
        # アクティブ状態での処理
        chains_df.at[idx, 'R3_Through'] = r3_through
        chains_df.at[idx, 'R3_Heaven'] = is_3chain_heaven
        
        # 2連発生位置を記録
        if first_2chain_pos is None:
            if is_2chain:
                first_2chain_pos = r3_through
                first_2chain_2nd_g = second_g
                first_2chain_2nd_status = second_status
                chains_df.at[idx, 'R3_2Chain_Loc'] = f'{first_2chain_pos}スルー目'
            else:
                chains_df.at[idx, 'R3_2Chain_Loc'] = '2連なし'
        else:
            chains_df.at[idx, 'R3_2Chain_Loc'] = f'{first_2chain_pos}スルー目'
        
        chains_df.at[idx, 'R3_2nd_G'] = first_2chain_2nd_g
        chains_df.at[idx, 'R3_2nd_Status'] = first_2chain_2nd_status
        
        # スルーカウント更新
        if is_3chain_heaven:
            r3_through = 0
            first_2chain_pos = None
            first_2chain_2nd_g = 0
            first_2chain_2nd_status = None
        else:
            r3_through += chain_len
    
    # 2連発生位置ごとの集計
    loc_labels = ['2連なし'] + [f'{i}スルー目' for i in range(11)]
    
    if loc_filter:
        loc_labels = [loc_filter]
    
    all_results = {}
    
    for loc_label in loc_labels:
        subset = chains_df[chains_df['R3_2Chain_Loc'] == loc_label]
        if len(subset) == 0:
            continue
        
        # 2連発生位置のオフセット計算
        if loc_label == '2連なし':
            base_offset = 0
        else:
            loc_num = int(loc_label.replace('スルー目', ''))
            base_offset = loc_num + 2
        
        loc_rate_results = []
        loc_sample_results = []
        
        for cat in categories:
            
            # カテゴリでフィルタ
            if cat == '全体':
                cat_subset = subset
            elif cat == '2連目1G_RB':
                cat_subset = subset[(subset['R3_2nd_G'] == 1) & (subset['R3_2nd_Status'] == 'RB')]
            elif cat == '2連目1G_BB':
                cat_subset = subset[(subset['R3_2nd_G'] == 1) & (subset['R3_2nd_Status'] == 'BB')]
            elif cat == '2連目[2-10,32]':
                cat_subset = subset[((subset['R3_2nd_G'] >= 2) & (subset['R3_2nd_G'] <= 10)) | (subset['R3_2nd_G'] == 32)]
            elif cat == '2連目[11-15]':
                cat_subset = subset[(subset['R3_2nd_G'] >= 11) & (subset['R3_2nd_G'] <= 15)]
            elif cat == '2連目[16-31]':
                cat_subset = subset[(subset['R3_2nd_G'] >= 16) & (subset['R3_2nd_G'] <= 31)]
            elif cat == '2連目BB_[2-10,32]':
                cat_subset = subset[(((subset['R3_2nd_G'] >= 2) & (subset['R3_2nd_G'] <= 10)) | (subset['R3_2nd_G'] == 32)) & (subset['R3_2nd_Status'] == 'BB')]
            elif cat == '2連目BB_[11-15]':
                cat_subset = subset[(subset['R3_2nd_G'] >= 11) & (subset['R3_2nd_G'] <= 15) & (subset['R3_2nd_Status'] == 'BB')]
            elif cat == '2連目BB_[16-31]':
                cat_subset = subset[(subset['R3_2nd_G'] >= 16) & (subset['R3_2nd_G'] <= 31) & (subset['R3_2nd_Status'] == 'BB')]
            elif cat == '2連目RB_[2-10,32]':
                cat_subset = subset[(((subset['R3_2nd_G'] >= 2) & (subset['R3_2nd_G'] <= 10)) | (subset['R3_2nd_G'] == 32)) & (subset['R3_2nd_Status'] == 'RB')]
            elif cat == '2連目RB_[11-15]':
                cat_subset = subset[(subset['R3_2nd_G'] >= 11) & (subset['R3_2nd_G'] <= 15) & (subset['R3_2nd_Status'] == 'RB')]
            elif cat == '2連目RB_[16-31]':
                cat_subset = subset[(subset['R3_2nd_G'] >= 16) & (subset['R3_2nd_G'] <= 31) & (subset['R3_2nd_Status'] == 'RB')]
            else:
                cat_subset = pd.DataFrame()
            
            # 各スルー回数ごとの天国移行率とサンプル数
            cat_rate = {'カテゴリ': cat}
            cat_sample = {'カテゴリ': cat}
            
            for rel_idx in range(relative_through_range):
                if loc_label == '2連なし':
                    thr_val = rel_idx
                else:
                    thr_val = base_offset + rel_idx
                
                at_through = cat_subset[cat_subset['R3_Through'] == thr_val]
                heaven_count = len(at_through[at_through['R3_Heaven'] == True])
                sample = len(at_through)
                
                rate = heaven_count / sample if sample > 0 else None
                cat_rate[f'{rel_idx}スルー'] = rate
                cat_sample[f'{rel_idx}スルー'] = sample if sample > 0 else None
            
            loc_rate_results.append(cat_rate)
            loc_sample_results.append(cat_sample)
        
        all_results[loc_label] = {
            'rate': pd.DataFrame(loc_rate_results),
            'sample': pd.DataFrame(loc_sample_results)
        }
    
    return all_results, chains_df


# ===== スタイリング関数 =====
def style_rate_table(df):
    """天国移行率のテーブルをスタイリング"""
    def highlight_rate(val):
        if pd.isna(val):
            return ''
        if isinstance(val, float):
            if val >= 0.5:
                return 'background-color: #92D050; color: black'
            elif val >= 0.35:
                return 'background-color: #FFEB9C; color: black'
            elif val >= 0.25:
                return 'background-color: #FFC7CE; color: black'
            else:
                return 'background-color: #FF6B6B; color: white'
        return ''
    
    return df.style.applymap(highlight_rate, subset=['天国移行率']).format({
        '天国移行率': '{:.1%}'
    })


def style_crosstab(df):
    """クロス集計テーブルをスタイリング"""
    def color_rate(val):
        if pd.isna(val):
            return 'background-color: #E0E0E0'
        if val >= 0.5:
            return 'background-color: #92D050'
        elif val >= 0.35:
            return 'background-color: #FFEB9C'
        elif val >= 0.25:
            return 'background-color: #FFC7CE'
        else:
            return 'background-color: #FF6B6B; color: white'
    
    return df.style.applymap(color_rate).format('{:.1%}', na_rep='[-]')


# ===== メイン =====
def main():
    st.title("🎰 デュオ天国移行率分析")
    st.markdown("---")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # ファイル選択
        st.subheader("📂 データソース")
        
        # キャッシュの存在確認
        cache_df, cache_chains, cache_time = load_cache()
        has_cache = cache_df is not None
        
        if has_cache:
            st.success(f"✅ キャッシュあり (作成: {cache_time})")
            use_cache = st.checkbox("キャッシュを使用（高速）", value=True)
        else:
            st.warning("⚠️ キャッシュなし（preprocess_data.pyを実行してください）")
            use_cache = False
        
        uploaded_file = st.file_uploader("または CSVをアップロード", type=['csv'])
        
        st.markdown("---")
        
        # パラメータ
        st.subheader("📊 分析パラメータ")
        heaven_threshold = st.slider("天国閾値 (G)", min_value=20, max_value=50, value=35)
        
        st.markdown("---")
        
        st.subheader("🔍 フィルター")
        min_chain = st.slider("最小連チャン数", min_value=2, max_value=10, value=2)
        max_through = st.slider("最大スルー表示", min_value=5, max_value=15, value=10)
    
    # データ読み込み
    if use_cache and has_cache:
        # キャッシュから高速読み込み
        df = cache_df
        chains_df = cache_chains
        st.sidebar.info(f"📊 キャッシュ読み込み完了\n{len(df):,}行, {len(chains_df):,}チェーン")
    elif uploaded_file is not None:
        df = load_data(uploaded_file)
        with st.spinner("解析中..."):
            df, chains_df = analyze_chains(df, heaven_threshold)
    else:
        try:
            df = load_data(DEFAULT_INPUT_FILE)
            with st.spinner("解析中（初回は時間がかかります）..."):
                df, chains_df = analyze_chains(df, heaven_threshold)
        except Exception as e:
            st.error(f"ファイル読み込みエラー: {e}")
            st.info("preprocess_data.py を実行してキャッシュを作成するか、CSVをアップロードしてください")
            return
    
    # 統計情報
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総ボーナス数", f"{len(df):,}")
    with col2:
        st.metric("チェーン数", f"{len(chains_df):,}")
    with col3:
        heaven_count = len(chains_df[chains_df['Is_Heaven'] == True])
        st.metric("天国回数", f"{heaven_count:,}")
    with col4:
        rate = heaven_count / len(chains_df) if len(chains_df) > 0 else 0
        st.metric("天国移行率", f"{rate:.1%}")
    
    st.markdown("---")
    
    # タブ
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 スルー回数別", "🔗 前回連チャン別", "📊 クロス集計", "🎯 3連基準2連スルー", "🔬 詳細データ"])
    
    with tab1:
        st.subheader("スルー回数別 天国移行率")
        rate_df = calculate_heaven_rate_by_through(chains_df, max_through)
        st.dataframe(style_rate_table(rate_df), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("前回連チャン別 天国移行率")
        chain_df = calculate_heaven_rate_by_chain(chains_df, min_chain, 20)
        st.dataframe(style_rate_table(chain_df), use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("クロス集計 (前回連チャン × スルー回数)")
        
        # フィルター
        col1, col2 = st.columns(2)
        with col1:
            chain_options = list(range(2, 21))
            selected_chains = st.multiselect("前回連チャン数", chain_options, default=list(range(2, 11)))
        with col2:
            through_options = list(range(0, max_through + 2))
            selected_through = st.multiselect("スルー回数", through_options, default=list(range(0, 6)))
        
        if selected_chains and selected_through:
            pivot_rate, pivot_count = calculate_crosstab(chains_df, selected_chains, selected_through)
            
            st.markdown("**天国移行率**")
            st.dataframe(style_crosstab(pivot_rate), use_container_width=True)
            
            st.markdown("**サンプル数**")
            st.dataframe(pivot_count.style.format('{:.0f}'), use_container_width=True)
        else:
            st.warning("フィルターを選択してください")
    
    with tab4:
        st.subheader("3連基準2連スルー分析")
        st.markdown("""
        **解説:** 3連以上を「真の天国」、2連を「隠れスルー（2回分）」として追跡
        """)
        
        # フィルター
        loc_options = ['2連なし'] + [f'{i}スルー目' for i in range(11)]
        selected_loc = st.selectbox("2連発生位置", loc_options, index=0)
        
        # 3連基準分析実行
        with st.spinner("3連基準分析中..."):
            r3_results, chains_with_r3 = calculate_3chain_rule_analysis(chains_df, selected_loc)
        
        if r3_results and selected_loc in r3_results:
            result_data = r3_results[selected_loc]
            df_rate = result_data['rate']
            df_sample = result_data['sample']
            
            if not df_rate.empty:
                # 2連発生位置のオフセット計算
                if selected_loc == '2連なし':
                    base_offset = 0
                else:
                    loc_num = int(selected_loc.replace('スルー目', ''))
                    base_offset = loc_num + 2
                
                col_names = [f'{i}スルー' for i in range(10)]
                
                # 3連後基準スルーのラベル行
                label_3chain = {'カテゴリ': '3連後基準スルー'}
                for i in range(10):
                    label_3chain[f'{i}スルー'] = f'{base_offset + i}スルー'
                
                # 2連発生基準スルーのラベル行
                label_2chain = {'カテゴリ': '2連発生基準スルー'}
                for i in range(10):
                    label_2chain[f'{i}スルー'] = f'{i}スルー'
                
                label_df = pd.DataFrame([label_3chain, label_2chain])
                
                # スタイル関数
                def color_cell(val):
                    if isinstance(val, str):
                        return 'background-color: #D9E2F3; font-weight: bold'
                    if pd.isna(val):
                        return 'background-color: #E0E0E0; color: #666'
                    if val >= 0.5:
                        return 'background-color: #92D050'
                    elif val >= 0.35:
                        return 'background-color: #FFEB9C'
                    elif val >= 0.25:
                        return 'background-color: #FFC7CE'
                    else:
                        return 'background-color: #FF6B6B; color: white'
                
                def format_rate(val):
                    if isinstance(val, str):
                        return val
                    if pd.isna(val):
                        return '[-]'
                    return f'{val:.1%}'
                
                def format_sample(val):
                    if isinstance(val, str):
                        return val
                    if pd.isna(val):
                        return '[-]'
                    return f'{int(val)}'
                
                # セクションヘッダー
                st.markdown(f"**【{selected_loc}】**")
                
                # ===== 天国移行率テーブル =====
                st.markdown("##### 天国移行率")
                rate_data = df_rate.copy()
                rate_data.columns = ['カテゴリ'] + col_names
                combined_rate = pd.concat([label_df, rate_data], ignore_index=True)
                combined_rate = combined_rate.set_index('カテゴリ')
                
                styled_rate = combined_rate.style.applymap(color_cell).format(format_rate)
                st.dataframe(styled_rate, use_container_width=True, height=450)
                
                # ===== サンプル数テーブル =====
                st.markdown("##### サンプル数")
                sample_data = df_sample.copy()
                sample_data.columns = ['カテゴリ'] + col_names
                combined_sample = pd.concat([label_df, sample_data], ignore_index=True)
                combined_sample = combined_sample.set_index('カテゴリ')
                
                def color_sample(val):
                    if isinstance(val, str):
                        return 'background-color: #D9E2F3; font-weight: bold'
                    if pd.isna(val):
                        return 'background-color: #E0E0E0; color: #666'
                    return ''
                
                styled_sample = combined_sample.style.applymap(color_sample).format(format_sample)
                st.dataframe(styled_sample, use_container_width=True, height=450)
            else:
                st.warning("データがありません")
        else:
            st.warning("データがありません")
    
    with tab5:
        st.subheader("チェーンデータ")
        
        # フィルター
        col1, col2, col3 = st.columns(3)
        with col1:
            show_heaven_only = st.checkbox("天国のみ表示")
        with col2:
            min_chain_filter = st.number_input("最小連チャン長", min_value=1, value=1)
        with col3:
            max_rows = st.number_input("最大表示行数", min_value=100, max_value=10000, value=1000)
        
        display_df = chains_df.copy()
        if show_heaven_only:
            display_df = display_df[display_df['Is_Heaven'] == True]
        if min_chain_filter > 1:
            display_df = display_df[display_df['Chain_Length'] >= min_chain_filter]
        
        display_cols = ['ID', 'Chain_Number', 'Chain_Length', 'Is_Heaven', 'First_G', 
                       'Through_Before', 'Prev_Chain_Length', 'Total_Dedama', 'Net_Diff']
        st.dataframe(display_df[display_cols].head(max_rows), use_container_width=True, hide_index=True)
        
        # ダウンロード
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 CSVダウンロード",
            data=csv,
            file_name=f"chain_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )


if __name__ == "__main__":
    main()
