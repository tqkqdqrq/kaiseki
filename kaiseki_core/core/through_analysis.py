"""スルー集計ユーティリティ。

「信号(例:CZ)が N回目で当選 (Nスルー目当選)」型の集計を機種非依存で扱う。

主な機能:
- 各信号行への Through_Index 付与 (1始まり、成功でリセット)
- 各信号行への Cum_G_At_Hit 付与 (前回成功からのStart累積、自身含む)
- 連荘起点が経由した信号の Through_Index (Chain_Start_Through_Index) リンク

使用シーン:
- CZ→AT機 (bigdream): RB を信号、CZ成功で AT到達 = リセット
- 同コンセプトで他機種にも適用可能 (例: ボーナス連 = 成功、単発 = 失敗)
"""

from __future__ import annotations
from typing import Iterable
import pandas as pd


def annotate_through_index(
    df: pd.DataFrame,
    *,
    signal_mask: pd.Series,
    success_mask: pd.Series,
    reset_mask: pd.Series,
    group_keys: list[str],
    sort_keys: list[str],
    through_col: str = "Through_Index",
    cum_g_col: str = "Cum_G_At_Hit",
    start_col: str = "Start",
) -> pd.DataFrame:
    """各信号行に Through_Index と Cum_G_At_Hit を付与。

    Args:
        df: 入力 DataFrame
        signal_mask: 集計対象信号行を示す bool Series (例: Is_CZ)
        success_mask: 信号が成功した行を示す bool Series (例: Is_CZ_Success)
        reset_mask: Through カウントをリセットする行を示す bool Series
                    (success_mask と同じか、AT到達行を含むなど)
        group_keys: 台日グルーピング ['Hall', 'Date', 'Machine_No']
        sort_keys: グループ内時系列ソートキー ['Time', 'Original_Order']
        through_col, cum_g_col: 新規列名
        start_col: 通常G消化を表す列名

    Returns:
        Through_Index と Cum_G_At_Hit を持つ DataFrame
    """
    out = df.copy()
    out[through_col] = 0
    out[cum_g_col] = 0

    sorted_idx = out.sort_values(group_keys + sort_keys).index.tolist()
    groups: dict[tuple, list[int]] = {}
    for i in sorted_idx:
        key = tuple(out.at[i, k] for k in group_keys)
        groups.setdefault(key, []).append(i)

    for _, indices in groups.items():
        count_since_reset = 0
        normal_g_cum = 0
        for i in indices:
            start_g = int(out.at[i, start_col])
            if signal_mask.at[i]:
                count_since_reset += 1
                cum_g = normal_g_cum + start_g
                out.at[i, through_col] = count_since_reset
                out.at[i, cum_g_col] = cum_g
                normal_g_cum = cum_g
                if reset_mask.at[i]:
                    count_since_reset = 0
                    normal_g_cum = 0
            else:
                # 信号以外の行 (例: AT行) は通常Gに加算しないか加算するかは機種依存
                # デフォルト: 加算しない (信号間隔をそのまま見る)
                if reset_mask.at[i]:
                    count_since_reset = 0
                    normal_g_cum = 0
    return out


def link_chain_to_signal(
    data_df: pd.DataFrame,
    chains_df: pd.DataFrame,
    *,
    chain_id_col: str = "Chain_ID",
    chain_start_through_col: str = "Chain_Start_Through_Index",
    chain_pre_g_col: str = "Chain_Pre_Normal_G",
    through_col: str = "Through_Index",
    cum_g_col: str = "Cum_G_At_Hit",
) -> pd.DataFrame:
    """連荘起点に「経由した信号の Through_Index と Cum_G」を ChainData に追記。

    各 chain について、データ側で Chain_Start_Through_Index 列が既に書かれている前提。
    chains_df に starting_through, pre_normal_g 列を追加して返す。
    """
    out = chains_df.copy()
    starts: list[int] = []
    pre_gs: list[int] = []
    for _, row in out.iterrows():
        cid = row[chain_id_col] if chain_id_col in row else row.get("chain_id")
        sub = data_df[data_df["Chain_ID"] == cid].sort_values("Original_Order")
        if sub.empty:
            starts.append(0)
            pre_gs.append(0)
        else:
            first = sub.iloc[0]
            starts.append(int(first.get(chain_start_through_col, 0)))
            pre_gs.append(int(first.get(chain_pre_g_col, 0)))
    out["starting_through"] = starts
    out["pre_normal_g"] = pre_gs
    return out
