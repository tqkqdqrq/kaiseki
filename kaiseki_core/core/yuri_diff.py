"""有利区間差枚 (advantage zone diff) 汎用計算。

スマスロ機種で「上位AT (またはそれに相当する有利区間リセット契機) を境に
有利区間内の累積差枚を行ごとに計算する」共通ロジック。

- 行ごとの delta = Dedama - Start * invest_per_g
- 累積 yuri_diff を時系列で計算 (台日単位)
- リセット契機 (state == reset_state) 行で yuri_diff = 0、その行から累積再開
- 各行に yuri_diff_pre (信号上がる前 = 前行終了時点) を書き込む
- 連荘起点行 (Chain_Position == 1) の yuri_diff_pre を chains_df に転記する
"""
from __future__ import annotations
import pandas as pd


def compute_yuri_diff_pre(
    df: pd.DataFrame,
    invest_per_g: float,
    reset_state: str = "上位AT",
    state_col: str = "State",
    start_col: str = "Start",
    dedama_col: str = "Dedama",
    sort_cols: tuple = ("Hall", "Date", "Machine_No", "Time", "Original_Order"),
    group_cols: tuple = ("Hall", "Date", "Machine_No"),
) -> pd.DataFrame:
    """全行に Yuri_Diff_Pre 列 (信号上がる前の有利区間差枚) を付与した df を返す。

    精査UI app.py の第3パスと完全に一致するロジック:
    - reset_state 行: 表示=0、内部 yuri_diff は新区間で累積継続
    - reset_state 連続行: 表示=0、累積継続
    - reset_state 終了後の最初の通常行: 表示=上位中累積、その後通常累積
    """
    out = df.copy()
    out["Yuri_Diff_Pre"] = 0
    sorted_idx = out.sort_values(list(sort_cols)).index.tolist()
    groups: dict[tuple, list[int]] = {}
    for i in sorted_idx:
        key = tuple(out.at[i, c] for c in group_cols)
        groups.setdefault(key, []).append(i)

    for indices in groups.values():
        yuri_diff = 0.0
        in_reset = False
        for i in indices:
            try:
                s = int(out.at[i, start_col])
            except Exception:
                s = 0
            try:
                d = int(out.at[i, dedama_col])
            except Exception:
                d = 0
            state = str(out.at[i, state_col])
            delta = d - s * invest_per_g
            is_reset_now = (state == reset_state)
            if is_reset_now:
                if not in_reset:
                    # 上位AT開始 = リセット
                    yuri_diff = 0.0
                out.at[i, "Yuri_Diff_Pre"] = 0
                yuri_diff += delta
            else:
                out.at[i, "Yuri_Diff_Pre"] = int(round(yuri_diff))
                yuri_diff += delta
            in_reset = is_reset_now
    return out


def attach_chain_yuri_diff_pre(
    chains_df: pd.DataFrame,
    data_df_with_yuri: pd.DataFrame,
    chain_id_col: str = "Chain_ID",
    chain_position_col: str = "Chain_Position",
) -> pd.DataFrame:
    """ChainData の各連荘に yuri_diff_pre (起点直前の累積差枚) を付与。

    data_df_with_yuri: compute_yuri_diff_pre 済みの data_df。
    chain_id ごとに Chain_Position==1 の Yuri_Diff_Pre を拾う。
    """
    if chains_df.empty:
        chains_df = chains_df.copy()
        chains_df["yuri_diff_pre"] = []
        return chains_df
    starts = data_df_with_yuri[data_df_with_yuri[chain_position_col] == 1][
        [chain_id_col, "Yuri_Diff_Pre"]
    ].rename(columns={chain_id_col: "chain_id", "Yuri_Diff_Pre": "yuri_diff_pre"})
    out = chains_df.merge(starts, on="chain_id", how="left")
    out["yuri_diff_pre"] = out["yuri_diff_pre"].fillna(0).astype(int)
    return out
