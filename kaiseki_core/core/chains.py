"""チェーン (連荘) 検出。終端判定は adapter.is_streak_break に委譲。

入力: normalize 済みの DataFrame
出力: (data_df, chains_df)
- data_df: 元の DataFrame に Chain_ID / Chain_Position / Chain_Length 列を追加
- chains_df: チェーン単位の集計行 (ChainRow ベース)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd

from .payout import compute_chain_financials

if TYPE_CHECKING:
    from ..machines.base import MachineAdapter


_GROUP_KEYS = ("Hall", "Date", "Machine_No")


def _iter_chain_indices(group: pd.DataFrame, adapter: "MachineAdapter") -> list[list[int]]:
    """1台1日分の DataFrame からチェーンごとのインデックスリストを返す。"""
    chains: list[list[int]] = []
    current: list[int] = []
    prev_row: pd.Series | None = None

    for idx, row in group.iterrows():
        if prev_row is None:
            current = [idx]
        else:
            if adapter.is_streak_break(prev_row, row):
                chains.append(current)
                current = [idx]
            else:
                current.append(idx)
        prev_row = row

    if current:
        chains.append(current)
    return chains


def detect_chains(
    df: pd.DataFrame, adapter: "MachineAdapter"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """normalize 済み DataFrame からチェーンを検出。"""
    work = df.copy()
    work["Chain_ID"] = 0
    work["Chain_Position"] = 0
    work["Chain_Length"] = 0

    chain_rows: list[dict] = []
    fin = adapter.financials
    streak_min = fin.streak_min_length
    next_chain_id = 1

    sorted_df = work.sort_values(list(_GROUP_KEYS) + ["Count"])

    for keys, group in sorted_df.groupby(list(_GROUP_KEYS), sort=False):
        chain_idx_lists = _iter_chain_indices(group, adapter)

        prev_chain_length = 0
        prev_special_judge = 0.0
        through_count = 0
        daily_balance = 0.0
        max_daily_balance = 0.0
        chain_number_in_day = 0

        for chain_idx_list in chain_idx_lists:
            chain_number_in_day += 1
            chain_id = next_chain_id
            next_chain_id += 1
            chain_len = len(chain_idx_list)

            for pos, idx in enumerate(chain_idx_list, start=1):
                work.at[idx, "Chain_ID"] = chain_id
                work.at[idx, "Chain_Position"] = pos
                work.at[idx, "Chain_Length"] = chain_len

            hits = [
                {
                    "Start": int(work.at[i, "Start"]),
                    "Dedama": int(work.at[i, "Dedama"]),
                    "Status": str(work.at[i, "Status"]),
                }
                for i in chain_idx_list
            ]
            fin_calc = compute_chain_financials(hits, fin)
            is_streak = chain_len >= streak_min

            id_value = str(work.at[chain_idx_list[0], "ID"])
            first_g = hits[0]["Start"]

            extras = adapter.chain_extras([work.loc[i] for i in chain_idx_list])

            balance_before = daily_balance
            max_balance_before = max_daily_balance
            daily_balance += fin_calc["net_diff"]
            if daily_balance > max_daily_balance:
                max_daily_balance = daily_balance
            is_reset = False

            chain_rows.append(
                {
                    "id": id_value,
                    "chain_id": chain_id,
                    "chain_number": chain_number_in_day,
                    "chain_length": chain_len,
                    "hit_games": [h["Start"] for h in hits],
                    "hit_payouts": [h["Dedama"] for h in hits],
                    "hit_kinds": [h["Status"] for h in hits],
                    "is_streak": is_streak,
                    "first_g": first_g,
                    "through_before": through_count,
                    "prev_chain_length": prev_chain_length,
                    "raw_payout": fin_calc["raw_payout"],
                    "net_payout": fin_calc["net_payout"],
                    "total_invest": fin_calc["total_invest"],
                    "streak_invest": fin_calc["streak_invest"],
                    "net_diff": fin_calc["net_diff"],
                    "special_judge": fin_calc["special_judge"],
                    "prev_special_judge": prev_special_judge,
                    "daily_balance_before": balance_before,
                    "max_daily_balance_before": max_balance_before,
                    "is_reset": is_reset,
                    "extra": extras,
                }
            )

            prev_chain_length = chain_len
            prev_special_judge = fin_calc["special_judge"]
            if is_streak:
                through_count = 0
            else:
                through_count += 1

    chains_df = pd.DataFrame(chain_rows)
    return work, chains_df
