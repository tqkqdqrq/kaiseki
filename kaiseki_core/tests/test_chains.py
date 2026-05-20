"""detect_chains: モック adapter での分割確認。"""

import pandas as pd
from kaiseki_core.core.chains import detect_chains
from kaiseki_core.core.normalize import normalize
from kaiseki_core.core.payout import Financials
from kaiseki_core.machines.base import BaseAdapter


class _MockAT(BaseAdapter):
    name = "_mock_at"
    machine_type = "AT"
    financials = Financials(invest_per_g=2.0, payout_per_g=2.8, streak_min_length=2)

    def is_streak_break(self, prev, curr) -> bool:
        return int(curr["Start"]) > 150


def _df():
    return pd.DataFrame(
        [
            {"ID": "TestHall_1_20260101", "Status": "AT", "Start": 250, "Dedama": 1500, "Time": "10:00"},
            {"ID": "TestHall_1_20260101", "Status": "AT", "Start": 80, "Dedama": 1200, "Time": "10:30"},
            {"ID": "TestHall_1_20260101", "Status": "AT", "Start": 400, "Dedama": 1800, "Time": "11:30"},
            {"ID": "TestHall_2_20260101", "Status": "AT", "Start": 180, "Dedama": 2000, "Time": "09:30"},
            {"ID": "TestHall_2_20260101", "Status": "AT", "Start": 30, "Dedama": 1600, "Time": "10:15"},
        ]
    )


def test_detect_chains_splits():
    adapter = _MockAT()
    norm = normalize(_df(), adapter)
    data_df, chains_df = detect_chains(norm, adapter)

    # 期待: Hall1 = [chain1: 250+80, chain2: 400], Hall2 = [chain1: 180+30] → 計3チェーン
    assert len(chains_df) == 3
    lengths = sorted(chains_df["chain_length"].tolist())
    assert lengths == [1, 2, 2]
    streak_counts = chains_df["is_streak"].sum()
    assert streak_counts == 2

    # through_before の単調性 (Hall1: 0, 0, Hall2: 0)
    hall1 = chains_df[chains_df["id"] == "TestHall_1_20260101"].sort_values("chain_number")
    assert list(hall1["through_before"]) == [0, 0]


def test_data_df_has_chain_columns():
    adapter = _MockAT()
    norm = normalize(_df(), adapter)
    data_df, _ = detect_chains(norm, adapter)
    for col in ("Chain_ID", "Chain_Position", "Chain_Length"):
        assert col in data_df.columns
    assert (data_df["Chain_ID"] > 0).all()
