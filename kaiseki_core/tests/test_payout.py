"""payout.compute_chain_financials の手計算一致テスト。"""

from kaiseki_core.core.payout import Financials, compute_chain_financials, machine_rate


def test_a_art_demo():
    fin = Financials(coin_hold_g_per_50=25.3, streak_min_length=2)
    hits = [
        {"Start": 300, "Dedama": 400, "Status": "BB"},
        {"Start": 10, "Dedama": 400, "Status": "BB"},
    ]
    r = compute_chain_financials(hits, fin)
    total_invest_expected = 310 / 25.3 * 50.0
    first_invest_expected = 300 / 25.3 * 50.0
    assert abs(r["total_invest"] - total_invest_expected) < 1e-6
    assert abs(r["streak_invest"] - (total_invest_expected - first_invest_expected)) < 1e-6
    assert r["raw_payout"] == 800
    assert abs(r["net_diff"] - (800 - total_invest_expected)) < 1e-6


def test_at_demo():
    fin = Financials(invest_per_g=2.0, payout_per_g=2.8, streak_min_length=2)
    hits = [
        {"Start": 250, "Dedama": 1500, "Status": "AT"},
        {"Start": 80, "Dedama": 1200, "Status": "AT"},
    ]
    r = compute_chain_financials(hits, fin)
    assert r["total_invest"] == 330 * 2.0
    assert r["streak_invest"] == 80 * 2.0
    assert r["raw_payout"] == 2700
    assert r["net_diff"] == 2700 - 660
    assert r["special_judge"] == r["net_payout"] - 1200


def test_machine_rate():
    assert machine_rate(0, 0) == 0.0
    assert abs(machine_rate(1100, 1000) - 110.0) < 1e-9


def test_financials_validation():
    try:
        Financials()
    except ValueError:
        pass
    else:
        raise AssertionError("Financials() should raise ValueError")
