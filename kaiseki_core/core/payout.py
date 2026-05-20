"""機械割・投資・純増の計算。A+ART式とAT式を Financials の値有無で分岐。"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Financials:
    coin_hold_g_per_50: Optional[float] = None
    invest_per_g: Optional[float] = None
    payout_per_g: Optional[float] = None
    streak_min_length: int = 2

    def __post_init__(self) -> None:
        if self.coin_hold_g_per_50 is None and self.invest_per_g is None:
            raise ValueError(
                "Financials requires either coin_hold_g_per_50 (A+ART) "
                "or invest_per_g (AT)."
            )

    @property
    def is_a_art(self) -> bool:
        return self.coin_hold_g_per_50 is not None

    @property
    def is_at(self) -> bool:
        return self.coin_hold_g_per_50 is None and self.invest_per_g is not None


def _invest_from_games(games: int, fin: Financials) -> float:
    if fin.is_a_art:
        return games / fin.coin_hold_g_per_50 * 50.0
    return float(games) * fin.invest_per_g


def compute_chain_financials(hits: list[dict], fin: Financials) -> dict:
    """1チェーン分のヒット (Start, Dedama を持つ dict のリスト) から
    投資・純増・差枚・special_judge を算出して返す。"""
    if not hits:
        raise ValueError("hits must not be empty")

    starts = [int(h["Start"]) for h in hits]
    payouts = [int(h["Dedama"]) for h in hits]

    total_g = sum(starts)
    total_invest = _invest_from_games(total_g, fin)
    first_invest = _invest_from_games(starts[0], fin)
    streak_invest = total_invest - first_invest
    raw_payout = sum(payouts)
    net_payout = raw_payout - streak_invest
    net_diff = raw_payout - total_invest
    last_payout = payouts[-1]
    special_judge = net_payout - last_payout

    return {
        "raw_payout": raw_payout,
        "net_payout": net_payout,
        "total_invest": total_invest,
        "streak_invest": streak_invest,
        "net_diff": net_diff,
        "special_judge": special_judge,
        "total_g": total_g,
    }


def machine_rate(total_payout: float, total_invest: float) -> float:
    """機械割 (%) = 払出 / 投入 * 100。投入0なら 0.0 を返す。"""
    if total_invest <= 0:
        return 0.0
    return total_payout / total_invest * 100.0
