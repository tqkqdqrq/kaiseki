"""データクラス定義: 機種非依存の正規化済みヒット行とチェーン行。"""

from dataclasses import dataclass, field
from typing import Literal, Any

HitKind = Literal["BB", "RB", "AT", "ART", "BONUS", "OTHER"]
MachineType = Literal["A_ART", "AT"]


@dataclass(frozen=True)
class NormalizedHit:
    hall: str
    machine_no: int
    date: str
    time: str
    order: int
    count: int
    start_g: int
    payout: int
    kind: HitKind
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChainRow:
    id: str
    chain_id: int
    chain_number: int
    chain_length: int
    hit_games: list[int]
    hit_payouts: list[int]
    hit_kinds: list[str]
    is_streak: bool
    first_g: int
    through_before: int
    prev_chain_length: int
    raw_payout: int
    net_payout: float
    total_invest: float
    streak_invest: float
    net_diff: float
    special_judge: float
    prev_special_judge: float
    daily_balance_before: float
    max_daily_balance_before: float
    is_reset: bool
    extra: dict[str, Any] = field(default_factory=dict)
