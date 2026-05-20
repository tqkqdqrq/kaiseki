"""AT機リファレンス実装。新規AT機はこれをコピペして定数とシートを差し替える。"""

from __future__ import annotations
import pandas as pd

from ..base import BaseAdapter
from ...core.payout import Financials
from ...core.excel_base import SheetSpec

from . import config


class TemplateATAdapter(BaseAdapter):
    name = "_template_at"
    machine_type = "AT"
    financials = Financials(
        invest_per_g=config.INVEST_PER_G,
        payout_per_g=config.PAYOUT_PER_G,
        streak_min_length=config.STREAK_MIN_LENGTH,
    )

    def column_map(self, df: pd.DataFrame) -> pd.DataFrame:
        # azukun準拠の入力ならno-op。実機CSVが違う列名の場合は rename を追加:
        # return df.rename(columns={"通常G": "Start", "AT出玉": "Dedama", "種別": "Status"})
        return df

    def is_streak_break(self, prev: pd.Series, curr: pd.Series) -> bool:
        return int(curr["Start"]) > config.NORMAL_G_BREAK

    def sheets(self) -> list[SheetSpec]:
        from .sheets import build_zone_expectation_sheet, build_at_chain_summary_sheet
        return [
            SheetSpec("ゾーン期待値", build_zone_expectation_sheet),
            SheetSpec("AT連荘サマリ", build_at_chain_summary_sheet),
        ]
        # CZ→AT 型ならここに以下を追加 (bigdream参照):
        # from ...core.excel_through import build_through_summary_formula_sheet
        # def my_through(d, c, wb):
        #     build_through_summary_formula_sheet(
        #         wb, "CZスルー別", invest_per_g=config.INVEST_PER_G,
        #         signal_col="Is_CZ", success_col="Is_CZ_Success",
        #         through_col="CZ_Through_Index", cum_g_col="CZ_Cum_G_At_Hit",
        #         chain_through_col="starting_cz_through", chain_payout_col="net_payout",
        #     )
        # SheetSpec("CZスルー別", my_through)


def build() -> TemplateATAdapter:
    return TemplateATAdapter()
