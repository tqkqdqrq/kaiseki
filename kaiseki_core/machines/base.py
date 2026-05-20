"""機種アダプタの Protocol と最低限のデフォルト実装。

新機種を追加する手順:
1. machines/<name>/ ディレクトリを作成
2. adapter.py に MachineAdapter 実装 (最低 name/machine_type/financials + is_streak_break)
3. 必要なら column_map で CSV列を正規化
4. 必要なら sheets() で機種固有Excelシートを定義
5. build() トップレベル関数からインスタンスを返す
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, TYPE_CHECKING
import pandas as pd

from ..core.payout import Financials
from ..core.schema import MachineType

if TYPE_CHECKING:
    from ..core.excel_base import SheetSpec


@runtime_checkable
class MachineAdapter(Protocol):
    name: str
    machine_type: MachineType
    financials: Financials

    def column_map(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def is_streak_break(self, prev: pd.Series, curr: pd.Series) -> bool: ...
    def chain_extras(self, chain_rows: list[pd.Series]) -> dict[str, Any]: ...
    def sheets(self) -> list["SheetSpec"]: ...


class BaseAdapter:
    """デフォルト実装。新機種はこれを継承して必要分だけ override。"""

    name: str = "base"
    machine_type: MachineType = "AT"
    financials: Financials

    def column_map(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def is_streak_break(self, prev: pd.Series, curr: pd.Series) -> bool:
        raise NotImplementedError("subclass must implement is_streak_break")

    def chain_extras(self, chain_rows: list[pd.Series]) -> dict[str, Any]:
        return {}

    def sheets(self) -> list["SheetSpec"]:
        return []

    def detect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """チェーン検出。デフォルトは core.detect_chains を is_streak_break で駆動。
        機種固有の連荘ロジック (例: AT単位集計, CZ成功判定) が必要なら override。"""
        from ..core.chains import detect_chains
        return detect_chains(df, self)
