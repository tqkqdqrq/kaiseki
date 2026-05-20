"""列正規化と Count / Original_Order の付与。"""

from typing import TYPE_CHECKING
import pandas as pd

from .id_split import split_id

if TYPE_CHECKING:
    from ..machines.base import MachineAdapter

REQUIRED_COLS = ("ID", "Status", "Start", "Dedama", "Time")


def normalize(df: pd.DataFrame, adapter: "MachineAdapter") -> pd.DataFrame:
    """1) Original_Order 付与
    2) adapter.column_map で機種固有列を正規列名に
    3) ID 分解で Hall/Machine_No/Date
    4) (Hall, Date, Machine_No, Time) ソート + cumcount で Count 付与
    5) Original_Order で元順に復元
    """
    work = df.copy()
    work["Original_Order"] = range(len(work))
    work = adapter.column_map(work)

    missing = [c for c in REQUIRED_COLS if c not in work.columns]
    if missing:
        raise KeyError(
            f"adapter.column_map did not produce required columns: {missing}. "
            f"got={list(work.columns)}"
        )

    work["Start"] = pd.to_numeric(work["Start"], errors="coerce").fillna(0).astype(int)
    work["Dedama"] = pd.to_numeric(work["Dedama"], errors="coerce").fillna(0).astype(int)

    work = split_id(work, "ID")

    work = work.sort_values(["Hall", "Date", "Machine_No", "Time", "Original_Order"]).reset_index(drop=True)
    work["Count"] = work.groupby(["Hall", "Date", "Machine_No"]).cumcount() + 1

    work = work.sort_values("Original_Order").reset_index(drop=True)
    return work
