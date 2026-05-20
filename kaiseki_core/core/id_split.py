"""ID列 ("Hall_台番_YYYYMMDD") を Hall / Machine_No / Date に分解。"""

import pandas as pd


def split_id(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """ID列の最後の "_" 区切りから日付を、その手前を台番号として切り出す。
    店舗名にアンダースコアが含まれていても末尾2要素で分解できるよう rsplit を使う。"""
    if id_col not in df.columns:
        raise KeyError(f"{id_col} column not found")

    parts = df[id_col].astype(str).str.rsplit("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise ValueError(
            f"{id_col} does not match 'Hall_MachineNo_YYYYMMDD' pattern. "
            f"sample={df[id_col].iloc[0] if len(df) else None}"
        )

    out = df.copy()
    out["Hall"] = parts[0]
    out["Machine_No"] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    out["Date"] = parts[2]
    return out
