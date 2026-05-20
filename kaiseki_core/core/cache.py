"""pickle ベースのチェーン検出キャッシュ。"""

from __future__ import annotations
import os
import pickle
from typing import Callable, Tuple
import pandas as pd

CacheBuilder = Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]


def load_or_build(cache_path: str, builder: CacheBuilder) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, tuple) and len(payload) == 2:
                return payload
        except Exception:
            pass

    data_df, chains_df = builder()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((data_df, chains_df), f)
    return data_df, chains_df
