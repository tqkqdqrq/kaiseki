"""CSV読込: utf-8-sig → cp932 の順でフォールバック。"""

import pandas as pd

_ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis")


def read_csv_auto(path: str, **kwargs) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
    raise UnicodeDecodeError(
        "kaiseki_core",
        b"",
        0,
        1,
        f"Failed to decode {path} with any of {_ENCODINGS}: {last_err}",
    )
