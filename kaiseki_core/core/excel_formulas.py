"""Excel数式テンプレ。文字列を返すだけのヘルパ。"""


def countifs(range_pairs: list[tuple[str, str]]) -> str:
    parts = ",".join(f"{r},{c}" for r, c in range_pairs)
    return f"=COUNTIFS({parts})"


def averageifs(target_range: str, range_pairs: list[tuple[str, str]]) -> str:
    parts = ",".join(f"{r},{c}" for r, c in range_pairs)
    return f"=AVERAGEIFS({target_range},{parts})"


def safe_divide(numer: str, denom: str) -> str:
    return f"=IFERROR({numer}/{denom},0)"
