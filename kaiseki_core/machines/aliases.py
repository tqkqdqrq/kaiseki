"""表示名 → ASCII モジュール名の解決表。"""

ALIASES: dict[str, str] = {
    "デュオ": "_template_a_art",
    "duo": "_template_a_art",
    "AT機": "_template_at",
    "at": "_template_at",
    "ビックドリーム": "bigdream",
    "ビッグドリーム": "bigdream",
}


def resolve(name: str) -> str:
    return ALIASES.get(name, name)
