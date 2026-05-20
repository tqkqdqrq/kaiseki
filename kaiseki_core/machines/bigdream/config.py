"""L ビックドリーム THE GOLDEN PUSHER (スマスロ, CZ→AT 型) パラメータ。

スペック出典:
- ちょんぼりすた https://chonborista.com/slot/sammy-slot/254356/ (2026-05-17 取得)
- DMMぱちタウン https://p-town.dmm.com/machines/4981

機種仕様要点 (詳細は Desktop\台データオンライン\ビッグドリーム\ビッグドリーム_読み方.md):
- 種別: RB=CZ信号 / BB,ART=AT信号
- 状態列(UI判定済): CZ成功/CZ失敗/朝一直撃/直撃/上位AT/下位AT/引き戻し
- 連荘判定: 直前ATなら継続、CZ挟んだら新連荘
"""

# === 期待値・機械割計算に使用する確定値 ===

# 通常時コインもち 30.7G/50枚 → 純減 50/30.7 = 1.629枚/G
COIN_HOLD_G_PER_50 = 30.7
INVEST_PER_G = 50.0 / COIN_HOLD_G_PER_50  # ≒ 1.629

# AT中純増 (参考値、機械割計算には不要)
AT_PAYOUT_PER_G = 8.2

# 着席ワームアップG: 着席から WARMUP_G 未満では当選しないものとする
# = 期待値計算で「打ち始めから WARMUP_G G消化するまで当選サンプル除外」
# bigdream は実機仕様 + 体感で 33G (0-32G は当選なし、33G以降から集計)
WARMUP_G = 33

# === 解析ロジック用パラメータ ===

STREAK_MIN_LENGTH = 2

VALID_STATUS = ("BB", "RB", "ART")
AT_STATUS = ("BB", "ART")
CZ_STATUS = ("RB",)

NEW_CHAIN_STATES = ("直撃", "朝一直撃")
CONTINUE_CHAIN_STATES = ("下位AT", "上位AT", "引き戻し")

CZ_SUCCESS_LABEL = "CZ成功"
CZ_FAIL_LABEL = "CZ失敗"

DEFAULT_YEAR = 2026

# === 参考: 機種スペック (計算には現状未使用、表示用に残す) ===
# 機械割: 設定1 97.4% / 設定6 111.3%
# CZ確率: 設定1 1/337.7 / 設定6 1/303.4
# AT初当り: 設定1 1/629.3 / 設定6 1/536.2
# 天井: 通常 1499G (999/555/333G選択あり), リセ後 999G
