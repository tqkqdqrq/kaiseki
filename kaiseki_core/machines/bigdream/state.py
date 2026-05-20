"""bigdream state判定 + 有利区間差枚 統合1パス。

ユーザー仕様 (2026-05-19 改訂):
- 上位AT は下位AT連荘中に「一撃1000枚到達 OR 有利区間差枚2300枚到達」してから
  5-25G で発動する。それ未満なら 5-25G AT が来ても引き戻し扱い。
- 上位AT 発動行で有利区間リセット (= yuri_diff = 0 リスタート)。
- 直撃 (Start>80) では有利区間リセットしない (実機との差異だが、ユーザー指定通り)。

1パス時系列で State と Yuri_Diff_Pre を同時に決定する。
state と yuri_diff は相互依存 (上位AT判定に yuri_diff、yuri_diffリセット点に上位AT判定が必要) のため。
"""
from __future__ import annotations
import pandas as pd

from . import config


JYOI_AT_LABEL = "上位AT"
HIKIMODOSHI_LABEL = "引き戻し"
KAGEI_AT_LABEL = "下位AT"
CHOKUGEKI_LABEL = "直撃"
MORNING_CHOKUGEKI_LABEL = "朝一直撃"
CZ_SUCCESS_LABEL = config.CZ_SUCCESS_LABEL
CZ_FAIL_LABEL = config.CZ_FAIL_LABEL

# 上位AT 連続行の合計純増がこの値未満なら引き戻し格下げ (2パス目)
JYOI_THRESHOLD_PAYOUT = 500
# CZ成功 = RB 直後の AT で Start≤この値
CZ_SUCCESS_NEXT_START_MAX = 3
# 上位AT 発動条件
JYOI_UNLOCK_RENCHAN = 1000   # 下位AT連荘中の純増累計しきい値
JYOI_UNLOCK_YURI_DIFF = 2300  # 有利区間差枚しきい値
JYOI_CONFIRM_THRESHOLD = 3000  # 下位純増がこの値以上で発動した上位は確定 (500枚格下げ免除)


def assign_states(df: pd.DataFrame) -> pd.DataFrame:
    """全行に State と Yuri_Diff_Pre を判定して書き込む。

    入力 df 必須列: Hall, Date, Machine_No, Time, Original_Order, Status, Start, Dedama
    出力: State 列, Yuri_Diff_Pre 列を上書き/追加した同形 df。
    """
    out = df.copy()
    if "State" not in out.columns:
        out["State"] = ""
    out["Yuri_Diff_Pre"] = 0

    invest_per_g = config.INVEST_PER_G

    sorted_idx = out.sort_values(
        ["Hall", "Date", "Machine_No", "Time", "Original_Order"]
    ).index.tolist()
    groups: dict[tuple, list[int]] = {}
    for i in sorted_idx:
        key = (out.at[i, "Hall"], out.at[i, "Date"], out.at[i, "Machine_No"])
        groups.setdefault(key, []).append(i)

    for indices in groups.values():
        # === 第1パス: 時系列1ループで State + Yuri_Diff_Pre 同時決定 ===
        first_at_done = False
        current_mode = KAGEI_AT_LABEL
        prev_status = None

        yuri_diff = 0.0          # 現有利区間累積差枚
        chain_net = 0.0          # 現連荘の純増累計 (Dedama - Start*invest)
        jyoi_unlocked = False    # 1000枚 or 2300枚 到達フラグ
        jyoi_fire_chain_net: dict = {}  # 行 index → 上位AT 発動瞬間の下位純増累計

        for pos, i in enumerate(indices):
            t = out.at[i, "Status"]
            try:
                s = int(out.at[i, "Start"])
            except Exception:
                s = 0
            try:
                d = int(out.at[i, "Dedama"])
            except Exception:
                d = 0
            delta = d - s * invest_per_g

            state = ""

            if t == "RB":
                # CZ 信号: 次AT が CZ成功/失敗 判定
                nxt = indices[pos + 1] if pos + 1 < len(indices) else None
                nxt_status = out.at[nxt, "Status"] if nxt is not None else None
                if nxt_status in config.AT_STATUS:
                    try:
                        nxt_s = int(out.at[nxt, "Start"])
                    except Exception:
                        nxt_s = 0
                    state = (CZ_SUCCESS_LABEL if nxt_s <= CZ_SUCCESS_NEXT_START_MAX
                             else CZ_FAIL_LABEL)
                else:
                    state = CZ_FAIL_LABEL
                # RB は連荘終了契機 (chain_net リセット)
                out.at[i, "Yuri_Diff_Pre"] = int(round(yuri_diff))
                yuri_diff += delta
                chain_net = 0.0
                # 有利区間差枚到達で unlock (RB Start 消化中でも)
                if yuri_diff >= JYOI_UNLOCK_YURI_DIFF:
                    jyoi_unlocked = True
                prev_status = "RB"
                out.at[i, "State"] = state
                continue

            if t not in config.AT_STATUS:
                out.at[i, "Yuri_Diff_Pre"] = int(round(yuri_diff))
                yuri_diff += delta
                out.at[i, "State"] = state
                continue

            # === AT (BB/ART) ===
            prev_i = indices[pos - 1] if pos > 0 else None
            prev_t = out.at[prev_i, "Status"] if prev_i is not None else None

            new_chain = False
            jyoi_now = False  # 今回の AT が上位AT 確定か

            if prev_t == "RB" and s <= CZ_SUCCESS_NEXT_START_MAX:
                # CZ成功
                state = KAGEI_AT_LABEL
                current_mode = KAGEI_AT_LABEL
                new_chain = True
            elif not first_at_done:
                # 朝一初AT
                state = MORNING_CHOKUGEKI_LABEL
                current_mode = KAGEI_AT_LABEL
                new_chain = True
            else:
                if s <= 4:
                    state = current_mode
                elif s <= 25:
                    # 上位AT 候補ゾーン
                    if prev_t in config.AT_STATUS and jyoi_unlocked:
                        state = JYOI_AT_LABEL
                        current_mode = JYOI_AT_LABEL
                        jyoi_now = True
                    elif prev_t in config.AT_STATUS:
                        # unlock してない → 引き戻し格下げ
                        state = HIKIMODOSHI_LABEL
                        current_mode = KAGEI_AT_LABEL
                        new_chain = True
                    else:
                        # prev=RB 経由の 5-25 = 直撃扱い
                        state = CHOKUGEKI_LABEL
                        current_mode = KAGEI_AT_LABEL
                        new_chain = True
                elif s <= 80:
                    if prev_t in config.AT_STATUS:
                        state = HIKIMODOSHI_LABEL
                        current_mode = KAGEI_AT_LABEL
                        new_chain = True
                    else:
                        state = CHOKUGEKI_LABEL
                        current_mode = KAGEI_AT_LABEL
                        new_chain = True
                else:
                    state = CHOKUGEKI_LABEL
                    current_mode = KAGEI_AT_LABEL
                    new_chain = True
            first_at_done = True
            out.at[i, "State"] = state

            # --- Yuri_Diff_Pre 書き込み + yuri_diff 更新 ---
            if jyoi_now:
                # リセット前に発動瞬間の下位純増累計を記録 (第2パスで 3000枚確定判定に使用)
                jyoi_fire_chain_net[i] = chain_net
                # 上位AT 発動 = 有利区間リセット (この行から新区間)
                out.at[i, "Yuri_Diff_Pre"] = 0
                yuri_diff = delta
                # 上位AT 区間中は jyoi_unlocked リセット (次回再到達まで保留)
                jyoi_unlocked = False
                chain_net = 0.0  # 連荘継続だが上位区間以降は再カウント
            else:
                out.at[i, "Yuri_Diff_Pre"] = int(round(yuri_diff))
                yuri_diff += delta

            # 連荘純増累計
            if new_chain:
                chain_net = delta
            else:
                chain_net += delta

            # unlock 判定 (連荘純増 or 有利区間差枚)
            if chain_net >= JYOI_UNLOCK_RENCHAN or yuri_diff >= JYOI_UNLOCK_YURI_DIFF:
                jyoi_unlocked = True

            prev_status = "AT"

        # === 第2パス: 上位AT 500枚検証 → 連続上位AT 合計純増<500 なら格下げ ===
        # ただし発動時下位累計が JYOI_CONFIRM_THRESHOLD (3000) 以上なら確定、格下げ免除
        downgrade_happened = False
        i_pos = 0
        while i_pos < len(indices):
            i = indices[i_pos]
            if out.at[i, "State"] == JYOI_AT_LABEL:
                # 3000枚確定免除: 上位AT 連続区間をスキップして次に進む
                if jyoi_fire_chain_net.get(i, 0.0) >= JYOI_CONFIRM_THRESHOLD:
                    j_pos = i_pos + 1
                    while j_pos < len(indices):
                        j = indices[j_pos]
                        nx_t = out.at[j, "Status"]
                        try:
                            nx_s = int(out.at[j, "Start"])
                        except Exception:
                            nx_s = 0
                        if nx_t in config.AT_STATUS and nx_s <= 4:
                            j_pos += 1
                        else:
                            break
                    i_pos = j_pos
                    continue
                total_d = int(out.at[i, "Dedama"] or 0)
                j_pos = i_pos + 1
                while j_pos < len(indices):
                    j = indices[j_pos]
                    nx_t = out.at[j, "Status"]
                    try:
                        nx_s = int(out.at[j, "Start"])
                    except Exception:
                        nx_s = 0
                    if nx_t in config.AT_STATUS and nx_s <= 4:
                        total_d += int(out.at[j, "Dedama"] or 0)
                        j_pos += 1
                    else:
                        break
                if total_d < JYOI_THRESHOLD_PAYOUT:
                    prev_i = indices[i_pos - 1] if i_pos > 0 else None
                    prev_t = out.at[prev_i, "Status"] if prev_i is not None else None
                    if prev_t in config.AT_STATUS:
                        out.at[i, "State"] = HIKIMODOSHI_LABEL
                    else:
                        out.at[i, "State"] = CHOKUGEKI_LABEL
                    for kk in range(i_pos + 1, j_pos):
                        ki = indices[kk]
                        if out.at[ki, "State"] == JYOI_AT_LABEL:
                            out.at[ki, "State"] = KAGEI_AT_LABEL
                    downgrade_happened = True
                i_pos = j_pos
            else:
                i_pos += 1

        # === 第3パス: 第2パスで格下げ発生時のみ Yuri_Diff_Pre を再計算 ===
        # (格下げ行がリセット契機でなくなるので、それ以降の累積がずれる)
        if downgrade_happened:
            yuri_diff = 0.0
            in_reset = False
            for i in indices:
                try:
                    s = int(out.at[i, "Start"])
                except Exception:
                    s = 0
                try:
                    d = int(out.at[i, "Dedama"])
                except Exception:
                    d = 0
                delta = d - s * invest_per_g
                state = out.at[i, "State"]
                is_reset_now = (state == JYOI_AT_LABEL)
                if is_reset_now:
                    if not in_reset:
                        yuri_diff = 0.0
                    out.at[i, "Yuri_Diff_Pre"] = 0
                    yuri_diff += delta
                else:
                    out.at[i, "Yuri_Diff_Pre"] = int(round(yuri_diff))
                    yuri_diff += delta
                in_reset = is_reset_now

    return out
