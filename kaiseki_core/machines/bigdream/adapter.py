"""Lビックドリーム (スマスロ, CZ→AT 型) アダプタ。

連荘判定 (台日内時系列):
- AT (BB/ART) 行で:
    - 直前行が RB (CZ経由) → 新連荘
    - Start > 80 (直撃)     → 新連荘
    - 朝一初AT              → 新連荘
    - それ以外 (Start≤80 かつ直前AT) → 連荘継続 (連続BB/上位AT/引き戻し)
"""

from __future__ import annotations
import pandas as pd

from ..base import BaseAdapter
from ...core.payout import Financials
from ...core.excel_base import SheetSpec
from ...core.yuri_diff import attach_chain_yuri_diff_pre

from . import config
from .state import (assign_states, JYOI_AT_LABEL, KAGEI_AT_LABEL,
                     HIKIMODOSHI_LABEL, CHOKUGEKI_LABEL, MORNING_CHOKUGEKI_LABEL)

CHOKUGEKI_STATES = (CHOKUGEKI_LABEL, MORNING_CHOKUGEKI_LABEL)


def _expand_azukun_id(raw_id: str, year: int) -> str:
    parts = str(raw_id).split("_", 3)
    if len(parts) != 4:
        return str(raw_id)
    m_str, d_str, hall, machine = parts
    try:
        m = int(m_str)
        d = int(d_str)
    except ValueError:
        return str(raw_id)
    return f"{hall}_{machine}_{year:04d}{m:02d}{d:02d}"


# 連荘終了とみなす Start G しきい値
NEW_CHAIN_START_G = 80

# 上位AT発動の Start G 範囲 (これも新連荘起点扱い = 有利区間リセット契機)
JYOI_START_G_MIN = 5
JYOI_START_G_MAX = 25


class BigdreamAdapter(BaseAdapter):
    name = "bigdream"
    machine_type = "AT"
    financials = Financials(
        invest_per_g=config.INVEST_PER_G,
        streak_min_length=config.STREAK_MIN_LENGTH,
    )

    def column_map(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rename_map = {"スタート": "Start", "出玉": "Dedama", "種別": "Status",
                      "時間": "Time", "状態": "State"}
        for k, v in rename_map.items():
            if k in out.columns and v not in out.columns:
                out = out.rename(columns={k: v})
        if "機種名" in out.columns:
            out = out.drop(columns=["機種名"])
        if "State" not in out.columns:
            out["State"] = ""
        out = out[out["Status"].isin(config.VALID_STATUS)].reset_index(drop=True)
        out["ID"] = out["ID"].map(lambda x: _expand_azukun_id(x, config.DEFAULT_YEAR))
        out["State"] = out["State"].fillna("").astype(str)
        return out

    def is_streak_break(self, prev: pd.Series, curr: pd.Series) -> bool:
        return True  # detect() override

    def detect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # State 判定 (精査UIと同じロジック: Start5-25→上位AT候補、連チャン累計≥500で確定)
        df = assign_states(df)
        work = df.copy()
        work["Chain_ID"] = 0
        work["Chain_Position"] = 0
        work["Chain_Length"] = 0
        work["Is_AT"] = work["Status"].isin(config.AT_STATUS)
        work["Chain_Pre_Normal_G"] = 0        # AT初動のみ: 連荘前通常G累計

        chain_rows: list[dict] = []
        chain_via_cz: dict[int, bool] = {}  # chain_id → 起点で prev_status==RB だったか
        next_chain_id = 1
        fin = self.financials

        # CSV読み込み順 (Original_Order) を維持 → ID初出順で groups を作る
        # (前処理で CSV は「ID毎にまとまり + 各ID内 時刻昇順」になっている前提)
        sorted_idx = work.sort_values("Original_Order").index.tolist()

        groups: dict[str, list[int]] = {}
        for i in sorted_idx:
            key = work.at[i, "ID"]
            groups.setdefault(key, []).append(i)

        for key, indices in groups.items():
            current_chain_indices: list[int] = []
            current_chain_id: int | None = None
            chain_number_in_day = 0
            normal_g_cumulative = 0    # AT到達でリセットされる通常G累計
            prev_status = None

            for i in indices:
                start_g = int(work.at[i, "Start"])
                status = work.at[i, "Status"]

                if status == "RB":
                    normal_g_cumulative += start_g  # CZ自身のStartも通常G消化として累積
                    prev_status = "RB"
                    continue

                # AT (BB/ART)
                # 連チャン = 起点AT + 下位AT/上位AT/引き戻し まとめて1連荘 (ユーザー仕様)
                # 上位AT発動 (prev=AT+Start5-25) は連荘継続。有利区間リセット契機は別軸 (has_jyoi で管理)
                is_new_chain = (
                    current_chain_id is None         # 朝一初AT
                    or prev_status == "RB"           # CZ介在 = 必ず新連荘 (CZ成功/失敗ともに)
                    or start_g > NEW_CHAIN_START_G   # 直撃 (>80)
                )

                if is_new_chain and current_chain_indices:
                    chain_rows.append(
                        self._finalize_chain(work, current_chain_indices, current_chain_id,
                                             chain_number_in_day, fin)
                    )
                    current_chain_indices = []

                if is_new_chain:
                    current_chain_id = next_chain_id
                    next_chain_id += 1
                    chain_number_in_day += 1
                    work.at[i, "Chain_Pre_Normal_G"] = normal_g_cumulative + start_g
                    # CZ成功のみ: prev=RB かつ Start≤3 (state.py の CZ_SUCCESS_NEXT_START_MAX)
                    chain_via_cz[current_chain_id] = (prev_status == "RB" and start_g <= 3)
                    # AT到達リセット
                    normal_g_cumulative = 0

                work.at[i, "Chain_ID"] = current_chain_id
                current_chain_indices.append(i)
                prev_status = "AT"

            if current_chain_indices:
                chain_rows.append(
                    self._finalize_chain(work, current_chain_indices, current_chain_id,
                                         chain_number_in_day, fin)
                )

        for cr in chain_rows:
            cid = cr["chain_id"]
            cl = cr["chain_length"]
            chain_idx = work[work["Chain_ID"] == cid].sort_values("Original_Order").index
            for pos, ix in enumerate(chain_idx, start=1):
                work.at[ix, "Chain_Position"] = pos
                work.at[ix, "Chain_Length"] = cl

        chains_df = pd.DataFrame(chain_rows)

        # 有利区間差枚は state.py 内で1パス統合計算済 (Yuri_Diff_Pre 列)
        # chain 起点の値だけ chains_df に転記
        chains_df = attach_chain_yuri_diff_pre(chains_df, work)

        # chain_number を2軸で持つ:
        #   chain_number_morning: 台日内連番 (現状の chain_number)
        #   chain_number_cut:     切断後N回目 (上位AT区間終了でリセット)
        if not chains_df.empty:
            jyoi_chain_ids = set(
                work.loc[work["State"] == JYOI_AT_LABEL, "Chain_ID"].unique()
            )
            chains_df["has_jyoi"] = chains_df["chain_id"].isin(jyoi_chain_ids)
            # 旧 chain_number = 朝N回目 (台日内連番) として保存
            chains_df["chain_number_morning"] = chains_df["chain_number"]
            # 上位AT終了察知フラグ: 直前連荘で上位AT区間あり = この連荘から新有利区間
            chains_df = chains_df.sort_values(["id", "chain_id"]).reset_index(drop=True)
            chains_df["prev_chain_has_jyoi"] = (
                chains_df.groupby("id")["has_jyoi"].shift(1).fillna(False).astype(bool)
            )
            # 直前 chain が直撃 (or 朝一直撃) だったか (= 直撃直後 chain フラグ)
            chains_df["prev_chain_chokugeki"] = (
                chains_df.groupby("id")["first_state"].shift(1)
                .isin(CHOKUGEKI_STATES).astype(int)
            )
            # 次 chain が引き戻しか (= この chain 後に引き戻し発生)
            chains_df["next_is_hikimodoshi"] = (
                chains_df.groupby("id")["first_state"].shift(-1)
                .eq("引き戻し").astype(int)
            )
            # 上位一撃枚数: chain 内で State='上位AT' の行 (work から抽出) の Dedama 合計
            jyoi_label = JYOI_AT_LABEL
            jyoi_sums = (
                work[work["State"] == jyoi_label]
                .groupby("Chain_ID")["Dedama"].sum()
            )
            chains_df["jyoi_chuugeki"] = (
                chains_df["chain_id"].map(jyoi_sums).fillna(0).astype(int)
            )
            # 切断後N回目 を新規計算 (連荘判定にStart 5-25含めた版)
            # 仕様:
            #   - 上位AT区間 (first_g 5-25 かつ has_jyoi=True): chain_number_cut=0 (集計外)
            #   - 短発上位AT (first_g 5-25 かつ has_jyoi=False): cut=1 (切断後1回目相当)
            #     (実機: 上位AT発動で有利区間リセット、連チャン<500で短く終わった = 切断後最初のAT)
            #   - その他連荘 (上位AT発動なし、前に切断経験あり): cut カウント +1
            #   - 切断未経験の連荘: cut=0
            # chain_id 順 = detect の時系列付与順 = Data 順と一致
            chains_df = chains_df.sort_values("chain_id").reset_index(drop=True)
            cut_numbers: list[int] = []
            cycle_g_pres: list = []
            chain_no = 0
            seen_jyoi = False
            seen_break_morning = False  # 台日内で 上位AT or 直撃 を経験したか
            chain_number_morning_safe_list: list = []
            prev_id = None
            cycle_g = 0       # 有利区間内累積通常G (上位AT終了でリセット)
            cycle_total_g = 0 # 有利区間内累積全G (通常G + AT中G)
            cycle_total_g_pres: list = []
            # 前回有利区間総G (直前 has_jyoi 連荘終了時の cycle_total_g スナップショット)
            prev_yuri_total_gs: list[float] = []
            current_prev_yuri_total_g = None
            for _, row in chains_df.iterrows():
                tid = row["id"]
                if tid != prev_id:
                    seen_break_morning = False  # 台日切替でリセット
                # 累積G カウンタ リセット契機:
                #   - 台日切替
                #   - 上位AT chain 終了直後 (prev_chain_has_jyoi)
                #   - 直撃 / 朝一直撃 chain 終了直後 (prev_chain_chokugeki)
                # ※ prev_yuri_total_g (前回有利区間総G) のスナップショットは
                #   上位AT 終了時のみ更新する仕様は維持 (直撃では有利区間扱いしない)
                if (tid != prev_id
                        or row["prev_chain_has_jyoi"]
                        or bool(row["prev_chain_chokugeki"])):
                    if tid != prev_id:
                        chain_no = 0
                        seen_jyoi = False
                        current_prev_yuri_total_g = None  # 台日切替で不明に
                    cycle_g = 0
                    cycle_total_g = 0
                first_g = int(row["first_g"])
                is_jyoi_trigger = (JYOI_START_G_MIN <= first_g <= JYOI_START_G_MAX)
                if row["has_jyoi"]:
                    # 上位AT発動連荘の扱い:
                    # - 台日内で最初の jyoi (seen_jyoi=False) → cut=0
                    #   (これ自体が「切断契機」で、切断「前」の連荘扱い)
                    # - 既に jyoi 経験あり (seen_jyoi=True) → 現サイクル内の N回目として
                    #   chain_no+=1 でカウント。打ち手目線で「切断後N回目で上位AT発展した
                    #   連荘」を集計に含める。
                    if seen_jyoi:
                        chain_no += 1
                        cut_numbers.append(chain_no)
                    else:
                        cut_numbers.append(0)
                    seen_jyoi = True
                elif is_jyoi_trigger:
                    # has_jyoi=False かつ first_g 5-25 = 短発上位AT
                    # (state.py 第2パスで <500枚 のため上位AT判定から格下げされた連荘)
                    chain_no = 1
                    seen_jyoi = True
                    cut_numbers.append(1)
                elif seen_jyoi:
                    chain_no += 1
                    cut_numbers.append(chain_no)
                else:
                    cut_numbers.append(0)
                # 有利区間内累積G (cycle_g_pre = 連荘起点時点)
                # 直撃チェインは自身の前ハマリG (pre_normal_g = first_g 相当) も累積に含める
                # (リセット直後の直撃が 0 になる問題への対処)
                pre_g = int(row["pre_normal_g"])
                at_g = int(round(row["net_payout"] / config.AT_PAYOUT_PER_G)) if config.AT_PAYOUT_PER_G > 0 else 0

                # cycle_g_pre / cycle_total_g_pre は従来通り (NaN 化しない)
                # — 累積G別cut シートで必要なため
                if row["first_state"] in CHOKUGEKI_STATES:
                    cycle_g_pres.append(int(cycle_g + pre_g))
                    cycle_total_g_pres.append(int(cycle_total_g + pre_g + at_g))
                else:
                    cycle_g_pres.append(int(cycle_g))
                    cycle_total_g_pres.append(int(cycle_total_g))

                # chain_number_morning_safe のみ「切断/直撃 経験後 NaN」で集計除外
                if seen_break_morning:
                    chain_number_morning_safe_list.append(None)
                else:
                    chain_number_morning_safe_list.append(int(row["chain_number_morning"]))
                    if row["has_jyoi"] or row["first_state"] in CHOKUGEKI_STATES:
                        seen_break_morning = True

                # 前回有利区間累積G (chain 起点時点で参照可能な値)
                prev_yuri_total_gs.append(current_prev_yuri_total_g
                                          if current_prev_yuri_total_g is not None else float('nan'))

                cycle_g += pre_g
                cycle_total_g += pre_g + at_g

                # has_jyoi=TRUE 連荘終了時 = サイクル終了 → 次サイクル用にスナップショット保存
                if row["has_jyoi"]:
                    current_prev_yuri_total_g = cycle_total_g

                prev_id = tid
            chains_df["chain_number_cut"] = cut_numbers
            chains_df["chain_number_morning_safe"] = chain_number_morning_safe_list
            chains_df["cycle_g_pre"] = cycle_g_pres
            chains_df["cycle_total_g_pre"] = cycle_total_g_pres
            chains_df["prev_yuri_total_g"] = prev_yuri_total_gs

            # 直撃フラグ + 次 chain 起点情報 (台日内シフト)
            chains_df["is_choku"] = chains_df["first_state"].isin(CHOKUGEKI_STATES).astype(int)
            chains_df["chain_via_cz"] = (
                chains_df["chain_id"].map(chain_via_cz).fillna(False).astype(int)
            )
            chains_df["next_chain_first_state"] = (
                chains_df.groupby("id")["first_state"].shift(-1).fillna("")
            )
            chains_df["next_chain_first_g"] = (
                chains_df.groupby("id")["first_g"].shift(-1)
            )

        # === segment 列を Data シートに付与 (引き戻し分布の元データ) ===
        # segment 区切り: chain 切替 / 引き戻し / 上位AT 開始 / 上位AT 終了直後
        work["Segment_Type"] = ""
        work["Is_Shimoi_Start"] = 0
        work["Is_Jyoi_Start"] = 0
        work["Segment_Dedama"] = 0
        work["Next_Jyoi"] = 0
        work["次イベントStart_G"] = None
        work["次イベント状態"] = ""
        work["次イベント_3_90内"] = 0

        # 台日内の全行 (RB含む) を時系列で持っておく (次イベント検索用)
        work_sorted = work.sort_values("Original_Order")
        all_day_indices: dict = {}
        for key, grp_all in work_sorted.groupby(["Hall", "Date", "Machine_No"], sort=False):
            all_day_indices[key] = list(grp_all.index)

        at_only = work[work["Is_AT"]].sort_values("Original_Order")
        for key_day, grp in at_only.groupby(["Hall", "Date", "Machine_No"], sort=False):
            day_segs: list[dict] = []
            cur_seg: dict | None = None
            prev_state = None
            prev_chain_id = None
            prev_at_idx = None
            for idx, row in grp.iterrows():
                state = row["State"]
                dedama = int(row["Dedama"] or 0)
                chain_id = int(row["Chain_ID"])
                start_g = int(row["Start"] or 0)

                new_seg = False
                seg_type = None
                if cur_seg is None:
                    new_seg = True
                    seg_type = ('上位一撃' if state == '上位AT'
                                else '引き戻し起点' if state == '引き戻し' else 'AT')
                elif chain_id != prev_chain_id:
                    new_seg = True
                    seg_type = '上位一撃' if state == '上位AT' else 'AT'
                elif state == '引き戻し':
                    new_seg = True; seg_type = '引き戻し起点'
                elif state == '上位AT' and prev_state != '上位AT':
                    new_seg = True; seg_type = '上位一撃'
                elif state != '上位AT' and cur_seg["type"] == '上位一撃':
                    new_seg = True; seg_type = 'AT'

                if new_seg:
                    if cur_seg is not None:
                        cur_seg["end_at_idx"] = prev_at_idx
                        day_segs.append(cur_seg)
                    cur_seg = {"start_idx": idx, "type": seg_type,
                               "start_state": state, "start_g": start_g,
                               "chain_id": chain_id, "dedama": 0}
                cur_seg["dedama"] += dedama
                prev_state = state
                prev_chain_id = chain_id
                prev_at_idx = idx
            if cur_seg is not None:
                cur_seg["end_at_idx"] = prev_at_idx
                day_segs.append(cur_seg)

            # 各 segment の「次イベント」(end_at_idx の次の行、RB含む) を特定
            all_idx_list = all_day_indices.get(key_day, [])
            pos_of = {ix: p for p, ix in enumerate(all_idx_list)}
            for seg in day_segs:
                ep = pos_of.get(seg["end_at_idx"])
                if ep is not None and ep + 1 < len(all_idx_list):
                    nxt_idx = all_idx_list[ep + 1]
                    seg["next_event_idx"] = nxt_idx
                    seg["next_event_start_g"] = int(work.at[nxt_idx, "Start"] or 0)
                    seg["next_event_state"] = str(work.at[nxt_idx, "State"] or "")
                else:
                    seg["next_event_idx"] = None
                    seg["next_event_start_g"] = None
                    seg["next_event_state"] = ""

            for i, seg in enumerate(day_segs):
                si = seg["start_idx"]
                work.at[si, "Segment_Type"] = seg["type"]
                if seg["type"] == "上位一撃":
                    work.at[si, "Is_Jyoi_Start"] = 1
                else:
                    work.at[si, "Is_Shimoi_Start"] = 1
                work.at[si, "Segment_Dedama"] = seg["dedama"]
                nxt = day_segs[i + 1] if i + 1 < len(day_segs) else None
                ns_state = nxt["start_state"] if nxt else ""
                work.at[si, "Next_Jyoi"] = 1 if ns_state == "上位AT" else 0

                # 次イベント (RB含む全行ベース) — 引き戻し分布の 3-90内当選 判定用
                # 3-90G 以内に「何かしら当たった」総数 (上位AT も含む)。
                neg = seg.get("next_event_start_g")
                nes = seg.get("next_event_state", "")
                if neg is not None:
                    work.at[si, "次イベントStart_G"] = neg
                work.at[si, "次イベント状態"] = nes
                if neg is not None and 3 <= int(neg) <= 90:
                    work.at[si, "次イベント_3_90内"] = 1

        # Data シートは CSV読み込み順 (Original_Order) のまま出力
        work = work.sort_values("Original_Order").reset_index(drop=True)

        return work, chains_df

    def _finalize_chain(self, work, indices, chain_id, chain_number, fin):
        starts = [int(work.at[i, "Start"]) for i in indices]
        payouts = [int(work.at[i, "Dedama"]) for i in indices]
        states = [str(work.at[i, "State"]) for i in indices]
        statuses = [str(work.at[i, "Status"]) for i in indices]

        total_g = sum(starts)
        raw_payout = sum(payouts)
        in_chain_invest = total_g * fin.invest_per_g

        # TY = 1連荘の総出玉 (= 連荘内 AT 行 Dedama 合計)
        # 仮想ヤメ / 25G試行 / 閾値ベースの打ち手シミュレーションは TY からは外す。
        # 連荘の境界は Chain_ID 作成側 (detect ループ) で既に確定 — TY ではさらに削らない。
        # 「即やめ想定の期待値」を別に見たい場合は別列・別シートで扱う方針。
        net_payout = raw_payout

        first_idx = indices[0]
        pre_normal_g = int(work.at[first_idx, "Chain_Pre_Normal_G"])

        return {
            "id": str(work.at[first_idx, "ID"]),
            "chain_id": chain_id,
            "chain_number": chain_number,
            "chain_length": len(indices),
            "first_g": starts[0],
            "first_state": states[0],
            "pre_normal_g": pre_normal_g,
            "chain_total_g": total_g,
            "is_streak": len(indices) >= fin.streak_min_length,
            "raw_payout": raw_payout,
            "net_payout": net_payout,
            "in_chain_invest": in_chain_invest,
            "net_diff": raw_payout - in_chain_invest - pre_normal_g * fin.invest_per_g,
        }

    def sheets(self) -> list[SheetSpec]:
        from .sheets import (
            build_at_tenjou_summary,
            build_at_tenjou_after_chokugeki,
            build_morning_all_formula, build_morning_all_preview,
            build_cut_all_formula, build_cut_all_preview,
            build_cycle_morning, build_cycle_cut,
            build_chokugeki_distribution, build_chokugeki_formula,
            build_hikimodoshi_distribution, build_hikimodoshi_formula,
            build_prev_yuri_morning_formula, build_prev_yuri_morning_preview,
            rename_headers_to_ja,
        )
        specs = [
            SheetSpec("AT間天井狙い", build_at_tenjou_summary),
            SheetSpec("AT間天井狙い_直撃後", build_at_tenjou_after_chokugeki),
            SheetSpec("AT間_morning", build_morning_all_formula),
            SheetSpec("AT間_morning_値", build_morning_all_preview),
            SheetSpec("AT間_cut", build_cut_all_formula),
            SheetSpec("AT間_cut_値", build_cut_all_preview),
            SheetSpec("累積G別AT間_morning", build_cycle_morning),
            SheetSpec("累積G別AT間_cut", build_cycle_cut),
            SheetSpec("直前有利区間G別期待値", build_prev_yuri_morning_formula),
            SheetSpec("直前有利区間G別期待値_値", build_prev_yuri_morning_preview),
            SheetSpec("直撃分布", build_chokugeki_formula),
            SheetSpec("直撃分布_値", build_chokugeki_distribution),
            SheetSpec("引き戻し分布", build_hikimodoshi_formula),
            SheetSpec("引き戻し分布_値", build_hikimodoshi_distribution),
            # rename は全 formula 生成の後 (build_g_bin_chain_formula_sheet が
            # ChainData ヘッダーを英語名で引くため、formula 生成前にリネームすると失敗する)
            SheetSpec("__rename_headers__", rename_headers_to_ja),
        ]
        return specs


def build() -> BigdreamAdapter:
    return BigdreamAdapter()
