# ==========================================
# ▼ 設定エリア（ここだけ変えればOK）
# ==========================================

# 1. 各段階のデータ設定
# 項目: [段階名, 直前の当選率(これを外すと到達), モード期待値(円), 純増(%) ]
# ※純増は「8.5%」なら「0.085」と入力
data_list = [
    # 0スルーを22.7%で当てずに、77.3%で1スルーへ到達
    {"name": "1スルー", "win_rate_prev": 0.227, "value": 2210, "net_increase": 0.089}, 
    
    # 1スルーを39.9%で当てずに、2スルーへ到達
    {"name": "2スルー", "win_rate_prev": 0.399, "value": 2168, "net_increase": 0.085},
    
    # 2スルーを50.4%で当てずに、3スルーへ到達
    {"name": "3スルー", "win_rate_prev": 0.504, "value": 14,   "net_increase": 0.001},
    
    # 3スルーを36.3%で当てずに、4スルーへ到達
    {"name": "4スルー", "win_rate_prev": 0.363, "value": 9245, "net_increase": 0.286},
]

# 2. 全体設定
flow_occurrence_rate = 31.34  # フロー発生率 (%)
base_coin_price = 60          # 1枚あたりの単価 (円)

# 3. 調整値（最後の微調整用）
adjust_games = 68.9           # 追加するゲーム数 (+G)
adjust_profit = -20           # 減算する利益 (-円)


# ==========================================
# ▼ 計算ロジック（ここは触らなくてOK）
# ==========================================
current_reach_rate = 1.0  # 初期到達率 (100%)
total_flow_value = 0      # フロー内の積上げ期待値
total_flow_games = 0      # フロー内の積上げゲーム数

print(f"--- 【フロー内訳】 ---")
for data in data_list:
    # 1. 到達率の更新 (前の段階をスルーする確率を掛ける)
    # ※最初の段階(1スルー)は、0スルーの当選率(win_rate_prev)を外して到達する
    reach_rate = current_reach_rate * (1 - data["win_rate_prev"])
    
    # 2. 貢献額 (モード期待値 × 到達率)
    contribution_value = data["value"] * reach_rate
    
    # 3. 必要ゲーム数 (モード期待値 ÷ (60円 × 純増))
    # 純増ごとの1G単価
    g_unit_price = base_coin_price * data["net_increase"]
    required_games = data["value"] / g_unit_price if g_unit_price > 0 else 0
    
    # 4. 実質貢献ゲーム数 (必要G数 × 到達率)
    contribution_games = required_games * reach_rate
    
    # 加算
    total_flow_value += contribution_value
    total_flow_games += contribution_games
    
    # 次のループのために到達率を更新しない（このロジックでは、次の段階の計算時に更新する）
    # ただし、表示用に変数を保持
    print(f"[{data['name']}] 到達率: {reach_rate*100:.1f}% | 貢献額: +{contribution_value:.0f}円 | 実質G数: {contribution_games:.1f}G")
    
    current_reach_rate = reach_rate # 次のステップの基準になる

# === 最終計算（発生率と調整値を反映） ===
# 発生率を掛ける
final_base_value = total_flow_value * (flow_occurrence_rate / 100)
final_base_games = total_flow_games * (flow_occurrence_rate / 100)

# 調整値を適用
final_profit = final_base_value + adjust_profit
final_games = final_base_games + adjust_games

# 機械割算出
total_investment = final_games * base_coin_price
total_return = total_investment + final_profit
machine_rate = (total_return / total_investment) * 100 if total_investment > 0 else 0

print(f"\n--- 【最終結果】 ---")
print(f"着席時の平均期待値 : +{final_profit:.1f} 円")
print(f"着席時の平均G数    : {final_games:.1f} G")
print(f"機械割 (出玉率)    : {machine_rate:.2f} %")