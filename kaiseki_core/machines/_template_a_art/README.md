# _template_a_art (後日デュオ統合用)

このディレクトリは A+ART 系（デュオ等）を core 上に移植する際の置き場。
現状は空。デュオ統合タスクで以下を実装する:

- `adapter.py`: `Financials(coin_hold_g_per_50=25.3, streak_min_length=2)` 等
- `is_streak_break`: `curr.Start > HEAVEN_THRESHOLD(=35)`
- `sheets()`: 天国移行率 / クロス集計 / 3連基準2連スルー / 跨ぎ2連 期待値

参照: `デュオ解析Python\デュオ解析_v2.py` の `analyze_chains()` と `write_excel()`。
