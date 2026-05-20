"""機種別の実践値公開フローを 1 コマンドで回す統合ラッパー。

前提:
- 解析xlsx が <kaiseki_dir>/output/ に最新の状態で出力済 (機種ごとの解析実行.py 実行後)
- machines.json に機種エントリが登録済

実行内容 (順番):
  1. generate_jissenchi.py     → Obsidian Vault に実践値解析 md / data md / data json 出力
  2. html_export                → 機種フォルダに dashboard.html (limit=1 単機種) 出力
  3. post_dashboard.py 相当      → WP media にアップ + machine_shukei (shortcode) 更新

使い方:
  python publish_machine.py --machine bigdream
  python publish_machine.py --machine bigdream --skip-wp     # Obsidian + dashboard まで
  python publish_machine.py --machine bigdream --only-wp     # 既存 dashboard.html を WP に上げ直すだけ
  python publish_machine.py --list                            # 登録機種一覧
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
import machines as _m  # noqa: E402

WP_AUTOMATION = r"C:\Users\tqkqdqrq\Desktop\clawdbot\WP編集\wp-automation"


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    if cwd:
        print(f"    (cwd: {cwd})")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"command failed (exit {r.returncode})")


def step_jissenchi(short: str) -> None:
    _run([sys.executable, "generate_jissenchi.py", "--machine", short], cwd=REPO_ROOT)


def step_dashboard(cfg: dict) -> None:
    out = _m.dashboard_path(cfg)
    in_dir = _m.xlsx_dir(cfg)
    _run([
        sys.executable, "-m", "kaiseki_core.viewer.html_export",
        "--output-dir", in_dir,
        "--out", out,
        "--limit", "1",
    ], cwd=REPO_ROOT)


def step_wp(short: str) -> None:
    _run([sys.executable, "post_dashboard.py", "--machine", short], cwd=WP_AUTOMATION)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--machine", help="機種短縮名 (machines.json の short)")
    p.add_argument("--list", action="store_true", help="登録機種一覧表示")
    p.add_argument("--skip-wp", action="store_true",
                   help="WP アップロード手前まで (Obsidian + dashboard 生成のみ)")
    p.add_argument("--only-wp", action="store_true",
                   help="既存 dashboard.html を WP に上げ直すだけ")
    p.add_argument("--skip-jissenchi", action="store_true",
                   help="Obsidian 出力をスキップ (dashboard + WP のみ)")
    args = p.parse_args()

    if args.list:
        for short, cfg in _m.MACHINES.items():
            print(f"  {short:<14} {cfg['full_name']}  ({cfg['kaiseki_dir']})")
        return 0

    if not args.machine:
        p.error("--machine が必要 (または --list)")

    cfg = _m.resolve(args.machine)
    print(f"=== publish: {cfg['full_name']} ({cfg['short']}) ===")

    if args.only_wp:
        step_wp(cfg["short"])
        return 0

    if not args.skip_jissenchi:
        step_jissenchi(cfg["short"])
    step_dashboard(cfg)
    if not args.skip_wp:
        step_wp(cfg["short"])

    print(f"\n=== done: {cfg['full_name']} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
