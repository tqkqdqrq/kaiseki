"""CLI: python -m kaiseki_core.core.runner --machine _template_at --input ... --output ..."""

from __future__ import annotations
import argparse
import importlib
import sys

from .io_csv import read_csv_auto
from .normalize import normalize
from .cache import load_or_build
from .excel_base import write_workbook
from ..machines.aliases import resolve


def _load_adapter(name: str):
    resolved = resolve(name)
    mod = importlib.import_module(f"kaiseki_core.machines.{resolved}.adapter")
    if not hasattr(mod, "build"):
        raise AttributeError(f"machines.{resolved}.adapter has no build() function")
    return mod.build()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaiseki_core")
    parser.add_argument("--machine", required=True, help="機種名 (例: _template_at, AT機)")
    parser.add_argument("--input", required=True, help="入力CSVパス")
    parser.add_argument("--output", required=True, help="出力xlsxパス")
    parser.add_argument("--cache", default="", help="pickleキャッシュパス (空ならキャッシュ無効)")
    args = parser.parse_args(argv)

    adapter = _load_adapter(args.machine)
    raw_df = read_csv_auto(args.input)
    norm_df = normalize(raw_df, adapter)

    def _build():
        return adapter.detect(norm_df)

    data_df, chains_df = load_or_build(args.cache, _build)
    write_workbook(args.output, data_df, chains_df, adapter.sheets())
    print(f"wrote {args.output}: data={len(data_df)} chains={len(chains_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
