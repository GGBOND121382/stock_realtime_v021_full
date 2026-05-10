#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch extract selected stock / ETF folders from data88 daily .7z archives.

Default symbols are the user's watchlist in this project. Output is one zip per
symbol per trading date, without persistent intermediate extracted folders.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import py7zr
from py7zr.io import BytesIOFactory

from extract_data88_selected import infer_date, normalize_archive_name, normalize_symbol


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


MAINBOARD_CODES = [
    "002601",
    "600438",
    "601138",
    "002460",
    "603259",
    "002311",
    "600406",
    "601179",
    "600312",
    "000657",
    "002080",
    "600176",
    "601985",
    "601899",
    "002261",
    "600096",
    "002895",
    "601100",
    "600276",
    "600309",
    "002518",
    "002297",
    "600919",
    "600361",
    "002270",
    "601567",
    "603308",
    "002028",
    "600885",
    "600030",
    "601818",
    "601336",
    "605499",
    "601390",
    "601186",
    "600016",
    "000786",
    "002128",
    "600522",
    "600487",
    "002364",
    "002714",
]

ETF_CODES = [
    "159566",
    "159320",
    "588210",
    "562550",
    "516510",
    "159201",
    "515880",
    "159507",
    "513980",
    "515180",
    "159595",
    "159300",
    "159361",
    "510510",
    "159941",
    "515650",
    "563020",
    "513400",
    "513650",
]


def default_symbols(include_stocks: bool = True, include_etfs: bool = True) -> List[str]:
    codes: List[str] = []
    if include_stocks:
        codes.extend(MAINBOARD_CODES)
    if include_etfs:
        codes.extend(ETF_CODES)
    return dedupe_symbols(normalize_symbol(code) for code in codes)


def dedupe_symbols(symbols: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for symbol in symbols:
        symbol = normalize_symbol(symbol)
        if symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


def parse_extra_symbols(text: str | None) -> List[str]:
    if not text:
        return []
    return [normalize_symbol(token) for token in text.replace(";", ",").split(",") if token.strip()]


def read_symbols_file(path: str | None) -> List[str]:
    if not path:
        return []
    symbols = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            symbols.append(normalize_symbol(token))
    return symbols


def resolve_archives(args: argparse.Namespace) -> List[Path]:
    if args.archive:
        archives = [Path(p.strip()) for p in args.archive.replace(";", ",").split(",") if p.strip()]
    else:
        archive_dir = Path(args.archive_dir)
        globber = archive_dir.glob if args.no_recursive else archive_dir.rglob
        archives = sorted(globber(args.pattern))
    missing = [str(p) for p in archives if not p.exists()]
    if missing:
        raise FileNotFoundError(f"archives not found: {missing}")
    if not archives:
        raise FileNotFoundError("no archives found")
    return archives


def list_archive_entries(archive: Path, date: str, symbols: Sequence[str]) -> Dict[str, List[Tuple[str, int]]]:
    wanted_prefix = {symbol: f"{date}\\{symbol}\\" for symbol in symbols}
    result: Dict[str, List[Tuple[str, int]]] = {symbol: [] for symbol in symbols}
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        for info in zf.list():
            name = normalize_archive_name(info.filename)
            for symbol, prefix in wanted_prefix.items():
                if name.startswith(prefix):
                    size = int(getattr(info, "uncompressed", 0) or 0)
                    result[symbol].append((info.filename, size))
                    break
    return result


def chunks(items: Sequence[str], n: int) -> Iterable[List[str]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield list(items[i : i + n])


def write_zip_from_factory(
    zip_path: Path,
    symbol: str,
    entries: Sequence[Tuple[str, int]],
    factory: BytesIOFactory,
    overwrite: bool,
) -> None:
    if zip_path.exists() and not overwrite:
        print(f"skip existing zip: {zip_path}")
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, _size in entries:
            product = factory.get(name)
            product.seek(0)
            data = product.read()
            arcname = normalize_archive_name(name)
            parts = arcname.split("\\")
            # Keep symbol folder inside the zip: 002714.SZ/行情.csv
            if len(parts) >= 3:
                arcname = "\\".join(parts[1:])
            zf.writestr(arcname.replace("\\", "/"), data)
    print(f"wrote {zip_path}")


def extract_archive(args: argparse.Namespace, archive: Path, symbols: Sequence[str]) -> dict:
    date = infer_date(archive, args.date)
    zip_root = Path(args.zip_dir) if args.zip_dir else SAVED_DATA_DIR / "data88_selected" / date / "_zip"
    print(f"\n### archive={archive} date={date} symbols={len(symbols)}")
    entries_by_symbol = list_archive_entries(archive, date, symbols)
    found_symbols = [s for s in symbols if entries_by_symbol.get(s)]
    missing_symbols = [s for s in symbols if not entries_by_symbol.get(s)]
    if missing_symbols:
        print(f"missing in archive ({len(missing_symbols)}): {', '.join(missing_symbols)}")

    returncodes = {s: 1 for s in missing_symbols}
    for batch_symbols in chunks(found_symbols, args.chunk_size):
        batch_entries = []
        batch_size = 0
        for symbol in batch_symbols:
            entries = entries_by_symbol[symbol]
            batch_entries.extend(name for name, _size in entries)
            batch_size += sum(size for _name, size in entries)
        limit = int(args.memory_limit_mb) * 1024 * 1024
        if batch_size > limit:
            raise MemoryError(
                f"batch {batch_symbols} needs {batch_size} bytes, above --memory-limit-mb={args.memory_limit_mb}; "
                "reduce --chunk-size or raise memory limit"
            )

        print(f"extract batch size={len(batch_symbols)} files={len(batch_entries)} bytes={batch_size}")
        factory = BytesIOFactory(limit)
        with py7zr.SevenZipFile(archive, mode="r") as zf:
            zf.extract(targets=batch_entries, factory=factory)

        for symbol in batch_symbols:
            zip_path = zip_root / f"{date}_{symbol}.zip"
            write_zip_from_factory(zip_path, symbol, entries_by_symbol[symbol], factory, args.overwrite)
            returncodes[symbol] = 0

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "archive": str(archive.resolve()),
        "archive_size": archive.stat().st_size,
        "date": date,
        "zip_root": str(zip_root.resolve()),
        "symbol_count": len(symbols),
        "found_count": len(found_symbols),
        "missing_count": len(missing_symbols),
        "returncodes": returncodes,
    }
    manifest_path = zip_root / f"{date}_batch_extract_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch extract selected stock/ETF zip packages from data88 .7z archives")
    p.add_argument("--archive", help="Comma separated archive paths; if omitted scan --archive-dir")
    p.add_argument("--archive-dir", default="stock_realtime/PurchasedData")
    p.add_argument("--pattern", default="*.7z")
    p.add_argument("--no-recursive", action="store_true", help="Only scan archive-dir itself; default scans subdirs too")
    p.add_argument("--date", help="Override inner date folder for all archives")
    p.add_argument("--zip-dir", help="Default: saved_data/data88_selected/<date>/_zip")
    p.add_argument("--symbols", help="Extra comma separated symbols")
    p.add_argument("--symbols-file", help="Extra one-symbol-per-line file")
    p.add_argument("--stocks-only", action="store_true")
    p.add_argument("--etfs-only", action="store_true")
    p.add_argument("--chunk-size", type=int, default=8, help="Symbols extracted from one archive pass")
    p.add_argument("--memory-limit-mb", type=int, default=768, help="Max uncompressed bytes per chunk")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list-archives", action="store_true", help="List matched archives and exit")
    p.add_argument("--write-symbols-file", help="Write resolved watchlist and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    include_stocks = not args.etfs_only
    include_etfs = not args.stocks_only
    symbols = default_symbols(include_stocks=include_stocks, include_etfs=include_etfs)
    symbols = dedupe_symbols([*symbols, *parse_extra_symbols(args.symbols), *read_symbols_file(args.symbols_file)])

    if args.write_symbols_file:
        path = Path(args.write_symbols_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
        print(f"wrote {path}: {len(symbols)} symbols")
        return

    archives = resolve_archives(args)
    if args.list_archives:
        for archive in archives:
            print(archive)
        print(f"archives={len(archives)}")
        return

    summary = []
    for archive in archives:
        summary.append(extract_archive(args, archive, symbols))

    grouped = defaultdict(int)
    for item in summary:
        grouped["archives"] += 1
        grouped["found"] += item["found_count"]
        grouped["missing"] += item["missing_count"]
    print(f"\nDONE archives={grouped['archives']} found_symbol_dates={grouped['found']} missing_symbol_dates={grouped['missing']}")


if __name__ == "__main__":
    main()
