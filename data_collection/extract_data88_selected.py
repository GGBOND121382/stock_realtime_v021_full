#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract selected stock folders from data88 daily .7z archives.

Example:
    python extract_data88_selected.py --archive PurchasedData/20260331.7z --symbols 002714,601899.SH

Default output:
    PurchasedData/selected/20260331/002714.SZ/行情.csv
    PurchasedData/selected/20260331/002714.SZ/逐笔委托.csv
    PurchasedData/selected/20260331/002714.SZ/逐笔成交.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_7Z_CANDIDATES = [
    "7z",
    "7za",
    "7zr",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    r"C:\Program Files\Huawei\PCManager\MobileAppEngine\7za.exe",
    r"C:\Program Files\MobileAppEngine\lib64\7za.exe",
]
PROJECT_DIR = Path(__file__).resolve().parents[1]
SAVED_DATA_DIR = PROJECT_DIR / "saved_data"


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if not text:
        raise ValueError("empty symbol")
    if "." in text:
        code, market = text.split(".", 1)
        return f"{code.zfill(6)}.{market}"
    code = "".join(ch for ch in text if ch.isdigit())
    if len(code) > 6:
        code = code[-6:]
    code = code.zfill(6)
    market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    return f"{code}.{market}"


def parse_symbols(args: argparse.Namespace) -> List[str]:
    symbols: List[str] = []
    if args.symbols:
        for token in args.symbols.replace(";", ",").split(","):
            token = token.strip()
            if token:
                symbols.append(normalize_symbol(token))
    if args.symbols_file:
        path = Path(args.symbols_file)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                symbols.append(normalize_symbol(token))
    deduped: List[str] = []
    seen = set()
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            deduped.append(symbol)
    if not deduped:
        raise ValueError("provide --symbols or --symbols-file")
    return deduped


def find_7z(explicit: Optional[str] = None) -> str:
    candidates = [explicit] if explicit else []
    candidates.extend(DEFAULT_7Z_CANDIDATES)
    for candidate in candidates:
        if not candidate:
            continue
        found = shutil.which(candidate)
        if found:
            return found
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "7z executable not found. Install 7-Zip or pass --sevenzip C:\\path\\to\\7z.exe"
    )


def infer_date(archive: Path, explicit_date: Optional[str] = None) -> str:
    if explicit_date:
        return explicit_date
    stem = archive.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    raise ValueError(f"cannot infer YYYYMMDD date from archive name: {archive.name}")


def run_7z(cmd: Sequence[str], quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    return subprocess.run(cmd, check=False, text=True)


def normalize_archive_name(name: str) -> str:
    return str(name).replace("/", "\\")


def py7zr_available() -> bool:
    try:
        import py7zr  # noqa: F401
    except Exception:
        return False
    return True


def list_symbol_py7zr(archive: Path, date: str, symbol: str) -> int:
    try:
        import py7zr
    except Exception as exc:
        print(f"py7zr import failed: {type(exc).__name__}: {exc}")
        return 2

    prefix = f"{date}\\{symbol}\\"
    matched = []
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        for info in zf.list():
            name = normalize_archive_name(info.filename)
            if name.startswith(prefix):
                matched.append((name, getattr(info, "uncompressed", None)))
    if not matched:
        print(f"no entries matched: {prefix}*")
        return 1
    for name, size in matched:
        print(f"{size if size is not None else ''}\t{name}")
    return 0


def extract_symbol_py7zr(
    archive: Path,
    date: str,
    symbol: str,
    out_root: Path,
    overwrite: bool,
) -> int:
    try:
        import py7zr
    except Exception as exc:
        print(f"py7zr import failed: {type(exc).__name__}: {exc}")
        return 2

    out_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{date}\\{symbol}\\"
    names = []
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        for name in zf.getnames():
            norm = normalize_archive_name(name)
            if norm.startswith(prefix):
                names.append(name)
    if not names:
        print(f"no entries matched: {prefix}*")
        return 1

    if not overwrite:
        names_to_extract = []
        for name in names:
            target = out_root / normalize_archive_name(name)
            if not target.exists():
                names_to_extract.append(name)
        if not names_to_extract:
            print(f"all files already exist for {symbol}; skip")
            return 0
        names = names_to_extract

    print(f"py7zr extracting {len(names)} entries -> {out_root}")
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extract(targets=names, path=out_root)
    return 0


def get_symbol_entries_py7zr(archive: Path, date: str, symbol: str) -> List[tuple[str, int]]:
    import py7zr

    prefix = f"{date}\\{symbol}\\"
    entries: List[tuple[str, int]] = []
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        for info in zf.list():
            name = normalize_archive_name(info.filename)
            if name.startswith(prefix):
                size = int(getattr(info, "uncompressed", 0) or 0)
                entries.append((info.filename, size))
    return entries


def zip_symbol_from_py7zr_memory(
    archive: Path,
    date: str,
    symbol: str,
    zip_path: Path,
    overwrite: bool,
    memory_limit_mb: int,
) -> int:
    try:
        import py7zr
        from py7zr.io import BytesIOFactory
    except Exception as exc:
        print(f"py7zr import failed: {type(exc).__name__}: {exc}")
        return 2

    if zip_path.exists() and not overwrite:
        print(f"skip existing zip: {zip_path}")
        return 0

    entries = get_symbol_entries_py7zr(archive, date, symbol)
    if not entries:
        print(f"no entries matched: {date}\\{symbol}\\*")
        return 1

    total_size = sum(size for _, size in entries)
    memory_limit = int(memory_limit_mb) * 1024 * 1024
    if total_size > memory_limit:
        print(f"matched files need {total_size} bytes, above --memory-limit-mb={memory_limit_mb}")
        return 4

    targets = [name for name, _ in entries]
    factory = BytesIOFactory(memory_limit)
    print(f"py7zr memory extracting {len(targets)} entries; total_uncompressed={total_size}")
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extract(targets=targets, factory=factory)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, _size in entries:
            product = factory.get(name)
            product.seek(0)
            data = product.read()
            arcname = normalize_archive_name(name)
            arcname = arcname.split("\\", 1)[1] if "\\" in arcname else arcname
            zf.writestr(arcname, data)
    print(f"wrote {zip_path}")
    return 0


def list_symbol(archive: Path, sevenzip: str, date: str, symbol: str) -> int:
    pattern = f"{date}\\{symbol}\\*"
    cmd = [sevenzip, "l", str(archive), pattern]
    proc = run_7z(cmd)
    if proc.returncode != 0:
        print(f"list failed for {symbol}; returncode={proc.returncode}")
    return proc.returncode


def extract_symbol(
    archive: Path,
    sevenzip: str,
    date: str,
    symbol: str,
    out_root: Path,
    overwrite: bool,
) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    pattern = f"{date}\\{symbol}\\*"
    mode = "-aoa" if overwrite else "-aos"
    cmd = [sevenzip, "x", str(archive), pattern, f"-o{out_root}", mode, "-y"]
    proc = run_7z(cmd)
    if proc.returncode != 0:
        print(f"extract failed for {symbol}; returncode={proc.returncode}")
    return proc.returncode


def resolve_backend(args: argparse.Namespace) -> tuple[str, Optional[str]]:
    if args.backend == "py7zr":
        if not py7zr_available():
            raise RuntimeError("py7zr is not installed. Install with: python -m pip install py7zr")
        return "py7zr", None
    if args.backend == "7z":
        return "7z", find_7z(args.sevenzip)
    if py7zr_available():
        return "py7zr", None
    return "7z", find_7z(args.sevenzip)


def write_manifest(
    manifest_path: Path,
    archive: Path,
    date: str,
    symbols: Iterable[str],
    out_root: Path,
    backend: str,
    sevenzip: Optional[str],
    mode: str,
    returncodes: dict,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "archive": str(archive.resolve()),
        "archive_size": archive.stat().st_size if archive.exists() else None,
        "date": date,
        "symbols": list(symbols),
        "out_root": str(out_root.resolve()),
        "backend": backend,
        "sevenzip": sevenzip,
        "mode": mode,
        "returncodes": returncodes,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract selected stock folders from data88 daily .7z archives")
    sub = p.add_subparsers(dest="cmd")

    extract = sub.add_parser("extract", help="Extract selected symbols from a daily .7z archive")
    extract.add_argument("--archive", required=True, help="Path to daily archive, e.g. PurchasedData/20260331.7z")
    extract.add_argument("--symbols", help="Comma separated symbols, e.g. 002714,601899.SH")
    extract.add_argument("--symbols-file", help="One symbol per line; comments after # are ignored")
    extract.add_argument("--date", help="Archive inner date folder, default inferred from archive file name")
    extract.add_argument("--out-dir", help="Default: saved_data/data88_selected")
    extract.add_argument("--sevenzip", help="Path to 7z/7za/7zr executable")
    extract.add_argument(
        "--backend",
        choices=["auto", "py7zr", "7z"],
        default="auto",
        help="Default auto: use py7zr when installed, otherwise fall back to 7z executable",
    )
    extract.add_argument("--list-only", action="store_true", help="Only list matching archive entries")
    extract.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted files")
    extract.add_argument(
        "--zip-after",
        action="store_true",
        help="Zip extracted symbol folders after extraction",
    )
    extract.add_argument(
        "--delete-after-zip",
        action="store_true",
        help="Delete extracted symbol folders after --zip-after succeeds",
    )
    extract.add_argument("--zip-dir", help="Default for --zip-after: out-dir / date / _zip")

    extract_zip = sub.add_parser("extract-zip", help="Extract selected symbols from .7z into zip files only")
    extract_zip.add_argument("--archive", required=True, help="Path to daily archive, e.g. PurchasedData/20260331.7z")
    extract_zip.add_argument("--symbols", help="Comma separated symbols, e.g. 002714,601899.SH")
    extract_zip.add_argument("--symbols-file", help="One symbol per line; comments after # are ignored")
    extract_zip.add_argument("--date", help="Archive inner date folder, default inferred from archive file name")
    extract_zip.add_argument("--zip-dir", help="Default: archive parent / selected / date / _zip")
    extract_zip.add_argument("--temp-dir", help="Temporary extraction root; default: system temp")
    extract_zip.add_argument(
        "--use-temp",
        action="store_true",
        help="Use temporary files instead of in-memory py7zr extraction",
    )
    extract_zip.add_argument(
        "--memory-limit-mb",
        type=int,
        default=512,
        help="Max total uncompressed bytes per symbol for no-temp py7zr mode",
    )
    extract_zip.add_argument("--sevenzip", help="Path to 7z/7za/7zr executable")
    extract_zip.add_argument(
        "--backend",
        choices=["auto", "py7zr", "7z"],
        default="auto",
        help="Default auto: use py7zr when installed, otherwise fall back to 7z executable",
    )
    extract_zip.add_argument("--overwrite", action="store_true")

    zip_cmd = sub.add_parser("zip-selected", help="Zip selected/date/symbol folders")
    zip_cmd.add_argument("--selected-dir", default=str(SAVED_DATA_DIR / "data88_selected"), help="Selected root directory")
    zip_cmd.add_argument("--date", required=True, help="YYYYMMDD selected date")
    zip_cmd.add_argument("--symbols", help="Comma separated symbols; default: all symbol dirs under selected/date")
    zip_cmd.add_argument("--symbols-file", help="One symbol per line")
    zip_cmd.add_argument("--zip-dir", help="Default: selected/date/_zip")
    zip_cmd.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def maybe_upgrade_legacy_args(args: Optional[Sequence[str]] = None) -> List[str]:
    import sys

    raw = list(sys.argv[1:] if args is None else args)
    if not raw:
        return raw
    known = {"extract", "extract-zip", "zip-selected", "-h", "--help"}
    if raw[0] not in known:
        return ["extract", *raw]
    return raw


def run_extract(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(archive)
    symbols = parse_symbols(args)
    date = infer_date(archive, args.date)
    out_root = Path(args.out_dir) if args.out_dir else SAVED_DATA_DIR / "data88_selected"
    backend, sevenzip = resolve_backend(args)

    returncodes = {}
    mode = "list" if args.list_only else "extract"
    for symbol in symbols:
        print(f"\n== {mode} {date} {symbol} via {backend} ==")
        if args.list_only:
            if backend == "py7zr":
                rc = list_symbol_py7zr(archive, date, symbol)
            else:
                assert sevenzip is not None
                rc = list_symbol(archive, sevenzip, date, symbol)
        else:
            if backend == "py7zr":
                rc = extract_symbol_py7zr(archive, date, symbol, out_root, args.overwrite)
            else:
                assert sevenzip is not None
                rc = extract_symbol(archive, sevenzip, date, symbol, out_root, args.overwrite)
        returncodes[symbol] = rc

    manifest = out_root / date / "_extract_manifest.json"
    write_manifest(manifest, archive, date, symbols, out_root, backend, sevenzip, mode, returncodes)
    failed = {symbol: rc for symbol, rc in returncodes.items() if rc != 0}
    if failed:
        raise SystemExit(f"failed symbols: {failed}")

    if args.zip_after and not args.list_only:
        zip_dir = Path(args.zip_dir) if args.zip_dir else out_root / date / "_zip"
        for symbol, rc in returncodes.items():
            if rc != 0:
                continue
            src_dir = out_root / date / symbol
            if src_dir.exists():
                zip_one_folder(src_dir, zip_dir / f"{date}_{symbol}.zip", args.overwrite)
                if args.delete_after_zip:
                    shutil.rmtree(src_dir)
                    print(f"deleted {src_dir}")


def zip_one_folder(src_dir: Path, zip_path: Path, overwrite: bool) -> None:
    if zip_path.exists() and not overwrite:
        print(f"skip existing zip: {zip_path}")
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w"
    with zipfile.ZipFile(zip_path, mode=mode, compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(src_dir.parent))
    print(f"wrote {zip_path}")


def run_zip_selected(args: argparse.Namespace) -> None:
    selected_dir = Path(args.selected_dir)
    date_dir = selected_dir / args.date
    if not date_dir.exists():
        fallback = Path(__file__).resolve().parent / args.selected_dir / args.date
        if fallback.exists():
            date_dir = fallback
            selected_dir = fallback.parent
        else:
            raise FileNotFoundError(date_dir)

    if args.symbols or args.symbols_file:
        symbols = parse_symbols(args)
    else:
        symbols = sorted(p.name for p in date_dir.iterdir() if p.is_dir() and "." in p.name)
    if not symbols:
        raise ValueError(f"no symbol folders found under {date_dir}")

    zip_dir = Path(args.zip_dir) if args.zip_dir else date_dir / "_zip"
    for symbol in symbols:
        src_dir = date_dir / symbol
        if not src_dir.exists():
            print(f"missing selected folder: {src_dir}")
            continue
        zip_one_folder(src_dir, zip_dir / f"{args.date}_{symbol}.zip", args.overwrite)


def run_extract_zip(args: argparse.Namespace) -> None:
    archive = Path(args.archive)
    if not archive.exists():
        raise FileNotFoundError(archive)
    symbols = parse_symbols(args)
    date = infer_date(archive, args.date)
    zip_dir = Path(args.zip_dir) if args.zip_dir else SAVED_DATA_DIR / "data88_selected" / date / "_zip"
    backend, sevenzip = resolve_backend(args)
    temp_parent = Path(args.temp_dir) if args.temp_dir else None

    returncodes = {}
    if backend == "py7zr" and not args.use_temp:
        for symbol in symbols:
            print(f"\n== extract-zip {date} {symbol} via py7zr memory ==")
            rc = zip_symbol_from_py7zr_memory(
                archive=archive,
                date=date,
                symbol=symbol,
                zip_path=zip_dir / f"{date}_{symbol}.zip",
                overwrite=args.overwrite,
                memory_limit_mb=args.memory_limit_mb,
            )
            returncodes[symbol] = rc
        manifest = zip_dir / f"{date}_extract_zip_manifest.json"
        write_manifest(manifest, archive, date, symbols, zip_dir, backend, sevenzip, "extract-zip-memory", returncodes)
        failed = {symbol: rc for symbol, rc in returncodes.items() if rc != 0}
        if failed:
            raise SystemExit(f"failed symbols: {failed}")
        return

    with tempfile.TemporaryDirectory(prefix=f"data88_{date}_", dir=temp_parent) as tmp:
        tmp_root = Path(tmp)
        print(f"temporary extraction root: {tmp_root}")
        for symbol in symbols:
            print(f"\n== extract-zip {date} {symbol} via {backend} ==")
            if backend == "py7zr":
                rc = extract_symbol_py7zr(archive, date, symbol, tmp_root, overwrite=True)
            else:
                assert sevenzip is not None
                rc = extract_symbol(archive, sevenzip, date, symbol, tmp_root, overwrite=True)
            returncodes[symbol] = rc
            if rc != 0:
                continue
            src_dir = tmp_root / date / symbol
            if not src_dir.exists():
                print(f"missing extracted folder: {src_dir}")
                returncodes[symbol] = 3
                continue
            zip_one_folder(src_dir, zip_dir / f"{date}_{symbol}.zip", args.overwrite)

    manifest = zip_dir / f"{date}_extract_zip_manifest.json"
    write_manifest(manifest, archive, date, symbols, zip_dir, backend, sevenzip, "extract-zip", returncodes)
    failed = {symbol: rc for symbol, rc in returncodes.items() if rc != 0}
    if failed:
        raise SystemExit(f"failed symbols: {failed}")


def main() -> None:
    args = parse_args_with_legacy()
    if args.cmd == "extract":
        run_extract(args)
    elif args.cmd == "extract-zip":
        run_extract_zip(args)
    elif args.cmd == "zip-selected":
        run_zip_selected(args)
    else:
        raise ValueError(args.cmd)


def parse_args_with_legacy() -> argparse.Namespace:
    parser_args = maybe_upgrade_legacy_args()
    return parse_args_from(parser_args)


def parse_args_from(argv: Sequence[str]) -> argparse.Namespace:
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return parse_args()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
