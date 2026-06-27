#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch AS1455 live common helpers and pipeline wiring."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER_COMMON = "# BEGIN AS1455_LIVE_FEATURE_ALIGNMENT_V2"

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

COMMON_HELPERS = r'''
# BEGIN AS1455_LIVE_FEATURE_ALIGNMENT_V2

def clean_code_value(value) -> str:
    """Return a zero-padded 6-digit A-share code from symbol/code-like values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    if "." in text:
        try:
            return symbol_code(text)
        except Exception:
            pass
    digits = re.sub(r"\D", "", text)
    if not digits:
        return ""
    return digits[:6].zfill(6)


def text_is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a", "unknown"}


def normalize_universe_meta(universe_meta: pd.DataFrame) -> pd.DataFrame:
    """Normalize symbol/code/industry/sector metadata for live/training feature parity."""
    meta = universe_meta.copy()
    if "symbol" not in meta.columns:
        if "code" in meta.columns:
            meta["symbol"] = meta["code"].map(normalize_symbol)
        else:
            raise ValueError("universe_meta must contain symbol or code")
    meta["symbol"] = meta["symbol"].map(normalize_symbol)
    meta["code"] = meta["symbol"].map(symbol_code)

    if "industry" not in meta.columns:
        meta["industry"] = "unknown"
    meta["industry"] = meta["industry"].where(~meta["industry"].map(text_is_missing), "unknown").astype(str)

    if "sector" in meta.columns:
        sector_numeric = pd.to_numeric(meta["sector"], errors="coerce")
    else:
        sector_numeric = pd.Series(np.nan, index=meta.index)

    if sector_numeric.notna().any():
        meta["sector"] = sector_numeric
        missing = meta["sector"].isna()
        if missing.any():
            ind = meta.loc[missing, "industry"].fillna("unknown").astype(str)
            meta.loc[missing, "sector"] = pd.factorize(ind)[0]
    else:
        meta["sector"] = pd.factorize(meta["industry"].fillna("unknown").astype(str))[0]

    meta["sector"] = pd.to_numeric(meta["sector"], errors="coerce").fillna(-1).astype(int)
    return meta.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


def load_sector_reference_from_model_data(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Load symbol->sector mapping from the training model_data HDF."""
    if path is None or str(path).strip() == "":
        return pd.DataFrame(columns=["symbol", "code", "sector"]), {"path": "", "loaded": False, "reason": "empty_path"}

    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["symbol", "code", "sector"]), {"path": str(p), "loaded": False, "reason": "missing_file"}

    reports = []
    try:
        with pd.HDFStore(p, mode="r") as store:
            for key in store.keys():
                try:
                    df = store[key]
                except Exception as exc:
                    reports.append({"key": key, "loaded": False, "reason": f"read_failed:{type(exc).__name__}:{exc}"})
                    continue

                if not isinstance(df, pd.DataFrame) or "sector" not in df.columns:
                    reports.append({"key": key, "loaded": False, "reason": "no_sector_column"})
                    continue

                sector = pd.to_numeric(df["sector"], errors="coerce")
                if sector.notna().sum() == 0:
                    reports.append({"key": key, "loaded": False, "reason": "all_sector_nan"})
                    continue

                symbol_values = None
                if "symbol" in df.columns:
                    symbol_values = df["symbol"]
                elif "code" in df.columns:
                    symbol_values = df["code"]
                elif isinstance(df.index, pd.MultiIndex):
                    names = list(df.index.names)
                    if "symbol" in names:
                        symbol_values = df.index.get_level_values("symbol")
                    elif "code" in names:
                        symbol_values = df.index.get_level_values("code")
                    else:
                        symbol_values = df.index.get_level_values(-1)
                else:
                    if df.index.name in {"symbol", "code"}:
                        symbol_values = df.index

                if symbol_values is None:
                    reports.append({"key": key, "loaded": False, "reason": "no_symbol_or_code"})
                    continue

                # Flatten both series before constructing tmp.  In the training HDF,
                # symbol may exist both as an index level and as a column; keeping the
                # MultiIndex here makes groupby("symbol") ambiguous.  Resetting to a
                # plain RangeIndex avoids the pandas "both an index level and a column"
                # failure while preserving row order for tail(1).
                symbol_flat = pd.Series(list(symbol_values), name="symbol").reset_index(drop=True)
                sector_flat = pd.Series(sector.to_numpy(), name="sector").reset_index(drop=True)
                tmp = pd.DataFrame({"symbol": symbol_flat, "sector": sector_flat}).reset_index(drop=True)
                tmp = tmp.dropna(subset=["symbol", "sector"]).copy()
                if tmp.empty:
                    reports.append({"key": key, "loaded": False, "reason": "empty_after_dropna"})
                    continue
                tmp["symbol"] = tmp["symbol"].map(normalize_symbol)
                tmp["code"] = tmp["symbol"].map(symbol_code)
                tmp["sector"] = pd.to_numeric(tmp["sector"], errors="coerce")
                tmp = tmp.dropna(subset=["sector"]).reset_index(drop=True).copy()
                tmp["sector"] = tmp["sector"].astype(int)

                ref = tmp.groupby("symbol", as_index=False, sort=False).tail(1)[["symbol", "code", "sector"]]
                ref = ref.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
                if len(ref):
                    return ref, {
                        "path": str(p),
                        "loaded": True,
                        "key": key,
                        "rows": int(len(ref)),
                        "unique_sector": int(ref["sector"].nunique(dropna=True)),
                        "attempts": reports,
                    }
                reports.append({"key": key, "loaded": False, "reason": "empty_ref"})
    except Exception as exc:
        return pd.DataFrame(columns=["symbol", "code", "sector"]), {
            "path": str(p),
            "loaded": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "attempts": reports,
        }

    return pd.DataFrame(columns=["symbol", "code", "sector"]), {
        "path": str(p),
        "loaded": False,
        "reason": "no_usable_key",
        "attempts": reports,
    }


def enrich_universe_meta_with_sector_reference(universe_meta: pd.DataFrame, sector_ref: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Inject training-time sector values into live universe metadata."""
    meta = normalize_universe_meta(universe_meta)
    report = {"sector_reference_rows": 0, "matched_symbols": 0, "unmatched_symbols": int(len(meta)), "used_training_sector": False}

    if sector_ref is None or sector_ref.empty:
        return meta, report

    ref = sector_ref.copy()
    if "symbol" not in ref.columns:
        if "code" in ref.columns:
            ref["symbol"] = ref["code"].map(normalize_symbol)
        else:
            return meta, report
    ref["symbol"] = ref["symbol"].map(normalize_symbol)
    ref["code"] = ref["symbol"].map(symbol_code)
    ref["sector"] = pd.to_numeric(ref["sector"], errors="coerce")
    ref = ref.dropna(subset=["sector"]).drop_duplicates("symbol")
    ref["sector"] = ref["sector"].astype(int)

    merged = meta.drop(columns=["sector"], errors="ignore").merge(
        ref[["symbol", "sector"]].rename(columns={"sector": "sector_from_training"}),
        on="symbol",
        how="left",
    )
    matched = int(merged["sector_from_training"].notna().sum())
    merged["sector"] = merged["sector_from_training"]
    merged.drop(columns=["sector_from_training"], inplace=True)
    merged["sector"] = pd.to_numeric(merged["sector"], errors="coerce").fillna(-1).astype(int)

    report = {
        "sector_reference_rows": int(len(ref)),
        "matched_symbols": matched,
        "unmatched_symbols": int(len(merged) - matched),
        "used_training_sector": True,
        "unique_sector": int(merged.loc[merged["sector"] >= 0, "sector"].nunique(dropna=True)),
    }
    return merged, report

# END AS1455_LIVE_FEATURE_ALIGNMENT_V2
'''

def patch_common(path: Path) -> None:
    s = read(path)

    # Idempotent repair: the first v2 package inserted a malformed helper block.
    # Always remove any existing BEGIN/END block and reinsert the corrected one.
    block_re = re.compile(
        r"\n?# BEGIN AS1455_LIVE_FEATURE_ALIGNMENT_V2.*?# END AS1455_LIVE_FEATURE_ALIGNMENT_V2\n?",
        re.DOTALL,
    )
    s = block_re.sub("\n", s)

    anchor = "\ndef compute_ch12_features("
    if anchor not in s:
        raise SystemExit(f"{path}: cannot find compute_ch12_features anchor")
    s = s.replace(anchor, "\n" + COMMON_HELPERS + anchor, 1)

    old_meta = """    meta = universe_meta.copy()
    if "code" not in meta.columns:
        meta["code"] = meta["symbol"].map(lambda s: normalize_symbol(s)[:6])
    if "industry" not in meta.columns:
        meta["industry"] = "unknown"
    meta["sector"] = pd.factorize(meta["industry"].fillna("unknown"))[0].astype(int)
    sector_map = meta.set_index("code")["sector"]
"""
    new_meta = """    meta = normalize_universe_meta(universe_meta)
    sector_map = meta.set_index("code")["sector"]
"""
    if old_meta in s:
        s = s.replace(old_meta, new_meta, 1)
    elif new_meta not in s:
        raise SystemExit(f"{path}: cannot replace universe meta block")

    old_rank = """    prices["dollar_vol_rank"] = dollar_vol_ma.rank(axis=1, ascending=False).stack().swaplevel()
"""
    new_rank = """    # Keep MultiIndex order as (date, symbol).  The old swaplevel() changed it to
    # (symbol, date), so assignment aligned nothing and live dollar_vol_rank became all NaN.
    prices["dollar_vol_rank"] = dollar_vol_ma.rank(axis=1, ascending=False).stack().reindex(prices.index)
"""
    if old_rank in s:
        s = s.replace(old_rank, new_rank, 1)
    elif 'prices["dollar_vol_rank"] = dollar_vol_ma.rank(axis=1, ascending=False).stack().reindex(prices.index)' not in s:
        raise SystemExit(f"{path}: cannot replace dollar_vol_rank line")

    old_sector = """    prices["sector"] = prices.index.get_level_values("symbol").astype(str).str.slice(0, 6).map(sector_map).fillna(-1).astype(int)
"""
    new_sector = """    price_codes = pd.Series(prices.index.get_level_values("symbol").map(symbol_code), index=prices.index)
    prices["sector"] = price_codes.map(sector_map).fillna(-1).astype(int)
"""
    if old_sector in s:
        s = s.replace(old_sector, new_sector, 1)
    elif 'price_codes = pd.Series(prices.index.get_level_values("symbol").map(symbol_code), index=prices.index)' not in s:
        raise SystemExit(f"{path}: cannot replace sector mapping line")

    write(path, s)

def patch_pipeline(path: Path) -> None:
    s = read(path)
    if "SECTOR_REFERENCE=" not in s:
        s = s.replace('FEATURE_COLUMNS="${FEATURE_COLUMNS:-}"\n',
                      'FEATURE_COLUMNS="${FEATURE_COLUMNS:-}"\nSECTOR_REFERENCE="${SECTOR_REFERENCE:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"\nALLOW_SECTOR_FALLBACK="${ALLOW_SECTOR_FALLBACK:-0}"\nVALIDATE_LIVE_FEATURES="${VALIDATE_LIVE_FEATURES:-1}"\n',
                      1)
    if "  SECTOR_REFERENCE=${SECTOR_REFERENCE}" not in s:
        s = s.replace('  FEATURE_COLUMNS=${FEATURE_COLUMNS:-<none>}\n',
                      '  FEATURE_COLUMNS=${FEATURE_COLUMNS:-<none>}\n  SECTOR_REFERENCE=${SECTOR_REFERENCE}\n  ALLOW_SECTOR_FALLBACK=${ALLOW_SECTOR_FALLBACK}\n  VALIDATE_LIVE_FEATURES=${VALIDATE_LIVE_FEATURES}\n',
                      1)
    if '--sector-reference "${SECTOR_REFERENCE}"' not in s:
        s = s.replace('    --min-feature-rows "${MIN_FEATURE_ROWS}"\n',
                      '    --min-feature-rows "${MIN_FEATURE_ROWS}"\n    --sector-reference "${SECTOR_REFERENCE}"\n',
                      1)
    if 'ALLOW_SECTOR_FALLBACK' in s and '--allow-sector-fallback' not in s:
        s = s.replace('  if [[ -n "${FEATURE_COLUMNS}" ]]; then\n',
                      '  if [[ "${ALLOW_SECTOR_FALLBACK}" == "1" ]]; then\n    args+=(--allow-sector-fallback)\n  fi\n\n  if [[ -n "${FEATURE_COLUMNS}" ]]; then\n',
                      1)
    if "validate_as1455_live_model_features_v2.py" not in s:
        s = s.replace('  json_bool_check "${LIVE_DIR}/12_feature_build_report.json" "feature_passed"\n',
                      '  json_bool_check "${LIVE_DIR}/12_feature_build_report.json" "feature_passed"\n  if [[ "${VALIDATE_LIVE_FEATURES}" == "1" ]]; then\n    "${PYTHON}" tools/validate_as1455_live_model_features_v2.py \\\n      --live-dir "${LIVE_DIR}" \\\n      --trade-date "${TRADE_DATE}" \\\n      --sector-reference "${SECTOR_REFERENCE}" \\\n      --min-feature-rows "${MIN_FEATURE_ROWS}"\n  fi\n',
                      1)
    write(path, s)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = Path(args.repo)
    patch_common(repo / "features" / "as1455_live_common.py")
    patch_pipeline(repo / "scripts" / "run_as1455_live_data_feature_pipeline.sh")
    print("[OK] patched features/as1455_live_common.py")
    print("[OK] patched scripts/run_as1455_live_data_feature_pipeline.sh")

if __name__ == "__main__":
    main()
