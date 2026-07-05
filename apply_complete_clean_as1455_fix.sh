#!/usr/bin/env bash
set -Eeuo pipefail

# Clean AS1455 adjusted-data fix.
# Run this from the repo root: ~/stock_realtime_v021_full
#
# This patch:
#   1) removes known wrong/one-off raw rebuild entrypoints;
#   2) creates a single model_data contract validator;
#   3) creates a clean adjusted rebuild script;
#   4) creates a clean one-key rebuild + weekly retrain/backtest script;
#   5) creates a bad-artifact cleanup script;
#   6) injects hard validation gates into weekly retrain and live feature pipeline.
#
# It intentionally does NOT change the ML4T horizon logic:
# pct_change/shift remain per-symbol observation horizon, matching the original book code.

ROOT="$(pwd)"

fail() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

[[ -f "scripts/build_ashare_ch12_as1455_model_data.py" ]] || fail "run from repo root; missing scripts/build_ashare_ch12_as1455_model_data.py"
[[ -f "scripts/build_ashare_ch12_as1455_lowmem.sh" ]] || fail "missing scripts/build_ashare_ch12_as1455_lowmem.sh"
[[ -f "scripts/run_as1455_top5_weekly_retrain_full_v7.sh" ]] || fail "missing scripts/run_as1455_top5_weekly_retrain_full_v7.sh"
[[ -f "scripts/run_as1455_live_data_feature_pipeline.sh" ]] || fail "missing scripts/run_as1455_live_data_feature_pipeline.sh"

mkdir -p tools scripts

remove_path() {
  local p="$1"
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git ls-files --error-unmatch "$p" >/dev/null 2>&1; then
    info "git rm ${p}"
    git rm -f "$p"
  else
    if [[ -e "$p" ]]; then
      info "rm ${p}"
      rm -rf "$p"
    else
      info "already absent: ${p}"
    fi
  fi
}

info "removing wrong/one-off raw rebuild entrypoints"
remove_path "scripts/run_as1455_extend_weekly_empty_v1.sh"
remove_path "scripts/repair_as1455_20260626_from_raw5m_v4.sh"

info "writing tools/validate_as1455_model_data_contract.py"
cat > tools/validate_as1455_model_data_contract.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate AS1455 model_data contract.

This is the hard gate that prevents raw-price model_data from entering
weekly retrain/backtest or live feature alignment.

The intended clean contract is:
  raw 5m cache / raw daily close+preclose
  -> manual front-adjustment factor from raw daily preclose / previous raw close
  -> adj_open/high/low/close_as1455
  -> ML4T-style per-symbol observation horizon features and forward labels
  -> model_data_as1455.h5

The horizon intentionally follows the original ML4T implementation:
per-symbol effective observation sequence, not a forced exchange-calendar reindex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "dollar_vol",
    "dollar_vol_rank",
    "rsi",
    "bb_high",
    "bb_low",
    "NATR",
    "ATR",
    "PPO",
    "MACD",
    "sector",
    "r01",
    "r05",
    "r10",
    "r21",
    "r42",
    "r63",
    "r01dec",
    "r05dec",
    "r10dec",
    "r21dec",
    "r42dec",
    "r63dec",
    "r01q_sector",
    "r05q_sector",
    "r10q_sector",
    "r21q_sector",
    "r42q_sector",
    "r63q_sector",
    "r01_fwd",
    "r05_fwd",
    "r21_fwd",
    "year",
    "month",
    "weekday",
]
OUTCOMES = ["r01_fwd", "r05_fwd", "r21_fwd"]
RET_WINDOWS = [1, 5, 10, 21, 42, 63]
FWD_WINDOWS = [1, 5, 21]

CONTRACT_NAME = "model_data_contract.json"
AUDIT_NAME = "model_data_contract_validation.json"
PRICE_BASIS = "manual_qfq_from_raw_daily_preclose"
BUILDER = "scripts/build_ashare_ch12_as1455_model_data.py"
HORIZON = "ml4t_per_symbol_observation_horizon"


def normalize_symbol(value: Any) -> str:
    s = str(value).strip().upper()
    if "." in s:
        a, b = s.split(".", 1)
        if a.isalpha():
            code = "".join(ch for ch in b if ch.isdigit())[:6].zfill(6)
            market = a
        else:
            code = "".join(ch for ch in a if ch.isdigit())[:6].zfill(6)
            market = b
        if market in {"XSHE", "SZSE"}:
            market = "SZ"
        if market in {"XSHG", "SSE"}:
            market = "SH"
        if market not in {"SH", "SZ"}:
            market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code}.{market}"
    digits = "".join(ch for ch in s if ch.isdigit())[:6].zfill(6)
    market = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return f"{digits}.{market}"


def code6(symbol: str) -> str:
    return normalize_symbol(symbol).split(".", 1)[0]


def is_mainboard_code(code: str) -> bool:
    return str(code).startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def load_hdf(path: Path, key: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty HDF: {path}")
    try:
        return pd.read_hdf(path, key)
    except Exception as exc:
        raise SystemExit(f"failed to read {path}:{key}: {type(exc).__name__}: {exc}") from exc


def align_symbol_date_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.MultiIndex):
        raise SystemExit(f"{name} must have MultiIndex")
    names = list(out.index.names)
    if names == ["date", "symbol"]:
        out = out.swaplevel("date", "symbol")
    elif names != ["symbol", "date"]:
        raise SystemExit(f"{name} index names must be ['symbol','date'] or ['date','symbol'], got {names}")
    symbols = [normalize_symbol(x) for x in out.index.get_level_values("symbol")]
    dates = pd.to_datetime(out.index.get_level_values("date"), errors="coerce").normalize()
    if pd.isna(dates).any():
        raise SystemExit(f"{name} contains invalid dates in index")
    out.index = pd.MultiIndex.from_arrays([symbols, dates], names=["symbol", "date"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def max_abs_diff(a: pd.Series, b: pd.Series) -> tuple[int, int, float | None]:
    both = ~(a.isna() | b.isna())
    if not bool(both.any()):
        return 0, 0, None
    d = (a.loc[both].astype(float) - b.loc[both].astype(float)).abs()
    return int(both.sum()), int((d > 1e-8).sum()), float(d.max())


def read_contract(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing contract: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"failed to read contract {path}: {type(exc).__name__}: {exc}") from exc
    return obj


def validate_contract_object(obj: dict[str, Any], contract_path: Path) -> None:
    required = {
        "builder": BUILDER,
        "price_basis": PRICE_BASIS,
        "adjust_factor_mode": "raw_preclose",
        "feature_horizon": HORIZON,
        "model_data_key": "model_data",
    }
    for key, expected in required.items():
        got = obj.get(key)
        if got != expected:
            raise SystemExit(f"{contract_path}: bad {key}: got {got!r}, expected {expected!r}")

    outcomes = obj.get("outcomes")
    if outcomes != OUTCOMES:
        raise SystemExit(f"{contract_path}: bad outcomes: got {outcomes!r}, expected {OUTCOMES!r}")

    if int(obj.get("schema_columns", -1)) != 34:
        raise SystemExit(f"{contract_path}: bad schema_columns: {obj.get('schema_columns')!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate AS1455 adjusted model_data contract")
    ap.add_argument("--model-data", default="saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5")
    ap.add_argument("--require-contract", action="store_true")
    ap.add_argument("--write-contract", action="store_true")
    ap.add_argument("--require-adjusted-artifacts", action="store_true")
    ap.add_argument("--max-mainboard-abs-r01-fail", type=float, default=0.50,
                    help="Hard fail if any mainboard abs(r01) exceeds this; adjusted data should not show split-like jumps")
    ap.add_argument("--warn-mainboard-abs-r01", type=float, default=0.20,
                    help="Report count of mainboard abs(r01) above this warning threshold")
    args = ap.parse_args()

    model_path = Path(args.model_data)
    root = model_path.parent
    contract_path = root / CONTRACT_NAME
    audit_path = root / AUDIT_NAME
    adj_path = root / "as1455_ohlcv_adj.h5"

    if args.require_contract:
        validate_contract_object(read_contract(contract_path), contract_path)

    if args.require_adjusted_artifacts and (not adj_path.exists() or adj_path.stat().st_size == 0):
        raise SystemExit(f"missing adjusted artifact required by clean contract: {adj_path}")

    model = align_symbol_date_index(load_hdf(model_path, "model_data"), "model_data")

    problems: list[str] = []
    if list(model.columns) != EXPECTED_COLUMNS:
        problems.append("model_data columns do not exactly match expected 34-column schema")
    if model.shape[1] != 34:
        problems.append(f"model_data must have 34 columns, got {model.shape[1]}")
    outcomes = model.filter(like="fwd").columns.tolist()
    if outcomes != OUTCOMES:
        problems.append(f"bad outcomes: {outcomes}")

    dates = pd.to_datetime(model.index.get_level_values("date"))
    symbols = pd.Index(model.index.get_level_values("symbol"))
    sectors = pd.to_numeric(model["sector"], errors="coerce") if "sector" in model else pd.Series(dtype=float)
    sector_nunique = int(sectors.nunique(dropna=True)) if len(sectors) else 0
    if sector_nunique <= 1:
        problems.append(f"sector degenerated: sector_nunique={sector_nunique}")

    codes = pd.Series([code6(x) for x in symbols], index=model.index)
    main_mask = codes.map(is_mainboard_code).astype(bool)
    r01 = pd.to_numeric(model["r01"], errors="coerce")
    main_abs = r01.loc[main_mask].abs().dropna()
    warn_main_rows = int(main_abs.gt(args.warn_mainboard_abs_r01).sum())
    fail_main_rows = int(main_abs.gt(args.max_mainboard_abs_r01_fail).sum())
    if fail_main_rows:
        worst = model.loc[main_abs.nlargest(min(20, fail_main_rows)).index, ["r01"]].reset_index()
        problems.append(
            f"mainboard abs(r01)>{args.max_mainboard_abs_r01_fail:g} rows={fail_main_rows}; "
            f"worst={worst.to_dict(orient='records')}"
        )

    recalc_summary: dict[str, Any] = {}
    if adj_path.exists() and adj_path.stat().st_size > 0:
        adj = align_symbol_date_index(load_hdf(adj_path, "ohlcv"), "as1455_ohlcv_adj")
        if "adj_close_as1455" not in adj.columns:
            problems.append(f"{adj_path} missing adj_close_as1455")
        else:
            common = model.index.intersection(adj.index)
            if len(common) == 0:
                problems.append("model_data and as1455_ohlcv_adj.h5 have no common index")
            close = pd.to_numeric(adj.loc[common, "adj_close_as1455"], errors="coerce").sort_index()
            m_common = model.loc[common].sort_index()
            by_symbol_close = close.groupby(level="symbol")
            for t in RET_WINDOWS:
                col = f"r{t:02}"
                recalc = by_symbol_close.pct_change(t).reindex(m_common.index)
                compared, bad, maxdiff = max_abs_diff(pd.to_numeric(m_common[col], errors="coerce"), recalc)
                recalc_summary[col] = {"compared": compared, "diff_gt_1e-8": bad, "max_abs_diff": maxdiff}
                if bad:
                    problems.append(f"{col} differs from adj_close_as1455 recomputation: bad={bad}, maxdiff={maxdiff}")
            for t in FWD_WINDOWS:
                col = f"r{t:02}_fwd"
                base = pd.to_numeric(m_common[f"r{t:02}"], errors="coerce")
                recalc = base.groupby(level="symbol").shift(-t).reindex(m_common.index)
                compared, bad, maxdiff = max_abs_diff(pd.to_numeric(m_common[col], errors="coerce"), recalc)
                recalc_summary[col] = {"compared": compared, "diff_gt_1e-8": bad, "max_abs_diff": maxdiff}
                if bad:
                    problems.append(f"{col} differs from ML4T shift recomputation: bad={bad}, maxdiff={maxdiff}")
    elif args.require_adjusted_artifacts:
        problems.append(f"missing adjusted artifact: {adj_path}")

    contract = {
        "builder": BUILDER,
        "price_basis": PRICE_BASIS,
        "adjust_factor_mode": "raw_preclose",
        "feature_horizon": HORIZON,
        "model_data_key": "model_data",
        "schema_columns": 34,
        "outcomes": OUTCOMES,
        "model_data_path": str(model_path),
        "adjusted_ohlcv_path": str(adj_path),
        "rows": int(len(model)),
        "symbols": int(symbols.nunique()),
        "date_min": dates.min().strftime("%Y-%m-%d") if len(dates) else "",
        "date_max": dates.max().strftime("%Y-%m-%d") if len(dates) else "",
        "sector_nunique": sector_nunique,
    }

    audit = dict(contract)
    audit.update(
        {
            "contract_path": str(contract_path),
            "warn_mainboard_abs_r01_threshold": args.warn_mainboard_abs_r01,
            "warn_mainboard_abs_r01_rows": warn_main_rows,
            "fail_mainboard_abs_r01_threshold": args.max_mainboard_abs_r01_fail,
            "fail_mainboard_abs_r01_rows": fail_main_rows,
            "recalc_summary": recalc_summary,
            "passed": not problems,
            "problems": problems,
        }
    )

    if args.write_contract:
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    if problems:
        raise SystemExit("AS1455 model_data contract validation failed; see " + str(audit_path))


if __name__ == "__main__":
    main()
PY
chmod +x tools/validate_as1455_model_data_contract.py

info "writing scripts/clean_bad_as1455_artifacts.sh"
cat > scripts/clean_bad_as1455_artifacts.sh <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "[INFO] This deletes known bad DERIVED AS1455 artifacts only."
echo "[INFO] It keeps raw/intermediate cache dirs:"
echo "       saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache"
echo "       saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache"
echo "       saved_data/ashare_ml4t/ch12_as1455/as1455_daily_cache"

CH12="saved_data/ashare_ml4t/ch12_as1455"

rm -f \
  "${CH12}/model_data_as1455.h5" \
  "${CH12}/model_data_as1455.h5.bak_before_extend_"* \
  "${CH12}/as1455_ohlcv_raw.h5" \
  "${CH12}/as1455_ohlcv_adj.h5" \
  "${CH12}/as1455_execution_metadata.h5" \
  "${CH12}/model_data_contract.json" \
  "${CH12}/model_data_contract_validation.json" 2>/dev/null || true

rm -rf "${CH12}/reports" 2>/dev/null || true

# Known bad weekly retrain/backtest outputs from raw/sectorfix extension attempts.
rm -rf \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_2026-06-26_after_hole_repair* \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_2026-06-26_sectorfix* \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_20260626_after_hole_repair* \
  saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_20260516_to_20260626_sectorfix* 2>/dev/null || true

echo "[DONE] bad derived artifacts removed"
BASH
chmod +x scripts/clean_bad_as1455_artifacts.sh

info "writing scripts/rebuild_as1455_model_data_clean_adj.sh"
cat > scripts/rebuild_as1455_model_data_clean_adj.sh <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"
START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-2026-06-26}"

OUT_DIR="${OUT_DIR:-saved_data/ashare_ml4t/ch12_as1455}"
UNIVERSE="${UNIVERSE:-saved_data/ashare_static_universe/07_universe_allA_top1000_static.csv}"
BAR_CACHE_DIR="${BAR_CACHE_DIR:-${OUT_DIR}/baostock_5m_cache}"
AS1455_DAILY_CACHE_DIR="${AS1455_DAILY_CACHE_DIR:-${OUT_DIR}/as1455_daily_cache}"
RAW_DAILY_CACHE_DIR="${RAW_DAILY_CACHE_DIR:-${OUT_DIR}/baostock_raw_daily_cache}"
QFQ_DAILY_CACHE_DIR="${QFQ_DAILY_CACHE_DIR:-saved_data/ashare_ml4t/ch12_reproduce/baostock_qfq_daily_cache}"

# Default is strict offline rebuild. Set FETCH_MISSING_RAW_DAILY=1 only if you intentionally want BaoStock network fetch.
FETCH_MISSING_BAOSTOCK="${FETCH_MISSING_BAOSTOCK:-0}"
FETCH_MISSING_RAW_DAILY="${FETCH_MISSING_RAW_DAILY:-0}"
FETCH_MISSING_QFQ_DAILY="${FETCH_MISSING_QFQ_DAILY:-0}"
QFQ5M_AUDIT_SAMPLES="${QFQ5M_AUDIT_SAMPLES:-0}"

fetch_baostock_arg="--no-fetch-missing-baostock"
fetch_raw_daily_arg="--no-fetch-missing-raw-daily"
fetch_qfq_daily_arg="--no-fetch-missing-qfq-daily"
[[ "${FETCH_MISSING_BAOSTOCK}" == "1" ]] && fetch_baostock_arg="--fetch-missing-baostock"
[[ "${FETCH_MISSING_RAW_DAILY}" == "1" ]] && fetch_raw_daily_arg="--fetch-missing-raw-daily"
[[ "${FETCH_MISSING_QFQ_DAILY}" == "1" ]] && fetch_qfq_daily_arg="--fetch-missing-qfq-daily"

echo "[CONFIG]"
echo "  OUT_DIR=${OUT_DIR}"
echo "  UNIVERSE=${UNIVERSE}"
echo "  BAR_CACHE_DIR=${BAR_CACHE_DIR}"
echo "  AS1455_DAILY_CACHE_DIR=${AS1455_DAILY_CACHE_DIR}"
echo "  RAW_DAILY_CACHE_DIR=${RAW_DAILY_CACHE_DIR}"
echo "  QFQ_DAILY_CACHE_DIR=${QFQ_DAILY_CACHE_DIR}"
echo "  START_DATE=${START_DATE}"
echo "  END_DATE=${END_DATE}"
echo "  FETCH_MISSING_BAOSTOCK=${FETCH_MISSING_BAOSTOCK}"
echo "  FETCH_MISSING_RAW_DAILY=${FETCH_MISSING_RAW_DAILY}"
echo "  FETCH_MISSING_QFQ_DAILY=${FETCH_MISSING_QFQ_DAILY}"

[[ -d "${BAR_CACHE_DIR}" ]] || { echo "[ERROR] missing BAR_CACHE_DIR=${BAR_CACHE_DIR}" >&2; exit 1; }
[[ -d "${RAW_DAILY_CACHE_DIR}" ]] || { echo "[ERROR] missing RAW_DAILY_CACHE_DIR=${RAW_DAILY_CACHE_DIR}" >&2; exit 1; }
[[ -f "${UNIVERSE}" ]] || { echo "[ERROR] missing UNIVERSE=${UNIVERSE}" >&2; exit 1; }

mkdir -p "${OUT_DIR}" "${AS1455_DAILY_CACHE_DIR}"

"${PYTHON}" scripts/build_ashare_ch12_as1455_model_data.py \
  --out-dir "${OUT_DIR}" \
  --universe "${UNIVERSE}" \
  --bar-root "${BAR_CACHE_DIR}" \
  --bar-glob "*_5m_raw.csv" \
  --baostock-5m-cache-dir "${BAR_CACHE_DIR}" \
  --as1455-daily-cache-dir "${AS1455_DAILY_CACHE_DIR}" \
  --raw-daily-cache-dir "${RAW_DAILY_CACHE_DIR}" \
  --qfq-daily-cache-dir "${QFQ_DAILY_CACHE_DIR}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --rebuild-as1455-daily-cache \
  --adjust-factor-mode raw_preclose \
  --qfq5m-audit-samples "${QFQ5M_AUDIT_SAMPLES}" \
  "${fetch_baostock_arg}" \
  "${fetch_raw_daily_arg}" \
  "${fetch_qfq_daily_arg}" \
  --profile-memory

"${PYTHON}" tools/validate_as1455_model_data_contract.py \
  --model-data "${OUT_DIR}/model_data_as1455.h5" \
  --write-contract \
  --require-adjusted-artifacts

echo "[DONE] clean adjusted AS1455 model_data rebuilt: ${OUT_DIR}/model_data_as1455.h5"
echo "[DONE] contract: ${OUT_DIR}/model_data_contract.json"
BASH
chmod +x scripts/rebuild_as1455_model_data_clean_adj.sh

info "writing scripts/run_as1455_clean_rebuild_weekly_retrain.sh"
cat > scripts/run_as1455_clean_rebuild_weekly_retrain.sh <<'BASH'
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-python3}"

# Rebuild range for the training HDF.
REBUILD_START_DATE="${REBUILD_START_DATE:-2020-01-01}"
REBUILD_END_DATE="${REBUILD_END_DATE:-2026-06-26}"

# Weekly retrain/backtest range.
START_DATE="${START_DATE:-2026-05-16}"
END_DATE="${END_DATE:-${REBUILD_END_DATE}}"

MODEL_DATA="${MODEL_DATA:-saved_data/ashare_ml4t/ch12_as1455/model_data_as1455.h5}"
RAW_DAILY="${RAW_DAILY:-saved_data/ashare_ml4t/ch12_as1455/baostock_raw_daily_cache}"
RAW_5M="${RAW_5M:-saved_data/ashare_ml4t/ch12_as1455/baostock_5m_cache}"
OUT_ROOT="${OUT_ROOT:-saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_${START_DATE}_to_${END_DATE}_clean_adj}"
FORCE="${FORCE:-1}"

echo "[1/3] clean adjusted model_data rebuild"
START_DATE="${REBUILD_START_DATE}" \
END_DATE="${REBUILD_END_DATE}" \
bash scripts/rebuild_as1455_model_data_clean_adj.sh

echo "[2/3] validate rebuilt model_data contract"
"${PYTHON}" tools/validate_as1455_model_data_contract.py \
  --model-data "${MODEL_DATA}" \
  --require-contract \
  --require-adjusted-artifacts

echo "[3/3] weekly retrain/backtest"
MODEL_DATA="${MODEL_DATA}" \
RAW_DAILY="${RAW_DAILY}" \
RAW_5M="${RAW_5M}" \
START_DATE="${START_DATE}" \
END_DATE="${END_DATE}" \
OUT_ROOT="${OUT_ROOT}" \
FORCE="${FORCE}" \
bash scripts/run_as1455_top5_weekly_retrain_full_v7.sh

echo "[DONE] OUT_ROOT=${OUT_ROOT}"
BASH
chmod +x scripts/run_as1455_clean_rebuild_weekly_retrain.sh

info "patching scripts/run_as1455_top5_weekly_retrain_full_v7.sh with contract gate"
python3 - <<'PY'
from pathlib import Path

p = Path("scripts/run_as1455_top5_weekly_retrain_full_v7.sh")
text = p.read_text(encoding="utf-8")
marker = "# BEGIN AS1455_ADJUSTED_CONTRACT_GATE"
if marker not in text:
    needle = 'mkdir -p "$OUT_ROOT" "$BT_ROOT"\n'
    insert = r"""
# BEGIN AS1455_ADJUSTED_CONTRACT_GATE
CONTRACT_VALIDATOR="${CONTRACT_VALIDATOR:-tools/validate_as1455_model_data_contract.py}"
if [[ "${SKIP_MODEL_DATA_CONTRACT_CHECK:-0}" != "1" ]]; then
  echo "[INFO] validating adjusted AS1455 model_data contract: ${MODEL_DATA}"
  python3 "${CONTRACT_VALIDATOR}" \
    --model-data "${MODEL_DATA}" \
    --require-contract \
    --require-adjusted-artifacts
else
  echo "[WARN] SKIP_MODEL_DATA_CONTRACT_CHECK=1; adjusted model_data contract validation skipped"
fi
# END AS1455_ADJUSTED_CONTRACT_GATE
"""
    if needle not in text:
        raise SystemExit(f"cannot patch {p}: needle not found")
    text = text.replace(needle, needle + insert + "\n", 1)
    p.write_text(text, encoding="utf-8")
    print(f"[OK] patched {p}")
else:
    print(f"[OK] already patched {p}")
PY

info "patching scripts/run_as1455_live_data_feature_pipeline.sh with sector-reference/model-data contract gate"
python3 - <<'PY'
from pathlib import Path

p = Path("scripts/run_as1455_live_data_feature_pipeline.sh")
text = p.read_text(encoding="utf-8")
marker = "# BEGIN AS1455_LIVE_SECTOR_REFERENCE_CONTRACT_GATE"
if marker not in text:
    needle = 'run_features() {\n  info "building live features"\n  mkdir -p "${LIVE_DIR}"\n'
    insert = r"""  # BEGIN AS1455_LIVE_SECTOR_REFERENCE_CONTRACT_GATE
  if [[ "${SKIP_SECTOR_REFERENCE_CONTRACT_CHECK:-0}" != "1" ]]; then
    "${PYTHON}" tools/validate_as1455_model_data_contract.py \
      --model-data "${SECTOR_REFERENCE}" \
      --require-contract \
      --require-adjusted-artifacts
  else
    echo "[WARN] SKIP_SECTOR_REFERENCE_CONTRACT_CHECK=1; sector reference contract validation skipped"
  fi
  # END AS1455_LIVE_SECTOR_REFERENCE_CONTRACT_GATE
"""
    if needle not in text:
        raise SystemExit(f"cannot patch {p}: run_features needle not found")
    text = text.replace(needle, needle + insert, 1)
    p.write_text(text, encoding="utf-8")
    print(f"[OK] patched {p}")
else:
    print(f"[OK] already patched {p}")
PY

info "checking syntax"
python3 -m py_compile tools/validate_as1455_model_data_contract.py

bash -n scripts/clean_bad_as1455_artifacts.sh
bash -n scripts/rebuild_as1455_model_data_clean_adj.sh
bash -n scripts/run_as1455_clean_rebuild_weekly_retrain.sh
bash -n scripts/run_as1455_top5_weekly_retrain_full_v7.sh
bash -n scripts/run_as1455_live_data_feature_pipeline.sh

info "searching for removed wrong entrypoint references"
grep -RIn \
  "run_as1455_extend_weekly_empty_v1\|repair_as1455_20260626_from_raw5m_v4" \
  scripts pipelines features code tools .github 2>/dev/null || true

echo
echo "[DONE] Clean AS1455 adjusted-data patch applied."
echo
echo "Next steps:"
echo "  1) Remove bad derived artifacts:"
echo "       bash scripts/clean_bad_as1455_artifacts.sh"
echo
echo "  2) One-key clean adjusted rebuild + weekly retrain/backtest:"
echo "       bash scripts/run_as1455_clean_rebuild_weekly_retrain.sh"
echo
echo "Optional overrides:"
echo "       REBUILD_END_DATE=2026-06-26 START_DATE=2026-05-16 END_DATE=2026-06-26 bash scripts/run_as1455_clean_rebuild_weekly_retrain.sh"
echo
echo "Git review:"
echo "       git status --short"
echo "       git diff -- scripts tools"
