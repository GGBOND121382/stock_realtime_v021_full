#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=""
REPO_DIR=""
PYTHON_CMD=""
FORCE_ASSETS=0
FORCE_CHAPTER12=0
FORCE_TRAINING=0
SKIP_BACKTEST=0
INSTALL_BACKTEST_DEPS=0
BACKTEST_ONLY=0
PREFLIGHT_BACKTEST=0
INGEST_LOCAL_QUANDL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --force-assets)
      FORCE_ASSETS=1
      shift
      ;;
    --force-chapter12)
      FORCE_CHAPTER12=1
      shift
      ;;
    --force-training)
      FORCE_TRAINING=1
      shift
      ;;
    --skip-backtest)
      SKIP_BACKTEST=1
      shift
      ;;
    --install-backtest-deps)
      INSTALL_BACKTEST_DEPS=1
      shift
      ;;
    --backtest-only)
      BACKTEST_ONLY=1
      shift
      ;;
    --preflight-backtest)
      PREFLIGHT_BACKTEST=1
      BACKTEST_ONLY=1
      shift
      ;;
    --ingest-local-quandl)
      INGEST_LOCAL_QUANDL=1
      BACKTEST_ONLY=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash run_ml4t_ch17_reproduce_ubuntu.sh [options]

Options:
  --work-dir DIR              Directory containing machine-learning-for-trading.
  --repo-dir DIR              Explicit machine-learning-for-trading path.
  --python CMD                Python command, e.g. "python3" or "/path/bin/python".
  --force-assets              Rebuild data/assets.h5 from local wiki_prices.csv.
  --force-chapter12           Re-run Chapter 12 model-data notebook.
  --force-training            Re-run Chapter 17 NN training notebook.
  --skip-backtest             Skip Zipline backtest notebook.
  --install-backtest-deps     pip install Zipline/pyfolio/alphalens dependencies.
  --backtest-only             Require existing data/results and run only Zipline backtest.
  --preflight-backtest        Check backtest env/data/bundle without running notebooks.
  --ingest-local-quandl       Build Zipline's quandl bundle from local data/assets.h5.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$SCRIPT_DIR"
fi
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$WORK_DIR/machine-learning-for-trading"
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo directory not found: $REPO_DIR" >&2
  exit 1
fi
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

if [[ -z "$PYTHON_CMD" ]]; then
  if [[ -x "$(dirname "$WORK_DIR")/.venv/bin/python" ]]; then
    PYTHON_CMD="$(dirname "$WORK_DIR")/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    echo "Python not found. Pass --python /path/to/python." >&2
    exit 1
  fi
fi

echo "=== ML4T Chapter 17 reproducibility runner (Ubuntu/Linux) ==="
echo "WorkDir: $WORK_DIR"
echo "RepoDir: $REPO_DIR"
echo "Python : $PYTHON_CMD"

run_py() {
  # shellcheck disable=SC2086
  $PYTHON_CMD "$@"
}

ensure_packages() {
  local label="$1"
  local imports="$2"
  local packages="$3"

  local missing
  missing="$(run_py - "$imports" <<'PY'
import sys
mods = sys.argv[1].split()
missing = []
for mod in mods:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
print(",".join(missing))
raise SystemExit(1 if missing else 0)
PY
)" || true

  if [[ -z "$missing" ]]; then
    echo "$label dependencies ok"
    return
  fi

  if [[ -z "$packages" ]]; then
    echo "$label dependencies missing: $missing" >&2
    exit 1
  fi

  echo "$label dependencies missing: $missing"
  echo "Installing packages: $packages"
  # shellcheck disable=SC2086
  run_py -m pip install $packages
}

patch_backtest_compatibility() {
  run_py - <<'PY'
from pathlib import Path
import importlib.util


def patch_file(path, replacements):
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_text()
    patched = raw
    for old, new in replacements:
        patched = patched.replace(old, new)
    if patched != raw:
        backup = path.with_suffix(path.suffix + ".ml4t_bak")
        if not backup.exists():
            backup.write_text(raw)
        path.write_text(patched)
        print(f"patched compatibility: {path}")


def patch_alphalens_utils(path):
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_text()
    lines = raw.splitlines(keepends=True)
    out = []
    changed = False
    i = 0
    target = "df.index.levels[0].freq = freq"
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Repair a previously malformed patch:
        #     try:
        #     df.index.levels[0].freq = freq
        # except ValueError:
        #     pass
        if stripped == "try:" and i + 1 < len(lines) and lines[i + 1].strip() == target:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}try:\n")
            out.append(f"{indent}    {target}\n")
            if i + 2 < len(lines) and lines[i + 2].strip() == "except ValueError:":
                out.append(f"{indent}except ValueError:\n")
                if i + 3 < len(lines) and lines[i + 3].strip() == "pass":
                    out.append(f"{indent}    pass\n")
                    i += 4
                else:
                    out.append(f"{indent}    pass\n")
                    i += 3
            else:
                out.append(f"{indent}except ValueError:\n")
                out.append(f"{indent}    pass\n")
                i += 2
            changed = True
            continue

        if stripped == target:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}try:\n")
            out.append(f"{indent}    {target}\n")
            out.append(f"{indent}except ValueError:\n")
            out.append(f"{indent}    pass\n")
            changed = True
            i += 1
            continue

        out.append(line)
        i += 1

    patched = "".join(out)
    if changed and patched != raw:
        backup = path.with_suffix(path.suffix + ".ml4t_bak")
        if not backup.exists():
            backup.write_text(raw)
        path.write_text(patched)
        print(f"patched compatibility: {path}")


spec = importlib.util.find_spec("trading_calendars")
if spec and spec.submodule_search_locations:
    pkg = Path(list(spec.submodule_search_locations)[0])
    patch_file(
        pkg / "calendar_helpers.py",
        [(
            "NP_NAT = np.array([pd.NaT], dtype=np.int64)[0]",
            "NP_NAT = np.datetime64('NaT').astype(np.int64)",
        )],
    )

spec = importlib.util.find_spec("alphalens")
if spec and spec.submodule_search_locations:
    pkg = Path(list(spec.submodule_search_locations)[0])
    patch_alphalens_utils(pkg / "utils.py")
PY
}

patch_hdf_compatibility() {
  local paths=("$@")
  run_py - "${paths[@]}" <<'PY'
import sys
from pathlib import Path

try:
    import tables
except Exception as exc:
    raise SystemExit(f"PyTables is required to patch HDF metadata: {exc}")

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.exists():
        continue
    changed = []
    with tables.open_file(path, mode="a") as h5:
        for node in h5.walk_nodes("/"):
            attrs = getattr(node, "_v_attrs", None)
            if attrs is None or "kind" not in attrs._v_attrnamesuser:
                continue
            kind = attrs.kind
            if isinstance(kind, bytes):
                kind_text = kind.decode()
            else:
                kind_text = str(kind)
            if kind_text == "datetime64[ns]":
                attrs.kind = "datetime64"
                changed.append(node._v_pathname)
    if changed:
        print(f"patched HDF datetime index metadata: {path} -> {', '.join(changed)}")
PY
}

write_local_quandl_extension() {
  run_py - "$ASSETS_PATH" <<'PY'
import os
import sys
from pathlib import Path

assets_path = Path(sys.argv[1]).resolve()
zipline_dir = Path.home() / ".zipline"
zipline_dir.mkdir(parents=True, exist_ok=True)
extension_path = zipline_dir / "extension.py"

content = f'''
from pathlib import Path

import numpy as np
import pandas as pd
from zipline.data.bundles import register

try:
    from zipline.data.bundles import unregister
except Exception:
    unregister = None


ASSETS_H5 = Path(r"{assets_path.as_posix()}")


def ml4t_quandl_bundle(environ,
                       asset_db_writer,
                       minute_bar_writer,
                       daily_bar_writer,
                       adjustment_writer,
                       calendar,
                       start_session,
                       end_session,
                       cache,
                       show_progress,
                       output_dir):
    prices = pd.read_hdf(ASSETS_H5, "quandl/wiki/prices").sort_index()
    if prices.index.names != ["date", "ticker"]:
        prices = prices.reorder_levels(["date", "ticker"]).sort_index()

    tickers = prices.index.get_level_values("ticker").unique().sort_values()
    sid_map = {{ticker: sid for sid, ticker in enumerate(tickers)}}

    stocks = pd.read_hdf(ASSETS_H5, "quandl/wiki/stocks")
    if "code" in stocks.columns:
        stocks = stocks.set_index("code")
    names = stocks["name"].to_dict() if "name" in stocks.columns else {{}}

    calendar_start = pd.Timestamp(calendar.first_session).tz_localize(None)
    calendar_end = pd.Timestamp(calendar.last_session).tz_localize(None)

    grouped = prices.groupby(level="ticker", sort=True)
    metadata = []
    daily_data = []
    splits = []
    dividends = []

    for ticker, df in grouped:
        sid = sid_map[ticker]
        df = df.droplevel("ticker").sort_index()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)

        start = max(df.index.min(), calendar_start)
        end = min(df.index.max(), calendar_end)
        if end < start:
            continue
        df = df.loc[start:end]
        if df.empty:
            continue

        metadata.append({{
            "sid": sid,
            "symbol": ticker,
            "asset_name": names.get(ticker, ticker),
            "start_date": start,
            "end_date": end,
            "first_traded": start,
            "auto_close_date": end + pd.Timedelta(days=1),
            "exchange": "QUANDL",
        }})

        ohlcv = pd.DataFrame({{
            "open": df["adj_open"] if "adj_open" in df else df["open"],
            "high": df["adj_high"] if "adj_high" in df else df["high"],
            "low": df["adj_low"] if "adj_low" in df else df["low"],
            "close": df["adj_close"] if "adj_close" in df else df["close"],
            "volume": df["adj_volume"] if "adj_volume" in df else df["volume"],
        }}).replace([np.inf, -np.inf], np.nan).dropna()
        sessions = calendar.sessions_in_range(
            pd.Timestamp(start),
            pd.Timestamp(end),
        ).tz_localize(None)
        ohlcv = ohlcv.reindex(sessions)
        close = ohlcv["close"].ffill().bfill()
        for col in ["open", "high", "low", "close"]:
            ohlcv[col] = ohlcv[col].fillna(close)
        ohlcv["volume"] = ohlcv["volume"].fillna(0).clip(lower=0)
        daily_data.append((sid, ohlcv))

        if "split_ratio" in df:
            split_rows = df[df["split_ratio"].fillna(1) != 1]
            for date, row in split_rows.iterrows():
                ratio = row["split_ratio"]
                if pd.notna(ratio) and ratio not in (0, 1):
                    splits.append({{"sid": sid, "effective_date": date, "ratio": 1.0 / float(ratio)}})

        if "ex-dividend" in df:
            div_rows = df[df["ex-dividend"].fillna(0) != 0]
            for date, row in div_rows.iterrows():
                amount = row["ex-dividend"]
                if pd.notna(amount) and amount != 0:
                    dividends.append({{
                        "sid": sid,
                        "ex_date": date,
                        "record_date": date,
                        "declared_date": date,
                        "pay_date": date,
                        "amount": float(amount),
                    }})

    exchanges = pd.DataFrame(
        [[
            "QUANDL",
            "QUANDL",
            "US",
        ]],
        columns=["exchange", "canonical_name", "country_code"],
    )
    asset_db_writer.write(
        equities=pd.DataFrame(metadata).set_index("sid"),
        exchanges=exchanges,
    )
    daily_bar_writer.write(daily_data, show_progress=show_progress)
    adjustment_writer.write(
        splits=pd.DataFrame(splits, columns=["sid", "effective_date", "ratio"]),
        dividends=pd.DataFrame(dividends, columns=["sid", "ex_date", "record_date", "declared_date", "pay_date", "amount"]),
    )


if unregister is not None:
    try:
        unregister("quandl")
    except Exception:
        pass

register("quandl", ml4t_quandl_bundle, calendar_name="XNYS")
'''

extension_path.write_text(content)
print(f"wrote local Zipline quandl extension: {extension_path}")
PY
}

ingest_local_quandl_bundle() {
  echo "Building Zipline quandl bundle from local assets.h5..."
  patch_backtest_compatibility
  patch_hdf_compatibility "$ASSETS_PATH"
  write_local_quandl_extension
  run_py -m zipline ingest -b quandl
}

preflight_backtest() {
  local nb_backtest="$1"
  echo "Running backtest preflight checks..."
  patch_backtest_compatibility
  patch_hdf_compatibility "$ASSETS_PATH" "$CHAPTER12_DATA" "$SCORES_PATH" "$PREDS_PATH"
  run_py - "$ASSETS_PATH" "$CHAPTER12_DATA" "$SCORES_PATH" "$PREDS_PATH" "$nb_backtest" <<'PY'
import ast
import json
import sys
from pathlib import Path

import pandas as pd

assets_path, chapter12_data, scores_path, preds_path, nb_path = map(Path, sys.argv[1:])


def require_hdf_key(path, key):
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    with pd.HDFStore(path) as store:
        keys = store.keys()
    wanted = "/" + key.strip("/")
    if wanted not in keys:
        raise SystemExit(f"missing HDF key {wanted} in {path}; found {keys}")
    print(f"ok hdf: {path}::{wanted}")


require_hdf_key(assets_path, "/quandl/wiki/prices")
require_hdf_key(assets_path, "/us_equities/stocks")
require_hdf_key(chapter12_data, "/model_data")
require_hdf_key(scores_path, "/ic_by_day")
require_hdf_key(preds_path, "/predictions")

preds = pd.read_hdf(preds_path, "predictions")
if preds.empty:
    raise SystemExit(f"empty predictions: {preds_path}")
if preds.index.names != ["symbol", "date"]:
    raise SystemExit(f"unexpected predictions index names: {preds.index.names}")
print(f"ok predictions: rows={len(preds):,}, columns={list(preds.columns)}")

mods = [
    "zipline",
    "pyfolio",
    "alphalens",
    "trading_calendars",
    "logbook",
    "pandas_datareader",
]
for mod in mods:
    __import__(mod)
print("ok imports:", ", ".join(mods))

from zipline.data import bundles

try:
    bundle = bundles.load("quandl")
except Exception as exc:
    raise SystemExit(
        "failed to load Zipline bundle 'quandl'. "
        "Run zipline ingest or check ~/.zipline before backtesting. "
        f"Original error: {type(exc).__name__}: {exc}"
    )
print("ok zipline bundle: quandl")

tickers = preds.index.get_level_values("symbol").unique().tolist()
try:
    assets = bundle.asset_finder.lookup_symbols(tickers, as_of_date=None)
except Exception as exc:
    raise SystemExit(
        "failed to map prediction tickers to Zipline assets. "
        "The 'quandl' bundle may not match test_preds.h5 symbols. "
        f"Original error: {type(exc).__name__}: {exc}"
    )
missing = [ticker for ticker, asset in zip(tickers, assets) if asset is None]
if missing:
    raise SystemExit(
        f"Zipline bundle did not resolve {len(missing)} prediction tickers; "
        f"examples: {missing[:10]}"
    )
print(f"ok asset lookup: {len(assets):,} prediction tickers")

nb = json.loads(nb_path.read_text(encoding="utf-8"))
for cell_no, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    code = "\n".join(
        line for line in "".join(cell.get("source", [])).splitlines()
        if not line.lstrip().startswith(("%", "!", "?"))
    )
    if code.strip():
        ast.parse(code, filename=f"{nb_path}:{cell_no}")
print(f"ok notebook syntax: {nb_path}")
PY
  echo "Backtest preflight passed. No training or backtest notebook was executed."
}

test_hdf_key() {
  local path="$1"
  local key="$2"
  [[ -f "$path" ]] || return 1
  run_py - "$path" "$key" <<'PY' >/dev/null 2>&1
import sys
import pandas as pd
path, key = sys.argv[1], sys.argv[2]
with pd.HDFStore(path) as store:
    keys = store.keys()
    ok = key in keys or ("/" + key.lstrip("/")) in keys
raise SystemExit(0 if ok else 1)
PY
}

invoke_notebook() {
  local notebook="$1"
  local output="$2"
  local notebook_dir
  local notebook_name
  notebook_dir="$(cd "$(dirname "$notebook")" && pwd)"
  notebook_name="$(basename "$notebook")"
  echo "Executing notebook: $notebook_dir/$notebook_name"
  (
    cd "$notebook_dir"
    run_py -m jupyter nbconvert \
      --to notebook \
      --execute "$notebook_name" \
      --output "$output" \
      --ExecutePreprocessor.timeout=-1
  )
}

patch_notebook_compatibility() {
  local path="$1"
  local repair_script="$WORK_DIR/repair_ch17_notebooks.py"
  if [[ -f "$repair_script" ]]; then
    run_py "$repair_script" "$path"
  fi
  run_py - "$path" <<'PY'
from pathlib import Path
import sys
import re

path = Path(sys.argv[1])
raw = path.read_text(encoding="utf-8")
patched = raw
patched = re.sub(
    r"if hasattr\(status, 'expect_partial'\):\n\s+if hasattr\(status, 'expect_partial'\):\n\s*status\.expect_partial\(\)",
    "if hasattr(status, 'expect_partial'):\\n            status.expect_partial()",
    patched,
)
patched = patched.replace("null_counts=True", "show_counts=True")
patched = patched.replace(".sort_index(1)", ".sort_index(axis=1)")
patched = patched.replace("pd.np.arange", "np.arange")
patched = patched.replace(
    "sys.path.insert(1, os.path.join(sys.path[0], '..'))",
    "sys.path.insert(1, str(Path.cwd().resolve().parent))",
)
patched = patched.replace(
    "prices.groupby(level='symbol').close.apply(RSI)",
    "prices.groupby(level='symbol', group_keys=False).close.apply(RSI)",
)
patched = patched.replace(
    ".groupby(level='symbol')\\n                      .close\\n                      .apply(compute_bb)",
    ".groupby(level='symbol', group_keys=False)\\n                      .close\\n                      .apply(compute_bb)",
)
patched = patched.replace(
    "prices.groupby(level='symbol').close.apply(talib.PPO)",
    "prices.groupby(level='symbol', group_keys=False).close.apply(talib.PPO)",
)
patched = patched.replace(
    ".groupby(level='date')\\n                             .apply(lambda x: pd.qcut",
    ".groupby(level='date', group_keys=False)\\n                             .apply(lambda x: pd.qcut",
)
patched = patched.replace(
    "preds.groupby(level='date').apply(lambda x: spearmanr(x.actual, x[epoch])[0])",
    "preds.groupby(level='date', group_keys=False).apply(lambda x: spearmanr(x.actual, x[epoch])[0])",
)
patched = patched.replace(
    "ic = []\\nscaler = StandardScaler()\\nfor params in param_grid:",
    "if (results_path / 'scores.h5').exists():\\n    print('Skipping NN CV training because results/scores.h5 already exists.')\\n    param_grid = []\\n\\nic = []\\nscaler = StandardScaler()\\nfor params in param_grid:",
)
patched = patched.replace(
    "model_data.columns = [s.split('_')[-1] for s in model_data.columns]\\n    model = sm.OLS",
    "model_data.columns = [s.split('_')[-1] for s in model_data.columns]\\n    model_data = model_data.apply(pd.to_numeric, errors='coerce').astype(float)\\n    model = sm.OLS",
)
patched = patched.replace("f'ckpt_{fold}_{epoch}'", "f'ckpt_{fold}_{epoch}.weights.h5'")
patched = patched.replace(
    "        status.expect_partial()",
    "        if hasattr(status, 'expect_partial'):\\n            status.expect_partial()",
)
patched = re.sub(
    r"if hasattr\(status, 'expect_partial'\):\n\s+if hasattr\(status, 'expect_partial'\):\n\s*status\.expect_partial\(\)",
    "if hasattr(status, 'expect_partial'):\\n            status.expect_partial()",
    patched,
)
patched = patched.replace(
    "pd.Int64Index([asset.sid for asset in assets])",
    "pd.Index([asset.sid for asset in assets], dtype='int64')",
)
if "FRED benchmark download failed" not in patched:
    patched = patched.replace(
        "benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()\\nbenchmark = benchmark.pct_change().tz_localize('UTC')",
        "try:\\n    benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()\\n    benchmark = benchmark.pct_change().tz_localize('UTC')\\nexcept Exception as exc:\\n    print(f'FRED benchmark download failed; using zero benchmark aligned to strategy returns: {exc}')\\n    benchmark = returns.copy() * 0",
    )
    patched = patched.replace(
        "\"benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()\\n\",\n    \"benchmark = benchmark.pct_change().tz_localize('UTC')\"",
        "\"try:\\n\",\n    \"    benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()\\n\",\n    \"    benchmark = benchmark.pct_change().tz_localize('UTC')\\n\",\n    \"except Exception as exc:\\n\",\n    \"    print(f'FRED benchmark download failed; using zero benchmark aligned to strategy returns: {exc}')\\n\",\n    \"    benchmark = returns.copy() * 0\"",
    )
if "start_date = pd.Timestamp(start_date).tz_localize(None)" not in patched:
    patched = patched.replace(
        "start_date, end_date = dates.min(), dates.max()",
        "start_date, end_date = dates.min(), dates.max()\\nstart_date = pd.Timestamp(start_date).tz_localize(None)\\nend_date = pd.Timestamp(end_date).tz_localize(None)",
    )
    patched = patched.replace(
        "\"start_date, end_date = dates.min(), dates.max()\"",
        "\"start_date, end_date = dates.min(), dates.max()\\n\",\n    \"start_date = pd.Timestamp(start_date).tz_localize(None)\\n\",\n    \"end_date = pd.Timestamp(end_date).tz_localize(None)\"",
    )
if "zipline_returns.csv" not in patched:
    patched = patched.replace(
        "returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)",
        "returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)\\nreturns.to_csv(results_path / 'zipline_returns.csv')\\npositions.to_csv(results_path / 'zipline_positions.csv')\\ntransactions.to_csv(results_path / 'zipline_transactions.csv')\\nresults.to_pickle(results_path / 'zipline_results.pkl')",
    )
    patched = patched.replace(
        "\"returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)\"",
        "\"returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)\\n\",\n    \"returns.to_csv(results_path / 'zipline_returns.csv')\\n\",\n    \"positions.to_csv(results_path / 'zipline_positions.csv')\\n\",\n    \"transactions.to_csv(results_path / 'zipline_transactions.csv')\\n\",\n    \"results.to_pickle(results_path / 'zipline_results.pkl')\"",
    )
if "PyFolio full tear sheet failed" not in patched:
    patched = patched.replace(
        "pf.create_full_tear_sheet(returns, \\n                          positions=positions, \\n                          transactions=transactions,\\n                          benchmark_rets=benchmark,\\n                          live_start_date=LIVE_DATE, \\n                          round_trips=True)",
        "try:\\n    pf.create_full_tear_sheet(returns, \\n                              positions=positions, \\n                              transactions=transactions,\\n                              benchmark_rets=benchmark,\\n                              live_start_date=LIVE_DATE, \\n                              round_trips=True)\\nexcept Exception as exc:\\n    print(f'PyFolio full tear sheet failed; saved raw backtest outputs and continuing: {type(exc).__name__}: {exc}')",
    )
    patched = patched.replace(
        "\"pf.create_full_tear_sheet(returns, \\n\",\\n    \"                          positions=positions, \\n\",\\n    \"                          transactions=transactions,\\n\",\\n    \"                          benchmark_rets=benchmark,\\n\",\\n    \"                          live_start_date=LIVE_DATE, \\n\",\\n    \"                          round_trips=True)\"",
        "\"try:\\n\",\\n    \"    pf.create_full_tear_sheet(returns, \\n\",\\n    \"                              positions=positions, \\n\",\\n    \"                              transactions=transactions,\\n\",\\n    \"                              benchmark_rets=benchmark,\\n\",\\n    \"                              live_start_date=LIVE_DATE, \\n\",\\n    \"                              round_trips=True)\\n\",\\n    \"except Exception as exc:\\n\",\\n    \"    print(f'PyFolio full tear sheet failed; saved raw backtest outputs and continuing: {type(exc).__name__}: {exc}')\"",
    )
if patched != raw:
    path.write_text(patched, encoding="utf-8")
    print(f"Patched notebook compatibility: {path}")
PY
}

build_assets_from_wiki_csv() {
  local data_dir="$1"
  run_py - "$data_dir" <<'PY'
from pathlib import Path
import sys
import pandas as pd

base = Path(sys.argv[1])
store_path = base / "assets.h5"
prices_path = base / "wiki_prices.csv"
stocks_path = base / "wiki_stocks.csv"
us_meta_path = base / "us_equities_meta_data.csv"

for path in [prices_path, stocks_path, us_meta_path]:
    if not path.exists():
        raise SystemExit(f"missing {path}")

if store_path.exists():
    store_path.unlink()

print(f"Reading {prices_path} ...", flush=True)
prices = (
    pd.read_csv(
        prices_path,
        parse_dates=["date"],
        index_col=["date", "ticker"],
    )
    .sort_index()
)
print(f"Writing {store_path}:/quandl/wiki/prices ...", flush=True)
with pd.HDFStore(store_path) as store:
    store.put("quandl/wiki/prices", prices)
del prices

print("Writing /quandl/wiki/stocks ...", flush=True)
wiki_stocks = pd.read_csv(stocks_path)
with pd.HDFStore(store_path) as store:
    store.put("quandl/wiki/stocks", wiki_stocks)
del wiki_stocks

print("Writing /us_equities/stocks ...", flush=True)
us_meta = pd.read_csv(us_meta_path)
with pd.HDFStore(store_path) as store:
    store.put("us_equities/stocks", us_meta.set_index("ticker"))
    print("Keys:", store.keys(), flush=True)
PY
}

echo "Checking Python dependencies..."
if [[ "$BACKTEST_ONLY" -eq 0 ]]; then
  ensure_packages \
    "Chapter 12/17 training" \
    "numpy pandas tables sklearn matplotlib seaborn scipy statsmodels jupyter nbconvert nbformat talib tensorflow tensorboard" \
    "numpy pandas tables scikit-learn matplotlib seaborn scipy statsmodels jupyter nbconvert nbformat TA-Lib tensorflow tensorboard"
else
  ensure_packages \
    "Backtest notebook execution" \
    "numpy pandas tables jupyter nbconvert nbformat matplotlib seaborn" \
    "numpy pandas tables jupyter nbconvert nbformat matplotlib seaborn"
fi

patch_backtest_compatibility

if [[ "$INSTALL_BACKTEST_DEPS" -eq 1 ]]; then
  ensure_packages \
    "Zipline backtest" \
    "zipline pyfolio alphalens trading_calendars logbook pandas_datareader" \
    "zipline-reloaded pyfolio-reloaded alphalens-reloaded trading-calendars logbook pandas_datareader"
fi

DATA_DIR="$REPO_DIR/data"
ASSETS_PATH="$DATA_DIR/assets.h5"
CHAPTER12_DIR="$REPO_DIR/12_gradient_boosting_machines"
CHAPTER17_DIR="$REPO_DIR/17_deep_learning"
CHAPTER12_DATA="$CHAPTER12_DIR/data.h5"
CHAPTER17_RESULTS="$CHAPTER17_DIR/results"
SCORES_PATH="$CHAPTER17_RESULTS/scores.h5"
PREDS_PATH="$CHAPTER17_RESULTS/test_preds.h5"

if [[ "$BACKTEST_ONLY" -eq 1 ]]; then
  echo "Backtest-only mode: not rebuilding assets.h5."
elif [[ "$FORCE_ASSETS" -eq 1 ]] || ! test_hdf_key "$ASSETS_PATH" "/quandl/wiki/prices" || ! test_hdf_key "$ASSETS_PATH" "/us_equities/stocks"; then
  echo "Building data/assets.h5 from existing local WIKI csv files..."
  build_assets_from_wiki_csv "$DATA_DIR"
else
  echo "Skipping assets.h5 build; required keys already exist."
fi

NB12="$CHAPTER12_DIR/04_preparing_the_model_data.ipynb"
NB17_TRAIN="$CHAPTER17_DIR/04_optimizing_a_NN_architecture_for_trading.ipynb"
NB17_BACKTEST="$CHAPTER17_DIR/05_backtesting_with_zipline.ipynb"

patch_notebook_compatibility "$NB12"
patch_notebook_compatibility "$NB17_TRAIN"
patch_notebook_compatibility "$NB17_BACKTEST"

if [[ "$PREFLIGHT_BACKTEST" -eq 1 ]]; then
  preflight_backtest "$NB17_BACKTEST"
  exit 0
fi

patch_hdf_compatibility "$ASSETS_PATH" "$CHAPTER12_DATA" "$SCORES_PATH" "$PREDS_PATH"

if [[ "$INGEST_LOCAL_QUANDL" -eq 1 ]]; then
  ingest_local_quandl_bundle
  echo "Local Zipline quandl bundle ingest completed."
  exit 0
fi

if [[ "$BACKTEST_ONLY" -eq 1 ]]; then
  if ! test_hdf_key "$ASSETS_PATH" "/quandl/wiki/prices" || ! test_hdf_key "$ASSETS_PATH" "/us_equities/stocks"; then
    echo "Backtest-only mode requires existing data/assets.h5 with /quandl/wiki/prices and /us_equities/stocks." >&2
    exit 1
  fi
  if ! test_hdf_key "$CHAPTER12_DATA" "/model_data"; then
    echo "Backtest-only mode requires existing 12_gradient_boosting_machines/data.h5::/model_data." >&2
    exit 1
  fi
  if ! test_hdf_key "$SCORES_PATH" "/ic_by_day" || ! test_hdf_key "$PREDS_PATH" "/predictions"; then
    echo "Backtest-only mode requires existing scores.h5::/ic_by_day and test_preds.h5::/predictions." >&2
    exit 1
  fi
  echo "Backtest-only mode: existing data/results verified."
elif [[ "$FORCE_CHAPTER12" -eq 1 ]] || ! test_hdf_key "$CHAPTER12_DATA" "/model_data"; then
  invoke_notebook "$NB12" "04_preparing_the_model_data.executed.ipynb"
else
  echo "Skipping Chapter 12 model data; 12_gradient_boosting_machines/data.h5::/model_data already exists."
fi

if [[ "$BACKTEST_ONLY" -eq 0 ]]; then
  if [[ "$FORCE_TRAINING" -eq 1 ]] || ! test_hdf_key "$SCORES_PATH" "/ic_by_day" || ! test_hdf_key "$PREDS_PATH" "/predictions"; then
    invoke_notebook "$NB17_TRAIN" "04_optimizing_a_NN_architecture_for_trading.executed.ipynb"
  else
    echo "Skipping Chapter 17 NN training; scores.h5 and test_preds.h5 already exist."
  fi
fi

if [[ "$SKIP_BACKTEST" -eq 1 ]]; then
  echo "Skipping Zipline backtest by request."
else
  patch_backtest_compatibility
  if ! run_py - <<'PY' >/dev/null 2>&1
import zipline, pyfolio, alphalens, trading_calendars, logbook, pandas_datareader
PY
  then
    cat >&2 <<'EOF'
Zipline/pyfolio/alphalens dependencies are not importable in this Python environment.
Re-run with --install-backtest-deps on a compatible Linux/WSL/Docker environment,
or use --skip-backtest to reproduce training/prediction only.
No local market data was re-downloaded.
EOF
    exit 1
  fi
  invoke_notebook "$NB17_BACKTEST" "05_backtesting_with_zipline.executed.ipynb"
fi

echo
echo "Done. Key outputs:"
echo "  $CHAPTER12_DATA"
echo "  $SCORES_PATH"
echo "  $PREDS_PATH"
echo "  $CHAPTER17_DIR/04_optimizing_a_NN_architecture_for_trading.executed.ipynb"
if [[ "$SKIP_BACKTEST" -eq 0 ]]; then
  echo "  $CHAPTER17_DIR/05_backtesting_with_zipline.executed.ipynb"
fi
