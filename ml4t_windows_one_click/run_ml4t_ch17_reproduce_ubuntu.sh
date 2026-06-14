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
SELF_TEST_PATCHES=0
SELF_TEST_DATA_FORMAT=0
SELF_TEST_NEGATIVE=0
LOCAL_BACKTEST=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      [[ $# -ge 2 ]] || { echo "--work-dir requires a directory argument" >&2; exit 2; }
      WORK_DIR="$2"
      shift 2
      ;;
    --repo-dir)
      [[ $# -ge 2 ]] || { echo "--repo-dir requires a directory argument" >&2; exit 2; }
      REPO_DIR="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "--python requires a command/path argument" >&2; exit 2; }
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
    --self-test-patches)
      SELF_TEST_PATCHES=1
      BACKTEST_ONLY=1
      shift
      ;;
    --self-test-data-format)
      SELF_TEST_DATA_FORMAT=1
      BACKTEST_ONLY=1
      shift
      ;;
    --self-test-negative)
      SELF_TEST_NEGATIVE=1
      BACKTEST_ONLY=1
      shift
      ;;
    --local-backtest)
      LOCAL_BACKTEST=1
      BACKTEST_ONLY=1
      SKIP_BACKTEST=1
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
  --self-test-patches         Run local patch self-tests without training/backtesting.
  --self-test-data-format     Build minimal synthetic HDF5 files and validate data format only.
  --self-test-negative        Run synthetic broken-data cases that must fail validation.
  --local-backtest            Run dependency-light local long/short backtest instead of Zipline.

Common server commands after training has completed:
  bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh --backtest-only --preflight-backtest
  bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh --backtest-only --ingest-local-quandl
  bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh --backtest-only
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
  validate_backtest_data
  write_local_quandl_extension
  run_py -m zipline ingest -b quandl
}

validate_backtest_data() {
  local validator="$WORK_DIR/validate_ch17_backtest_data.py"
  if [[ ! -f "$validator" ]]; then
    echo "Backtest data validator not found, skipping: $validator"
    return
  fi
  echo "Validating Chapter 17 backtest data without Zipline..."
  run_py "$validator" \
    --repo-dir "$REPO_DIR" \
    --patch-hdf-metadata \
    --output "$BACKTEST_VALIDATION_REPORT"
}

preflight_backtest() {
  local nb_backtest="$1"
  echo "Running backtest preflight checks..."
  patch_backtest_compatibility
  patch_hdf_compatibility "$ASSETS_PATH" "$CHAPTER12_DATA" "$SCORES_PATH" "$PREDS_PATH"
  validate_backtest_data
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
import json

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
if "Benchmark disabled; using zero benchmark aligned to strategy returns" not in patched:
    patched = patched.replace(
        "benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()\\nbenchmark = benchmark.pct_change().tz_localize('UTC')",
        "print('Benchmark disabled; using zero benchmark aligned to strategy returns.')\\nbenchmark = returns.copy() * 0",
    )
    patched = patched.replace(
        "\"benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()\\n\",\n    \"benchmark = benchmark.pct_change().tz_localize('UTC')\"",
        "\"print('Benchmark disabled; using zero benchmark aligned to strategy returns.')\\n\",\n    \"benchmark = returns.copy() * 0\"",
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
if path.suffix == ".ipynb":
    nb = json.loads(patched)
    nb_changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if "extract_rets_pos_txn_from_zipline" in src:
            cell["source"] = [
                "results.to_pickle(results_path / 'zipline_results.pkl')\n",
                "returns = results['returns'].copy()\n",
                "returns.to_csv(results_path / 'zipline_returns.csv')\n",
                "positions = pd.DataFrame()\n",
                "transactions = pd.DataFrame()\n",
                "positions.to_csv(results_path / 'zipline_positions.csv')\n",
                "transactions.to_csv(results_path / 'zipline_transactions.csv')\n",
                "print('Skipping PyFolio position/transaction extraction; raw Zipline results and returns are saved in results/.')",
            ]
            nb_changed = True

        if "pf.create_full_tear_sheet(" in src:
            cell["source"] = [
                "print('Skipping PyFolio full tear sheet; raw backtest outputs are saved in results/.')",
            ]
            nb_changed = True

        if (
            "start_date, end_date = dates.min(), dates.max()" in src
            and "start_date = pd.Timestamp(start_date).tz_localize(None)" not in src
        ):
            src = src.replace(
                "start_date, end_date = dates.min(), dates.max()",
                "start_date, end_date = dates.min(), dates.max()\n"
                "start_date = pd.Timestamp(start_date).tz_localize(None)\n"
                "end_date = pd.Timestamp(end_date).tz_localize(None)",
            )
            cell["source"] = [line + "\n" for line in src.splitlines()]
            if src and not src.endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            nb_changed = True

        if (
            "benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()" in src
            and "Benchmark disabled; using zero benchmark aligned to strategy returns" not in src
        ):
            cell["source"] = [
                "print('Benchmark disabled; using zero benchmark aligned to strategy returns.')\n",
                "benchmark = returns.copy() * 0",
            ]
            nb_changed = True

    if nb_changed:
        patched = json.dumps(nb, ensure_ascii=False, indent=1)
        if "pf.create_full_tear_sheet(" in patched:
            raise SystemExit("Notebook patch failed: pf.create_full_tear_sheet is still present.")
        if "extract_rets_pos_txn_from_zipline" in patched:
            raise SystemExit("Notebook patch failed: extract_rets_pos_txn_from_zipline is still present.")
if patched != raw:
    path.write_text(patched, encoding="utf-8")
    print(f"Patched notebook compatibility: {path}")
PY
}

self_test_patches() {
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' RETURN

  run_py - "$tmp_dir" <<'PY'
import json
import sys
from pathlib import Path

tmp = Path(sys.argv[1])

cells = [
    "from pathlib import Path\nimport pandas as pd\nimport pandas_datareader.data as web\nimport pyfolio as pf\n",
    "dates = predictions.index.get_level_values('date')\nstart_date, end_date = dates.min(), dates.max()",
    "returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)",
    "benchmark = web.DataReader('SP500', 'fred', '2014', '2018').squeeze()\nbenchmark = benchmark.pct_change().tz_localize('UTC')",
    "pf.create_full_tear_sheet(returns, \n                          positions=positions, \n                          transactions=transactions,\n                          benchmark_rets=benchmark,\n                          live_start_date=LIVE_DATE, \n                          round_trips=True)",
]

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }
        for src in cells
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(tmp / "patch_self_test.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
PY

  patch_notebook_compatibility "$tmp_dir/patch_self_test.ipynb"
  patch_notebook_compatibility "$tmp_dir/patch_self_test.ipynb"

  run_py - "$tmp_dir/patch_self_test.ipynb" <<'PY'
import ast
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_text(encoding="utf-8")
nb = json.loads(raw)
code_cells = ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]

checks = {
    "removed_extract_rets_pos_txn": "extract_rets_pos_txn_from_zipline" not in raw,
    "removed_full_tear_sheet": "pf.create_full_tear_sheet(" not in raw,
    "has_returns_save": raw.count("zipline_returns.csv") == 1,
    "has_positions_skip": raw.count("Skipping PyFolio position/transaction extraction") == 1,
    "has_full_tear_skip": raw.count("Skipping PyFolio full tear sheet") == 1,
    "has_aligned_zero_benchmark": raw.count("Benchmark disabled; using zero benchmark aligned to strategy returns") == 1
    and raw.count("benchmark = returns.copy() * 0") == 1,
    "has_naive_dates": raw.count("start_date = pd.Timestamp(start_date).tz_localize(None)") == 1,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"patch self-test failed: {failed}")
for i, code in enumerate(code_cells):
    if code.strip():
        ast.parse(code, filename=f"{path}:{i}")
print("notebook patch self-test passed")
PY

  run_py - "$tmp_dir" <<'PY'
import sys
from pathlib import Path

tmp = Path(sys.argv[1])

calendar_helpers = tmp / "calendar_helpers.py"
calendar_helpers.write_text(
    "import numpy as np\n"
    "import pandas as pd\n"
    "NP_NAT = np.array([pd.NaT], dtype=np.int64)[0]\n",
    encoding="utf-8",
)
raw = calendar_helpers.read_text(encoding="utf-8")
patched = raw.replace(
    "NP_NAT = np.array([pd.NaT], dtype=np.int64)[0]",
    "NP_NAT = np.datetime64('NaT').astype(np.int64)",
)
calendar_helpers.write_text(patched, encoding="utf-8")
patched_again = calendar_helpers.read_text(encoding="utf-8").replace(
    "NP_NAT = np.array([pd.NaT], dtype=np.int64)[0]",
    "NP_NAT = np.datetime64('NaT').astype(np.int64)",
)
calendar_helpers.write_text(patched_again, encoding="utf-8")
text = calendar_helpers.read_text(encoding="utf-8")
if text.count("NP_NAT = np.datetime64('NaT').astype(np.int64)") != 1:
    raise SystemExit(f"trading_calendars patch self-test failed:\n{text}")
compile(text, str(calendar_helpers), "exec")

alphalens_utils = tmp / "alphalens_utils.py"
alphalens_utils.write_text(
    "def set_freq(df, freq):\n"
    "    df.index.levels[0].freq = freq\n"
    "    return df\n",
    encoding="utf-8",
)
raw = alphalens_utils.read_text(encoding="utf-8")
lines = raw.splitlines(keepends=True)
out = []
target = "df.index.levels[0].freq = freq"
for line in lines:
    if line.strip() == target:
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f"{indent}try:\n")
        out.append(f"{indent}    {target}\n")
        out.append(f"{indent}except ValueError:\n")
        out.append(f"{indent}    pass\n")
    else:
        out.append(line)
patched = "".join(out)
alphalens_utils.write_text(patched, encoding="utf-8")
text = alphalens_utils.read_text(encoding="utf-8")
if text.count("try:") != 1 or text.count("except ValueError:") != 1 or text.count(target) != 1:
    raise SystemExit(f"alphalens patch self-test failed:\n{text}")
compile(text, str(alphalens_utils), "exec")

print("package compatibility patch self-test passed")
PY

  if run_py - <<'PY' >/dev/null 2>&1
import tables
PY
  then
  run_py - "$tmp_dir/self_test.h5" <<'PY'
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
idx = pd.MultiIndex.from_product(
    [["A"], pd.date_range("2020-01-01", periods=2)],
    names=["symbol", "date"],
)
pd.DataFrame({"x": [1.0, 2.0]}, index=idx).to_hdf(path, "predictions")
PY
  patch_hdf_compatibility "$tmp_dir/self_test.h5"
  run_py - "$tmp_dir/self_test.h5" <<'PY'
import sys
from pathlib import Path
import tables

path = Path(sys.argv[1])
with tables.open_file(path) as h5:
    kinds = [
        str(node._v_attrs.kind)
        for node in h5.walk_nodes("/")
        if hasattr(node, "_v_attrs") and "kind" in node._v_attrs._v_attrnamesuser
    ]
if "datetime64[ns]" in kinds:
    raise SystemExit(f"HDF patch self-test failed: {kinds}")
print("HDF metadata patch self-test passed")
PY
  else
    echo "Skipping HDF metadata patch self-test because PyTables is not installed in this Python."
  fi

  echo "All patch self-tests passed. No training/backtest was executed."
}

self_test_data_format() {
  local validator="$WORK_DIR/validate_ch17_backtest_data.py"
  [[ -f "$validator" ]] || { echo "Backtest data validator not found: $validator" >&2; exit 1; }
  run_py "$validator" --self-test-synthetic
  echo "Synthetic Chapter 17 data-format self-test passed. No training/backtest was executed."
}

run_pre_backtest_parity_guards() {
  local validator="$WORK_DIR/validate_ch17_backtest_data.py"
  [[ -f "$validator" ]] || { echo "Backtest data validator not found: $validator" >&2; exit 1; }

  echo "Running pre-backtest parity guards..."
  echo "  1/3 notebook patch self-test"
  self_test_patches
  echo "  2/3 synthetic fake-bundle/fake-Zipline full-flow self-test"
  run_py "$validator" --self-test-synthetic
  echo "  3/3 synthetic broken-data negative self-tests"
  run_py "$validator" --self-test-negative
  echo "Pre-backtest parity guards passed."
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

if [[ "$SELF_TEST_PATCHES" -eq 1 ]]; then
  echo "Running patch self-tests..."
  ensure_packages \
    "Patch self-test" \
    "pandas" \
    "pandas"
  self_test_patches
  exit 0
fi

if [[ "$SELF_TEST_DATA_FORMAT" -eq 1 ]]; then
  echo "Running synthetic data-format self-test..."
  ensure_packages \
    "Data-format self-test" \
    "numpy pandas tables" \
    "numpy pandas tables"
  self_test_data_format
  exit 0
fi

if [[ "$SELF_TEST_NEGATIVE" -eq 1 ]]; then
  echo "Running synthetic negative data-flow self-tests..."
  ensure_packages \
    "Negative data-flow self-test" \
    "numpy pandas tables" \
    "numpy pandas tables"
  validator="$WORK_DIR/validate_ch17_backtest_data.py"
  [[ -f "$validator" ]] || { echo "Backtest data validator not found: $validator" >&2; exit 1; }
  run_py "$validator" --self-test-negative
  exit 0
fi

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
BACKTEST_VALIDATION_REPORT="$CHAPTER17_RESULTS/backtest_data_validation.json"

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
  validate_backtest_data
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

if [[ "$LOCAL_BACKTEST" -eq 1 ]]; then
  local_backtest="$WORK_DIR/run_ch17_local_backtest.py"
  [[ -f "$local_backtest" ]] || { echo "Local backtest script not found: $local_backtest" >&2; exit 1; }
  run_py "$local_backtest" --repo-dir "$REPO_DIR" --output-dir "$WORK_DIR/out"
  exit 0
fi

if [[ "$SKIP_BACKTEST" -eq 1 ]]; then
  echo "Skipping Zipline backtest by request."
else
  patch_backtest_compatibility
  run_pre_backtest_parity_guards
  validate_backtest_data
  if ! run_py - <<'PY' >/dev/null 2>&1
import zipline, pyfolio, alphalens, trading_calendars, logbook, pandas_datareader
PY
  then
    cat >&2 <<'EOF'
Zipline/pyfolio/alphalens dependencies are not importable in this Python environment.
Data/results were checked before this point if they existed. To continue on Ubuntu, run:
  bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh --backtest-only --install-backtest-deps

If the server already has a separate Zipline environment, point the runner at it:
  bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh --backtest-only --python /path/to/venv/bin/python

If the Zipline 'quandl' bundle has not been ingested from local assets.h5 yet, run:
  bash ml4t_windows_one_click/run_ml4t_ch17_reproduce_ubuntu.sh --backtest-only --ingest-local-quandl

Use --skip-backtest only when you want to reproduce training/prediction without backtesting.
No local market data was re-downloaded.
EOF
    exit 1
  fi
  preflight_backtest "$NB17_BACKTEST"
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
