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
import importlib.util
import sys
mods = sys.argv[1].split()
missing = [m for m in mods if importlib.util.find_spec(m) is None]
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
ensure_packages \
  "Chapter 12/17 training" \
  "numpy pandas tables sklearn matplotlib seaborn scipy statsmodels jupyter nbconvert nbformat talib tensorflow tensorboard" \
  "numpy pandas tables scikit-learn matplotlib seaborn scipy statsmodels jupyter nbconvert nbformat TA-Lib tensorflow tensorboard"

if [[ "$INSTALL_BACKTEST_DEPS" -eq 1 ]]; then
  ensure_packages \
    "Zipline backtest" \
    "zipline pyfolio alphalens trading_calendars logbook" \
    "zipline-reloaded pyfolio-reloaded alphalens-reloaded trading-calendars logbook"
fi

DATA_DIR="$REPO_DIR/data"
ASSETS_PATH="$DATA_DIR/assets.h5"
CHAPTER12_DIR="$REPO_DIR/12_gradient_boosting_machines"
CHAPTER17_DIR="$REPO_DIR/17_deep_learning"
CHAPTER12_DATA="$CHAPTER12_DIR/data.h5"
CHAPTER17_RESULTS="$CHAPTER17_DIR/results"
SCORES_PATH="$CHAPTER17_RESULTS/scores.h5"
PREDS_PATH="$CHAPTER17_RESULTS/test_preds.h5"

if [[ "$FORCE_ASSETS" -eq 1 ]] || ! test_hdf_key "$ASSETS_PATH" "/quandl/wiki/prices" || ! test_hdf_key "$ASSETS_PATH" "/us_equities/stocks"; then
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

if [[ "$FORCE_CHAPTER12" -eq 1 ]] || ! test_hdf_key "$CHAPTER12_DATA" "/model_data"; then
  invoke_notebook "$NB12" "04_preparing_the_model_data.executed.ipynb"
else
  echo "Skipping Chapter 12 model data; 12_gradient_boosting_machines/data.h5::/model_data already exists."
fi

if [[ "$FORCE_TRAINING" -eq 1 ]] || ! test_hdf_key "$SCORES_PATH" "/ic_by_day" || ! test_hdf_key "$PREDS_PATH" "/predictions"; then
  invoke_notebook "$NB17_TRAIN" "04_optimizing_a_NN_architecture_for_trading.executed.ipynb"
else
  echo "Skipping Chapter 17 NN training; scores.h5 and test_preds.h5 already exist."
fi

if [[ "$SKIP_BACKTEST" -eq 1 ]]; then
  echo "Skipping Zipline backtest by request."
else
  if ! run_py - <<'PY' >/dev/null 2>&1
import zipline, pyfolio, alphalens, trading_calendars, logbook
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
