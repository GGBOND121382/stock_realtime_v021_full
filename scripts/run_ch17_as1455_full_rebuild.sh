#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-all}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv_as1455}"
INSTALL_MISSING_DEPS="${INSTALL_MISSING_DEPS:-1}"
REBUILD_ROOT="${REBUILD_ROOT:-saved_data/ashare_ml4t/rebuild_ch17_as1455}"
STATE_DIR="${STATE_DIR:-$REBUILD_ROOT/state}"
CPU_THREADS="${CPU_THREADS:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$CPU_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

bash -n scripts/as1455_python_memory_guard.sh
bash -n scripts/rebuild_ch17_as1455_from_scratch.sh
command -v "$BASE_PYTHON" >/dev/null 2>&1 || {
  printf '[ERROR] base Python not found: %s\n' "$BASE_PYTHON" >&2
  exit 127
}

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV_DIR"
  fi
  REAL_PYTHON="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")/bin/python"
else
  REAL_PYTHON="$(command -v "$BASE_PYTHON")"
fi

[[ -x "$REAL_PYTHON" ]] || {
  printf '[ERROR] Python environment is unavailable: %s\n' "$REAL_PYTHON" >&2
  exit 1
}

if ! "$REAL_PYTHON" - <<'PY'
mods = ['numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'baostock', 'tables', 'talib', 'tensorflow', 'matplotlib']
for name in mods:
    __import__(name)
print('[OK] Python dependency imports passed')
PY
then
  [[ "$INSTALL_MISSING_DEPS" == "1" ]] || {
    printf '[ERROR] dependencies are incomplete and INSTALL_MISSING_DEPS=0\n' >&2
    exit 1
  }
  "$REAL_PYTHON" -m pip install --upgrade pip setuptools wheel
  "$REAL_PYTHON" -m pip install \
    numpy pandas scipy scikit-learn joblib baostock tables TA-Lib tensorflow matplotlib psutil
  "$REAL_PYTHON" - <<'PY'
import baostock, joblib, matplotlib, numpy, pandas, scipy, sklearn, tables, talib, tensorflow
print('[OK] Python dependency imports passed after installation')
PY
fi

mkdir -p "$STATE_DIR"
PYTHON_GUARD="$STATE_DIR/python_with_memory_guard"
install -m 700 scripts/as1455_python_memory_guard.sh "$PYTHON_GUARD"

exec env \
  PYTHON_BIN="$PYTHON_GUARD" \
  AS1455_REAL_PYTHON="$REAL_PYTHON" \
  bash scripts/rebuild_ch17_as1455_from_scratch.sh "$MODE"
