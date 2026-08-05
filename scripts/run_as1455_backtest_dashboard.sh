#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv_as1455/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv_as1455/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"
MATRIX_ROOT="${MATRIX_ROOT:-saved_data/ashare_ml4t/ch17_as1455_global_fixed_signal_matrix/refresh_all_v1}"

"$PYTHON_BIN" - <<'PY'
try:
    import streamlit  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "streamlit is not installed; run: "
        ".venv_as1455/bin/pip install -r requirements-dashboard.txt"
    ) from exc
PY

export AS1455_MATRIX_ROOT="$MATRIX_ROOT"
exec "$PYTHON_BIN" -m streamlit run \
  dashboard/as1455_backtest_dashboard.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
