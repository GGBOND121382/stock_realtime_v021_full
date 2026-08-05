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
BASE_URL_PATH="${BASE_URL_PATH:-}"
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

streamlit_args=(
  --server.address "$HOST"
  --server.port "$PORT"
  --server.headless true
  --browser.gatherUsageStats false
)

if [[ -n "$BASE_URL_PATH" ]]; then
  BASE_URL_PATH="${BASE_URL_PATH#/}"
  BASE_URL_PATH="${BASE_URL_PATH%/}"
  [[ -n "$BASE_URL_PATH" ]] || {
    printf 'BASE_URL_PATH must not contain only slashes\n' >&2
    exit 2
  }
  [[ "$BASE_URL_PATH" =~ ^[A-Za-z0-9._/-]+$ ]] || {
    printf 'BASE_URL_PATH contains unsupported characters: %s\n' "$BASE_URL_PATH" >&2
    exit 2
  }
  streamlit_args+=(--server.baseUrlPath "$BASE_URL_PATH")
fi

export AS1455_MATRIX_ROOT="$MATRIX_ROOT"
exec "$PYTHON_BIN" -m streamlit run \
  dashboard/as1455_backtest_dashboard.py \
  "${streamlit_args[@]}"
