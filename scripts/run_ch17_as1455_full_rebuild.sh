#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-all}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv_as1455}"
REBUILD_ROOT="${REBUILD_ROOT:-saved_data/ashare_ml4t/rebuild_ch17_as1455}"
STATE_DIR="${STATE_DIR:-$REBUILD_ROOT/state}"
FEATURE_PRESETS="${FEATURE_PRESETS:-rotation_onehot rotation_addon_onehot}"
TARGETS="${TARGETS:-r01_fwd r05_fwd r21_fwd}"
CPU_THREADS="${CPU_THREADS:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$CPU_THREADS}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

mkdir -p "$STATE_DIR"
if [[ -s "$STATE_DIR/run_stamp.txt" ]]; then
  RUN_STAMP="$(tr -d '\r\n' < "$STATE_DIR/run_stamp.txt")"
else
  RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
  printf '%s\n' "$RUN_STAMP" > "$STATE_DIR/run_stamp.txt"
fi

rebalance_for_target() {
  case "$1" in
    r01_fwd) printf '1\n' ;;
    r05_fwd) printf '5\n' ;;
    r21_fwd) printf '21\n' ;;
    *) printf '[ERROR] unsupported target: %s\n' "$1" >&2; exit 2 ;;
  esac
}

# The strict-OOS implementation writes the manifest inside the grid directory.
# Keep the existing implementation unchanged and expose a stable root-level link
# for the final audit. No existing file is replaced or deleted.
for target in $TARGETS; do
  rebalance="$(rebalance_for_target "$target")"
  for preset in $FEATURE_PRESETS; do
    out_root="saved_data/ashare_ml4t/ch17_as1455_fold0_forward_backtest/${preset}_${target}_reb${rebalance}_${RUN_STAMP}"
    mkdir -p "$out_root"
    link="$out_root/strict_oos_manifest.json"
    if [[ ! -e "$link" && ! -L "$link" ]]; then
      ln -s '01_close_auction_grid/strict_oos_manifest.json' "$link"
    fi
  done
done

case "$MODE" in
  status)
    exec env BASE_PYTHON="$BASE_PYTHON" USE_VENV="$USE_VENV" VENV_DIR="$VENV_DIR" \
      bash scripts/rebuild_ch17_as1455_from_scratch.sh status
    ;;
  preflight)
    exec env BASE_PYTHON="$BASE_PYTHON" USE_VENV="$USE_VENV" VENV_DIR="$VENV_DIR" \
      bash scripts/rebuild_ch17_as1455_from_scratch.sh preflight
    ;;
  all|history|model_data|selfcheck|training|historical|forward|audit)
    # Build/reuse the real Python environment with the existing preflight first.
    env BASE_PYTHON="$BASE_PYTHON" USE_VENV="$USE_VENV" VENV_DIR="$VENV_DIR" \
      bash scripts/rebuild_ch17_as1455_from_scratch.sh preflight
    if [[ "$USE_VENV" == "1" ]]; then
      REAL_PYTHON="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")/bin/python"
    else
      REAL_PYTHON="$(command -v "$BASE_PYTHON")"
    fi
    [[ -x "$REAL_PYTHON" ]] || {
      printf '[ERROR] real Python is unavailable: %s\n' "$REAL_PYTHON" >&2
      exit 1
    }

    # The controller expects PY to be one executable path. Install an executable
    # copy of the repository memory guard into the rebuild state directory.
    PYTHON_GUARD="$STATE_DIR/python_with_memory_guard"
    install -m 700 scripts/as1455_python_memory_guard.sh "$PYTHON_GUARD"

    exec env \
      BASE_PYTHON="$PYTHON_GUARD" \
      USE_VENV=0 \
      AS1455_REAL_PYTHON="$REAL_PYTHON" \
      VENV_DIR="$VENV_DIR" \
      bash scripts/rebuild_ch17_as1455_from_scratch.sh "$MODE"
    ;;
  *)
    printf '[ERROR] unsupported mode: %s\n' "$MODE" >&2
    exit 2
    ;;
esac
