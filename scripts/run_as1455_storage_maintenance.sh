#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE="${BASE:-saved_data/ashare_ml4t}"
APPLY="${APPLY:-0}"
KEEP_LIVE_DATES="${KEEP_LIVE_DATES:-3}"
INCLUDE_OBSOLETE="${INCLUDE_OBSOLETE:-1}"
PRUNE_GRID_RUNS="${PRUNE_GRID_RUNS:-1}"
COMPRESS_REPORTS="${COMPRESS_REPORTS:-1}"
COMPRESS_MIN_MB="${COMPRESS_MIN_MB:-20}"
SKIP_FORWARD_ARTIFACTS="${SKIP_FORWARD_ARTIFACTS:-0}"
SKIP_LIVE="${SKIP_LIVE:-0}"
SKIP_PREDICTION_CSV="${SKIP_PREDICTION_CSV:-0}"
ALLOW_ACTIVE_PROCESSES="${ALLOW_ACTIVE_PROCESSES:-0}"
RUN_FULL_CHECKS="${RUN_FULL_CHECKS:-0}"
TOP_FILES="${TOP_FILES:-80}"
DU_DEPTH="${DU_DEPTH:-2}"
DU_LINES="${DU_LINES:-160}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$BASE/storage_maintenance_$RUN_STAMP}"

for name in \
  APPLY INCLUDE_OBSOLETE PRUNE_GRID_RUNS COMPRESS_REPORTS \
  SKIP_FORWARD_ARTIFACTS SKIP_LIVE SKIP_PREDICTION_CSV \
  ALLOW_ACTIVE_PROCESSES RUN_FULL_CHECKS; do
  value="${!name}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "[ERROR] $name must be 0 or 1, got: $value" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
CONSOLE_LOG="$OUT_DIR/console.log"
CONFIG_FILE="$OUT_DIR/run_config.env"
DIAG_BEFORE="$OUT_DIR/diagnostics_before.txt"
DIAG_AFTER="$OUT_DIR/diagnostics_after.txt"
DRY_MANIFEST="$OUT_DIR/cleanup_dry_run.json"
APPLY_MANIFEST="$OUT_DIR/cleanup_apply.json"
SHARE_FILE="$OUT_DIR/share_me.txt"

exec > >(tee -a "$CONSOLE_LOG") 2>&1

cat >"$CONFIG_FILE" <<EOF
BASE=$BASE
APPLY=$APPLY
KEEP_LIVE_DATES=$KEEP_LIVE_DATES
INCLUDE_OBSOLETE=$INCLUDE_OBSOLETE
PRUNE_GRID_RUNS=$PRUNE_GRID_RUNS
COMPRESS_REPORTS=$COMPRESS_REPORTS
COMPRESS_MIN_MB=$COMPRESS_MIN_MB
SKIP_FORWARD_ARTIFACTS=$SKIP_FORWARD_ARTIFACTS
SKIP_LIVE=$SKIP_LIVE
SKIP_PREDICTION_CSV=$SKIP_PREDICTION_CSV
ALLOW_ACTIVE_PROCESSES=$ALLOW_ACTIVE_PROCESSES
RUN_FULL_CHECKS=$RUN_FULL_CHECKS
TOP_FILES=$TOP_FILES
DU_DEPTH=$DU_DEPTH
DU_LINES=$DU_LINES
RUN_STAMP=$RUN_STAMP
OUT_DIR=$OUT_DIR
EOF

echo "===== AS1455 storage maintenance ====="
echo "mode=$([[ "$APPLY" == "1" ]] && echo apply || echo audit-only)"
echo "base=$BASE"
echo "out_dir=$OUT_DIR"
echo "config=$CONFIG_FILE"

echo "===== Preflight syntax and retention checks ====="
"$PYTHON_BIN" -m compileall -q \
  scripts/check_as1455_disk_space.py \
  scripts/cleanup_as1455_storage.py \
  scripts/export_as1455_storage_diagnostics.py \
  scripts/check_as1455_artifact_retention.py \
  utils/as1455_artifact_retention.py
bash -n scripts/run_as1455_storage_maintenance.sh
"$PYTHON_BIN" scripts/check_as1455_artifact_retention.py

if [[ "$RUN_FULL_CHECKS" == "1" ]]; then
  echo "===== Full AS1455 refactor checks ====="
  bash scripts/check_ch17_as1455_refactor.sh
fi

echo "===== Diagnostics before cleanup ====="
"$PYTHON_BIN" scripts/export_as1455_storage_diagnostics.py \
  --base "$BASE" \
  --out "$DIAG_BEFORE" \
  --top-files "$TOP_FILES" \
  --du-depth "$DU_DEPTH" \
  --du-lines "$DU_LINES"

cleanup_common=(
  scripts/cleanup_as1455_storage.py
  --base "$BASE"
  --keep-live-dates "$KEEP_LIVE_DATES"
  --compress-min-mb "$COMPRESS_MIN_MB"
)
[[ "$INCLUDE_OBSOLETE" == "1" ]] && cleanup_common+=(--include-obsolete)
[[ "$PRUNE_GRID_RUNS" == "1" ]] && cleanup_common+=(--prune-grid-runs)
[[ "$COMPRESS_REPORTS" == "1" ]] && cleanup_common+=(--compress-reports)
[[ "$SKIP_FORWARD_ARTIFACTS" == "1" ]] && cleanup_common+=(--skip-forward-artifacts)
[[ "$SKIP_LIVE" == "1" ]] && cleanup_common+=(--skip-live)
[[ "$SKIP_PREDICTION_CSV" == "1" ]] && cleanup_common+=(--skip-prediction-csv)
[[ "$ALLOW_ACTIVE_PROCESSES" == "1" ]] && cleanup_common+=(--allow-active-processes)

echo "===== Cleanup dry-run ====="
"$PYTHON_BIN" "${cleanup_common[@]}" --manifest "$DRY_MANIFEST"

if [[ "$APPLY" == "1" ]]; then
  echo "===== Cleanup apply ====="
  "$PYTHON_BIN" "${cleanup_common[@]}" --apply --manifest "$APPLY_MANIFEST"

  echo "===== Diagnostics after cleanup ====="
  "$PYTHON_BIN" scripts/export_as1455_storage_diagnostics.py \
    --base "$BASE" \
    --out "$DIAG_AFTER" \
    --top-files "$TOP_FILES" \
    --du-depth "$DU_DEPTH" \
    --du-lines "$DU_LINES"
else
  echo "[SAFE] audit-only completed; no files were deleted"
  echo "[NEXT] review $DRY_MANIFEST, then run:"
  echo "       APPLY=1 bash scripts/run_as1455_storage_maintenance.sh"
fi

{
  echo "AS1455 STORAGE MAINTENANCE SHARE FILE"
  echo "generated_at=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "mode=$([[ "$APPLY" == "1" ]] && echo apply || echo audit-only)"
  echo "project=$(pwd)"
  echo "base=$BASE"
  echo "out_dir=$OUT_DIR"
  echo
  echo "===== RUN CONFIG ====="
  cat "$CONFIG_FILE"
  echo
  echo "===== DIAGNOSTICS BEFORE ====="
  cat "$DIAG_BEFORE"
  echo
  echo "===== CLEANUP DRY-RUN MANIFEST ====="
  "$PYTHON_BIN" -m json.tool "$DRY_MANIFEST"
  if [[ "$APPLY" == "1" ]]; then
    echo
    echo "===== CLEANUP APPLY MANIFEST ====="
    "$PYTHON_BIN" -m json.tool "$APPLY_MANIFEST"
    echo
    echo "===== DIAGNOSTICS AFTER ====="
    cat "$DIAG_AFTER"
  fi
  echo
  echo "===== CONSOLE LOG TAIL ====="
  tail -n 250 "$CONSOLE_LOG"
} >"$SHARE_FILE"

echo "===== Completed ====="
echo "console_log=$CONSOLE_LOG"
echo "dry_run_manifest=$DRY_MANIFEST"
[[ "$APPLY" == "1" ]] && echo "apply_manifest=$APPLY_MANIFEST"
echo "share_file=$SHARE_FILE"
echo
if [[ "$APPLY" == "1" ]]; then
  echo "[DONE] cleanup applied and post-cleanup diagnostics exported"
else
  echo "[DONE] audit-only; copy share_file if review is needed"
fi
