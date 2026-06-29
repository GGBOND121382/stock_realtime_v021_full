#!/usr/bin/env bash
set -euo pipefail

# Validate weekly_predictions.h5 has enough predictions for every prediction date.
# Defaults match run_as1455_extend_weekly_empty_v1.sh.

PRED="${PRED:-saved_data/ashare_ml4t/ch17_as1455_weekly_retrain_empty_2026-05-16_to_2026-06-26/00_weekly_retrain/results/weekly_predictions.h5}"
MIN_ROWS_PER_DATE="${MIN_ROWS_PER_DATE:-1000}"
OUT_CSV="${OUT_CSV:-$(dirname "$PRED")/weekly_prediction_coverage_check.csv}"
export PRED MIN_ROWS_PER_DATE OUT_CSV

python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

pred = Path(os.environ["PRED"])
min_rows = int(os.environ.get("MIN_ROWS_PER_DATE", "1000"))
out_csv = Path(os.environ["OUT_CSV"])
if not pred.exists():
    raise FileNotFoundError(pred)

df = pd.read_hdf(pred)
if not isinstance(df.index, pd.MultiIndex):
    raise RuntimeError(f"predictions index is not MultiIndex: {type(df.index)}")
if "date" not in df.index.names:
    raise RuntimeError(f"predictions index names={df.index.names}; missing date")

dates = pd.to_datetime(df.index.get_level_values("date"), errors="coerce").normalize()
counts = dates.value_counts().sort_index().rename_axis("date").reset_index(name="rows")
counts["date"] = pd.to_datetime(counts["date"]).dt.strftime("%Y-%m-%d")
counts["min_rows_per_date"] = min_rows
counts["ok"] = counts["rows"] >= min_rows
out_csv.parent.mkdir(parents=True, exist_ok=True)
counts.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(counts.to_string(index=False))
print(f"[INFO] wrote {out_csv}")
if not counts["ok"].all():
    bad = counts[~counts["ok"]]
    raise SystemExit("prediction coverage failed:\n" + bad.to_string(index=False))
print("[OK] prediction coverage passed")
PY
