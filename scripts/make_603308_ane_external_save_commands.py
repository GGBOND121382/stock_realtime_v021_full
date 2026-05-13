#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import re
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--leaderboard", required=True)
p.add_argument("--samples", required=True)
p.add_argument("--intraday-bars", required=True)
p.add_argument("--stock-code", default="603308.SH")
p.add_argument("--out", required=True)
p.add_argument("--top", type=int, default=5)
args = p.parse_args()

df = pd.read_csv(args.leaderboard)
if df.empty:
    raise SystemExit("empty leaderboard")

# Sort defensively if summarizer did not preserve order.
sort_cols = [c for c in ["compound_return", "avg_return", "profit_factor", "win_rate", "trades"] if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

def clean(x):
    s = str(x)
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    return s.strip("_").lower() or "x"

lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "PYTHON=\"${PYTHON:-python3}\"",
    "",
    "# Auto-generated. Review before running if needed.",
    "",
]

seen = set()
count = 0
for _, r in df.iterrows():
    if count >= args.top:
        break
    entry = str(r.get("entry_policy", "unknown"))
    label = str(r.get("label_mode", "unknown"))
    if label == "unknown" and "target_hit_bps" in r and float(r.get("target_hit_bps") or 0) >= 80:
        label = "hit"
    target = int(float(r.get("target_hit_bps", 50))) if pd.notna(r.get("target_hit_bps", 50)) else 50
    fg = str(r.get("feature_group", ""))
    model = str(r.get("model_name", ""))
    if not fg or not model or entry == "unknown" or label == "unknown":
        continue
    key = (entry, label, target, fg, model)
    if key in seen:
        continue
    seen.add(key)
    count += 1
    artifact = f"nextday_{clean(entry)}_{clean(label)}_{target}bps_{clean(model)}_{clean(fg)}_ane_live_board_v2"
    lines += [
        f"echo '[SAVE {count}] {artifact}'",
        "\"$PYTHON\" model_saving/save_nextday_model.py \\",
        f"  --stock-code {args.stock_code} \\",
        f"  --artifact-name {artifact} \\",
        f"  --samples {args.samples} \\",
        f"  --intraday-bars {args.intraday_bars} \\",
        "  --out-dir saved_models \\",
        f"  --feature-group {fg} \\",
        f"  --model-name {model} \\",
        f"  --label-mode {label} \\",
        f"  --entry-policy {entry} \\",
        "  --entry-vwap-premium-bps 50 \\",
        f"  --target-hit-bps {target} \\",
        "  --round-trip-cost-bps 1.7 \\",
        "  --valid-rows 252 \\",
        "  --min-train-entries 80 \\",
        "  --min-valid-trades 8 \\",
        "  --quantiles 0.5,0.6,0.7,0.8",
        "",
    ]

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out}; commands={count}")
