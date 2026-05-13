#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--leaderboard", required=True)
p.add_argument("--top", type=int, default=30)
args = p.parse_args()

path = Path(args.leaderboard)
if not path.exists():
    raise SystemExit(f"missing leaderboard: {path}")

df = pd.read_csv(path)
cols = [c for c in [
    "entry_policy", "label_mode", "target_hit_bps", "feature_group", "model_name",
    "trades", "win_rate", "avg_return", "median_return", "compound_return",
    "max_drawdown", "profit_factor", "windows", "rank_score"
] if c in df.columns]

print("\n=== TOP CANDIDATES ===")
print(df[cols].head(args.top).to_string(index=False))

print("\n=== BEST BY feature_group ===")
if "feature_group" in df.columns:
    x = df.sort_values([c for c in ["compound_return", "avg_return", "profit_factor"] if c in df.columns], ascending=False)
    print(x.groupby("feature_group", dropna=False).head(3)[cols].to_string(index=False))
