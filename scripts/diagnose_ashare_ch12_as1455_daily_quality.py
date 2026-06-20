#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_ashare_ch12_as1455_model_data import (
    DEFAULT_OUT_DIR,
    DEFAULT_QFQ_DAILY_CACHE,
    write_as1455_vs_daily_quality_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild as1455 versus daily quality reports from the existing adjusted HDF")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--qfq-daily-cache-dir", default=str(DEFAULT_QFQ_DAILY_CACHE))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    adjusted_path = out_dir / "as1455_ohlcv_adj.h5"
    if not adjusted_path.exists():
        raise SystemExit(f"adjusted HDF not found: {adjusted_path}")
    adj = pd.read_hdf(adjusted_path, "ohlcv")
    write_as1455_vs_daily_quality_report(adj, Path(args.qfq_daily_cache_dir), out_dir / "reports")
    print(out_dir / "reports" / "as1455_vs_daily_integrity_violations.csv")


if __name__ == "__main__":
    main()
