from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path("saved_data/603308_pipeline_out")
FILES = [
    ("01_samples", ROOT / "01_samples" / "training_samples.csv"),
    ("01_samples_asof1455", ROOT / "01_samples_asof1455" / "training_samples_asof1455.csv"),
    ("02_fundamental", ROOT / "02_fundamental" / "training_samples_with_fundamentals.csv"),
    ("03_sector", ROOT / "03_sector" / "training_samples_with_sector.csv"),
    ("04_external_aero", ROOT / "04_external" / "aero_nuclear_equipment" / "training_samples_with_aero_nuclear_equipment_external.csv"),
]

REQUIRED_ASOF = [
    "open_asof1455",
    "high_asof1455",
    "low_asof1455",
    "close_asof1455",
    "volume_asof1455",
    "amount_asof1455",
    "vwap_asof1455",
]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def summarize_file(label: str, path: Path) -> None:
    print(f"\n## {label}")
    print(path)
    if not path.exists():
        print("exists=False")
        return
    print(f"mtime={pd.Timestamp(path.stat().st_mtime, unit='s')}")
    df = load_csv(path)
    print(f"rows={len(df)} cols={len(df.columns)}")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        print(f"date_min={dates.min().date() if dates.notna().any() else None}")
        print(f"date_max={dates.max().date() if dates.notna().any() else None}")
    print("required_asof_present=" + ",".join([c for c in REQUIRED_ASOF if c in df.columns]))
    print("required_asof_missing=" + ",".join([c for c in REQUIRED_ASOF if c not in df.columns]))
    if "feature_time_mode" in df.columns:
        print("feature_time_mode_counts=" + str(df["feature_time_mode"].value_counts(dropna=False).to_dict()))
    if "valuation_time_mode" in df.columns:
        print("valuation_time_mode_counts=" + str(df["valuation_time_mode"].value_counts(dropna=False).to_dict()))
    if "next_day_close" in df.columns:
        print(f"next_day_close_nan={int(df['next_day_close'].isna().sum())}")

    miss = df.isna().mean().sort_values(ascending=False)
    print(f"cols_missing_gt_70={int((miss > 0.70).sum())}")
    print(f"cols_missing_gt_30={int((miss > 0.30).sum())}")
    top = miss[miss > 0].head(20)
    for col, ratio in top.items():
        nonnull = int(df[col].notna().sum())
        first = ""
        last = ""
        if "date" in df.columns and nonnull:
            valid_dates = pd.to_datetime(df.loc[df[col].notna(), "date"], errors="coerce").dropna()
            if not valid_dates.empty:
                first = str(valid_dates.min().date())
                last = str(valid_dates.max().date())
        print(f"missing {col}: ratio={ratio:.3f} nonnull={nonnull} first={first} last={last}")


def column_delta() -> None:
    print("\n## Column lineage deltas")
    loaded = {}
    for label, path in FILES:
        if path.exists():
            loaded[label] = load_csv(path)
    pairs = [
        ("01_samples", "02_fundamental"),
        ("01_samples_asof1455", "02_fundamental"),
        ("02_fundamental", "03_sector"),
        ("03_sector", "04_external_aero"),
    ]
    for a, b in pairs:
        if a not in loaded or b not in loaded:
            continue
        ca = set(loaded[a].columns)
        cb = set(loaded[b].columns)
        added = sorted(cb - ca)
        dropped = sorted(ca - cb)
        print(f"{a} -> {b}: added={len(added)} dropped={len(dropped)}")
        print("  added_head=" + ",".join(added[:30]))
        print("  dropped_head=" + ",".join(dropped[:30]))


def print_json(path: Path) -> None:
    print(f"\n## JSON {path}")
    if not path.exists():
        print("exists=False")
        return
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    print(json.dumps(obj, ensure_ascii=False, indent=2)[:6000])


def print_pipeline_stages(path: Path) -> None:
    print(f"\n## Pipeline stages {path}")
    if not path.exists():
        print("exists=False")
        return
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    print(f"feature_time_mode={obj.get('feature_time_mode')}")
    print(f"final_samples={obj.get('final_samples')}")
    for stage in obj.get("stages", []):
        name = stage.get("name")
        status = stage.get("status")
        command = stage.get("command") or []
        print(f"\nstage={name} status={status}")
        if command:
            print("command=" + " ".join(map(str, command)))
            for flag in [
                "--samples",
                "--daily-samples",
                "--out-dir",
                "--profile",
                "--feature-time-mode",
                "--cutoff-time",
            ]:
                if flag in command:
                    idx = command.index(flag)
                    val = command[idx + 1] if idx + 1 < len(command) else ""
                    print(f"  {flag} {val}")
        if stage.get("outputs"):
            print("outputs=" + json.dumps(stage["outputs"], ensure_ascii=False))


def print_validation_reports() -> None:
    print("\n## Validation reports")
    for path in [
        ROOT / "01_samples" / "validation_report.json",
        ROOT / "01_samples_asof1455" / "validation_report.json",
        ROOT / "02_fundamental" / "validation_report.json",
        ROOT / "03_sector" / "validation_report.json",
        ROOT / "04_external" / "aero_nuclear_equipment" / "validation_report.json",
    ]:
        print(f"\n{path}")
        if not path.exists():
            print("exists=False")
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        keys = [
            "status",
            "rows",
            "cols",
            "date_min",
            "date_max",
            "missing_required_columns",
            "high_missing_columns",
            "max_missing_ratio",
        ]
        compact = {k: obj.get(k) for k in keys if k in obj}
        print(json.dumps(compact if compact else obj, ensure_ascii=False, indent=2)[:3000])


def list_stage_files() -> None:
    print("\n## Stage files")
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".log", ".txt"}:
            rel = path.relative_to(ROOT)
            size = path.stat().st_size
            mtime = pd.Timestamp(path.stat().st_mtime, unit="s")
            print(f"{rel} size={size} mtime={mtime}")


def main() -> None:
    if os.environ.get("CONCISE") == "1":
        print("## Concise file mtimes and asof compatibility")
        for label, path in FILES + [("pipeline_summary", ROOT / "pipeline_summary.json"), ("run_log", ROOT / "run.log")]:
            print(f"\n{label}: {path}")
            if not path.exists():
                print("exists=False")
                continue
            print(f"mtime={pd.Timestamp(path.stat().st_mtime, unit='s')}")
            if path.suffix.lower() == ".csv":
                df = load_csv(path)
                print(f"rows={len(df)} cols={len(df.columns)}")
                print("required_asof_missing=" + ",".join([c for c in REQUIRED_ASOF if c not in df.columns]))
                if "feature_time_mode" in df.columns:
                    print("feature_time_mode_counts=" + str(df["feature_time_mode"].value_counts(dropna=False).to_dict()))
        print_pipeline_stages(ROOT / "pipeline_summary.json")
        print("\n## External raw cache date ranges")
        for path in sorted((ROOT / "04_external" / "aero_nuclear_equipment" / "baostock_external_cache").rglob("*_daily_raw.csv")):
            df = load_csv(path)
            date_col = "date" if "date" in df.columns else "交易日期" if "交易日期" in df.columns else None
            dates = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.Series(dtype="datetime64[ns]")
            print(
                f"{path.relative_to(ROOT)} rows={len(df)} "
                f"date_min={dates.min().date() if dates.notna().any() else None} "
                f"date_max={dates.max().date() if dates.notna().any() else None}"
            )
        return

    list_stage_files()
    for label, path in FILES:
        summarize_file(label, path)
    column_delta()
    print_pipeline_stages(ROOT / "pipeline_summary.json")
    print_validation_reports()
    for path in [
        ROOT / "pipeline_summary.json",
        ROOT / "02_fundamental" / "build_summary.json",
        ROOT / "03_sector" / "build_summary.json",
        ROOT / "04_external" / "aero_nuclear_equipment" / "build_summary.json",
    ]:
        print_json(path)


if __name__ == "__main__":
    main()
