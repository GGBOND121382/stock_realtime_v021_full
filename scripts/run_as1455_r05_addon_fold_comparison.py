#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


helpers = load_module(
    "as1455_independent_fold_helpers",
    PROJECT_DIR / "scripts" / "run_as1455_independent_fold_backtests.py",
)
bt = helpers.bt

TARGET = "r05_fwd"
PRESET = "rotation_addon_onehot"
EXPECTED_TARGET_FOLDS = tuple(range(5, -1, -1))
FREQUENCIES = ("daily", "weekly", "monthly")
COPY_FILES = (
    "close_auction_nav.csv",
    "daily_drawdown.csv",
    "monthly_summary.csv",
    "yearly_summary.csv",
    "fee_summary.csv",
    "turnover_summary.csv",
    "summary.json",
    "config.json",
    "close_auction_summary.json",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | None, *, required: bool, label: str) -> Path | None:
    if not value:
        if required:
            raise RuntimeError(f"missing required path: {label}")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path = path.resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    if not required and not path.exists():
        raise FileNotFoundError(
            f"recorded optional input is missing for reproducibility: {label}: {path}"
        )
    return path


def choose_path(
    override: str | None,
    recorded: Any,
    *,
    required: bool,
    label: str,
) -> Path | None:
    value = override if override not in (None, "") else recorded
    return resolve_path(
        str(value) if value not in (None, "") else None,
        required=required,
        label=label,
    )


def load_single_pair(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 1:
        raise RuntimeError(
            f"expected exactly one result pair in {path}, got "
            f"{type(pairs).__name__}/{len(pairs) if isinstance(pairs, list) else 'n/a'}"
        )
    pair = pairs[0]
    if pair.get("target_col") != TARGET or pair.get("feature_preset") != PRESET:
        raise RuntimeError(
            f"pair mismatch: expected target={TARGET} preset={PRESET}, "
            f"actual target={pair.get('target_col')} preset={pair.get('feature_preset')}"
        )
    return pair


def mapping_by_target_fold(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        target_fold = int(row.get("target_fold", int(row["source_fold"]) - 1))
        source_fold = int(row["source_fold"])
        start = pd.Timestamp(row["start"]).normalize()
        end = pd.Timestamp(row["end"]).normalize()
        if start > end:
            raise RuntimeError(f"target_fold{target_fold} mapping reversed: {start} > {end}")
        result[target_fold] = {
            **row,
            "target_fold": target_fold,
            "source_fold": source_fold,
            "start_ts": start,
            "end_ts": end,
        }
    if set(result) != set(EXPECTED_TARGET_FOLDS):
        raise RuntimeError(
            f"target fold mapping mismatch: expected={list(EXPECTED_TARGET_FOLDS)} "
            f"actual={sorted(result, reverse=True)}"
        )
    return result


def normalize_dates(values: Any) -> pd.DatetimeIndex:
    return helpers.normalize_dates(values)


def scalar_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if value is None or isinstance(value, (str, int, float, bool, pd.Timestamp))
    }


def load_materialized_continuous_run(
    pair: dict[str, Any],
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    historical = pair["historical"]
    source_nav = Path(historical["nav_file"]).resolve()
    source_dir = source_nav.parent
    config = read_json(Path(historical["config_file"]).resolve())
    nav = pd.read_csv(source_nav)
    if "date" not in nav.columns or "nav" not in nav.columns:
        raise RuntimeError(f"materialized NAV lacks date/nav columns: {source_nav}")
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = (
        nav.dropna(subset=["date", "nav"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    if len(nav) < 2:
        raise RuntimeError(f"materialized NAV has fewer than two rows: {source_nav}")
    return source_dir, config, nav


def copy_materialized_run(source_dir: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in COPY_FILES:
        source = source_dir / name
        if source.is_file() and source.stat().st_size > 0:
            shutil.copy2(source, destination / name)
            copied.append(name)
    if (
        "close_auction_nav.csv" not in copied
        or "config.json" not in copied
        or "summary.json" not in copied
    ):
        raise RuntimeError(f"materialized run is incomplete: {source_dir}")
    return copied


def build_boundary_audit(
    mapping: dict[int, dict[str, Any]],
    prediction_dates: pd.DatetimeIndex,
    execution_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[int, pd.DatetimeIndex]]:
    rows: list[dict[str, Any]] = []
    fold_dates: dict[int, pd.DatetimeIndex] = {}
    previous_end: pd.Timestamp | None = None

    for target_fold in EXPECTED_TARGET_FOLDS:
        item = mapping[target_fold]
        start = item["start_ts"]
        end = item["end_ts"]
        pred = prediction_dates[(prediction_dates >= start) & (prediction_dates <= end)]
        exec_dates = execution_dates[(execution_dates >= start) & (execution_dates <= end)]
        if len(pred) < 2:
            raise RuntimeError(f"target_fold{target_fold} has fewer than two prediction dates")
        if not pred.equals(exec_dates):
            missing_prediction = exec_dates.difference(pred)
            missing_execution = pred.difference(exec_dates)
            raise RuntimeError(
                f"target_fold{target_fold} prediction/execution calendar mismatch; "
                f"missing_prediction={list(map(str, missing_prediction[:10]))} "
                f"missing_execution={list(map(str, missing_execution[:10]))}"
            )

        actual_start = pd.Timestamp(pred[0]).normalize()
        actual_end = pd.Timestamp(pred[-1]).normalize()
        trading_gap_days = 0
        calendar_gap_days = 0
        if previous_end is not None:
            if actual_start <= previous_end:
                raise RuntimeError(
                    f"target_fold{target_fold} overlaps or reverses previous fold: "
                    f"previous_end={previous_end:%Y-%m-%d} "
                    f"start={actual_start:%Y-%m-%d}"
                )
            bridge = execution_dates[
                (execution_dates > previous_end) & (execution_dates < actual_start)
            ]
            trading_gap_days = int(len(bridge))
            calendar_gap_days = max(0, int((actual_start - previous_end).days - 1))

        rows.append(
            {
                "target_fold": target_fold,
                "source_model_fold": int(item["source_fold"]),
                "manifest_start": start.strftime("%Y-%m-%d"),
                "manifest_end": end.strftime("%Y-%m-%d"),
                "actual_start": actual_start.strftime("%Y-%m-%d"),
                "actual_end": actual_end.strftime("%Y-%m-%d"),
                "prediction_days": int(len(pred)),
                "previous_end": (
                    previous_end.strftime("%Y-%m-%d")
                    if previous_end is not None
                    else None
                ),
                "calendar_gap_days": calendar_gap_days,
                "trading_gap_days": trading_gap_days,
            }
        )
        if trading_gap_days != 0:
            raise RuntimeError(
                f"target_fold{target_fold} is not contiguous with the previous fold: "
                f"trading_gap_days={trading_gap_days}"
            )
        fold_dates[target_fold] = pred
        previous_end = actual_end

    return pd.DataFrame(rows), fold_dates


def curve_from_nav(nav: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    curve = nav[["date", "nav"]].copy()
    curve["return_pct"] = (curve["nav"] / float(initial_cash) - 1.0) * 100.0
    return curve


def sample_curve(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    return helpers.sample_curve(frame, frequency)


def plot_curves(curves: list[dict[str, Any]], out_dir: Path, title_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for frequency in FREQUENCIES:
        exported = helpers.plot_frequency(
            curves=curves,
            frequency=frequency,
            out_file=out_dir / f"return_curve_{frequency}.png",
            title=f"{title_prefix} ({frequency})",
            sample_curve=sample_curve,
            plt=plt,
        )
        exported.to_csv(
            out_dir / f"return_curve_{frequency}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def segment_metrics(
    nav: pd.DataFrame,
    fold_dates: dict[int, pd.DatetimeIndex],
) -> pd.DataFrame:
    nav = nav.copy().sort_values("date").reset_index(drop=True)
    nav["daily_return"] = pd.to_numeric(nav.get("daily_return"), errors="coerce")
    rows: list[dict[str, Any]] = []
    for target_fold in EXPECTED_TARGET_FOLDS:
        dates = fold_dates[target_fold]
        segment = nav[nav["date"].isin(dates)].copy()
        if len(segment) != len(dates):
            raise RuntimeError(
                f"continuous NAV coverage mismatch for target_fold{target_fold}: "
                f"expected={len(dates)} actual={len(segment)}"
            )
        returns = pd.to_numeric(segment["daily_return"], errors="coerce").fillna(0.0)
        segment_return = float((1.0 + returns).prod() - 1.0)
        first_index = int(segment.index.min())
        start_nav_before = (
            float(nav.loc[first_index - 1, "nav"])
            if first_index > 0
            else float(
                segment["nav"].iloc[0] / max(1.0 + returns.iloc[0], 1e-12)
            )
        )
        rows.append(
            {
                "target_fold": target_fold,
                "start_date": pd.Timestamp(segment["date"].iloc[0]).strftime(
                    "%Y-%m-%d"
                ),
                "end_date": pd.Timestamp(segment["date"].iloc[-1]).strftime(
                    "%Y-%m-%d"
                ),
                "n_days": int(len(segment)),
                "start_nav_before_segment": start_nav_before,
                "end_nav": float(segment["nav"].iloc[-1]),
                "segment_return": segment_return,
                "end_positions": (
                    int(segment["n_positions"].iloc[-1])
                    if "n_positions" in segment.columns
                    else None
                ),
                "end_cash": (
                    float(segment["cash"].iloc[-1])
                    if "cash" in segment.columns
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create r05_fwd rotation_addon_onehot independent per-fold backtests "
            "and expose the authoritative materialized continuous cross-fold result."
        )
    )
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--initial-cash", type=float, default=200000.0)
    parser.add_argument("--output-mode", choices=["compact", "full"], default="compact")
    parser.add_argument("--raw-daily-cache-dir", default=None)
    parser.add_argument("--raw-5m-cache-dir", default=None)
    parser.add_argument("--last5-panel", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--st-symbols", default=None)
    parser.add_argument("--st-status", default=None)
    parser.add_argument("--corporate-actions", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.initial_cash <= 0:
        raise SystemExit("--initial-cash must be positive")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pair_path = Path(args.pair_manifest).expanduser().resolve()
    pair = load_single_pair(pair_path)
    historical = pair["historical"]
    mapping = mapping_by_target_fold(historical["fold_mapping"])

    source_run_dir, stored_config, materialized_nav = load_materialized_continuous_run(
        pair
    )
    source_meta_path = source_run_dir / "close_auction_summary.json"
    source_meta = read_json(source_meta_path) if source_meta_path.is_file() else {}

    raw_daily = choose_path(
        args.raw_daily_cache_dir,
        source_meta.get("raw_daily_cache_dir"),
        required=True,
        label="raw daily cache",
    )
    raw_5m = choose_path(
        args.raw_5m_cache_dir,
        source_meta.get("raw_5m_cache_dir"),
        required=False,
        label="raw 5m cache",
    )
    last5_panel_path = choose_path(
        args.last5_panel,
        source_meta.get("last5_panel"),
        required=False,
        label="last5 panel",
    )
    universe_path = choose_path(
        args.universe,
        source_meta.get("universe"),
        required=False,
        label="universe",
    )
    st_symbols_path = choose_path(
        args.st_symbols,
        source_meta.get("st_symbols"),
        required=False,
        label="static ST symbols",
    )
    st_status_path = choose_path(
        args.st_status,
        source_meta.get("st_status"),
        required=False,
        label="historical ST status",
    )
    corporate_actions_path = choose_path(
        args.corporate_actions,
        source_meta.get("corporate_actions"),
        required=False,
        label="corporate actions",
    )

    selection = historical["selection"]
    predictions = helpers.load_selected_predictions(
        Path(historical["prediction_file"]).resolve(), selection
    )
    prediction_dates = normalize_dates(predictions["date"])
    symbols = sorted(predictions["symbol"].astype(str).unique())

    universe = bt.read_universe(universe_path)
    st_symbols = bt.load_st_symbols(st_symbols_path)
    st_status = bt.load_st_status(st_status_path)
    last5_panel = bt.load_last5_panel(last5_panel_path)
    corporate_actions = bt.load_corporate_actions(corporate_actions_path)

    capacity_mode = str(stored_config["capacity_mode"])
    if capacity_mode != "none" and raw_5m is None and last5_panel.empty:
        raise RuntimeError(
            f"frozen capacity_mode={capacity_mode} requires raw 5m data or a last5 panel"
        )

    execution_panel, execution_report = bt.build_execution_panel(
        symbols,
        raw_daily,
        universe,
        st_symbols,
        st_status=st_status,
        last5_panel=last5_panel,
        raw_5m_cache_dir=raw_5m,
    )
    if execution_panel.empty:
        raise RuntimeError("execution panel is empty")
    execution_report.to_csv(
        out_root / "execution_data_report.csv", index=False, encoding="utf-8-sig"
    )
    execution_dates = normalize_dates(execution_panel["date"])

    audit, fold_dates = build_boundary_audit(
        mapping, prediction_dates, execution_dates
    )
    audit.to_csv(
        out_root / "fold_boundary_audit.csv", index=False, encoding="utf-8-sig"
    )

    expected_continuous_dates = normalize_dates(
        [date for target_fold in EXPECTED_TARGET_FOLDS for date in fold_dates[target_fold]]
    )
    materialized_dates = normalize_dates(materialized_nav["date"])
    if not expected_continuous_dates.equals(materialized_dates):
        missing_in_materialized = expected_continuous_dates.difference(
            materialized_dates
        )
        extra_in_materialized = materialized_dates.difference(
            expected_continuous_dates
        )
        raise RuntimeError(
            "materialized continuous NAV does not match the six target-fold calendars; "
            f"missing={list(map(str, missing_in_materialized[:10]))} "
            f"extra={list(map(str, extra_in_materialized[:10]))}"
        )

    per_fold_root = out_root / "per_fold"
    per_fold_curves: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    original_offset = int(stored_config["rebalance_offset"])
    full_dates = normalize_dates(predictions["date"])

    for target_fold in EXPECTED_TARGET_FOLDS:
        dates = fold_dates[target_fold]
        mapping_row = mapping[target_fold]
        effective_offset, skipped_dates = helpers.effective_offset_for_crop(
            full_dates=full_dates,
            crop_dates=dates,
            original_offset=original_offset,
            rebalance_every=int(stored_config["rebalance_every"]),
        )
        cfg = helpers.config_to_trade_config(
            stored_config,
            initial_cash=args.initial_cash,
            effective_offset=effective_offset,
        )
        prediction_slice = predictions[predictions["date"].isin(dates)].copy()
        execution_slice = execution_panel[
            execution_panel["date"].isin(dates)
        ].copy()
        precheck = helpers.capacity_precheck(
            execution_slice,
            prediction_slice,
            str(cfg.capacity_mode),
        )
        result = bt.backtest(
            prediction_slice,
            execution_slice,
            cfg,
            corporate_actions=corporate_actions,
        )
        run_dir = per_fold_root / f"target_fold{target_fold}"
        manifest = {
            "result_type": "independent_fold",
            "target_col": TARGET,
            "feature_preset": PRESET,
            "target_fold": target_fold,
            "source_model_fold": int(mapping_row["source_fold"]),
            "source_prediction_file": historical["prediction_file"],
            "source_config_file": historical["config_file"],
            "source_materialized_run": str(source_run_dir),
            "signal": selection,
            "initial_state": "empty_positions_and_initial_cash",
            "initial_cash": float(args.initial_cash),
            "start_date": pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
            "trading_days": int(len(dates)),
            "original_rebalance_offset": original_offset,
            "skipped_overlap_dates_before_fold": skipped_dates,
            "effective_local_rebalance_offset": effective_offset,
            "frozen_config": cfg.__dict__,
            "capacity_precheck": precheck,
        }
        helpers.write_independent_run(
            run_dir=run_dir,
            result=result,
            cfg=cfg,
            output_mode=args.output_mode,
            manifest=manifest,
        )
        curve = helpers.curve_from_result(result, args.initial_cash)
        label = (
            f"target_fold{target_fold} "
            f"(source_model_fold{mapping_row['source_fold']})"
        )
        per_fold_curves.append(
            {
                "label": label,
                "run_name": f"target_fold{target_fold}",
                "curve": curve,
            }
        )
        comparison_rows.append(
            {
                "result_type": "independent_fold",
                "target_fold": target_fold,
                "source_model_fold": int(mapping_row["source_fold"]),
                **scalar_summary(result["summary"]),
            }
        )
        run_records.append(
            {**manifest, "run_dir": str(run_dir), "summary": result["summary"]}
        )
        print(
            f"[OK] independent target_fold{target_fold} "
            f"source_model_fold{mapping_row['source_fold']} "
            f"{dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d} "
            f"offset={original_offset}->{effective_offset}"
        )

    plot_curves(
        per_fold_curves,
        per_fold_root / "plots",
        "r05_fwd rotation_addon_onehot independent folds",
    )

    cross_fold_root = out_root / "cross_fold"
    copied_files = copy_materialized_run(
        source_run_dir, cross_fold_root / "materialized_run"
    )
    materialized_summary = read_json(source_run_dir / "summary.json")
    materialized_initial_cash = float(
        stored_config.get("initial_cash", args.initial_cash)
    )
    cross_curve = curve_from_nav(materialized_nav, materialized_initial_cash)
    plot_curves(
        [
            {
                "label": "continuous cross-fold account",
                "run_name": str(historical["run_name"]),
                "curve": cross_curve,
            }
        ],
        cross_fold_root / "plots",
        "r05_fwd rotation_addon_onehot continuous cross-fold",
    )
    materialized_nav.to_csv(
        cross_fold_root / "continuous_nav.csv",
        index=False,
        encoding="utf-8-sig",
    )
    continuous_segments = segment_metrics(materialized_nav, fold_dates)
    continuous_segments.to_csv(
        cross_fold_root / "continuous_fold_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_rows.append(
        {
            "result_type": "continuous_cross_fold",
            "target_fold": None,
            "source_model_fold": None,
            **scalar_summary(materialized_summary),
        }
    )

    pd.DataFrame(comparison_rows).to_csv(
        out_root / "r05_addon_backtest_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    source_nav = Path(historical["nav_file"]).resolve()
    manifest = {
        "mode": "r05_addon_independent_and_continuous",
        "target_col": TARGET,
        "feature_preset": PRESET,
        "prediction_generation": False,
        "parameter_grid": False,
        "training": False,
        "data_refresh": False,
        "independent_backtest_count": len(run_records),
        "continuous_result_source": "reused_authoritative_materialized_run",
        "continuous_result_rerun": False,
        "initial_cash_independent": float(args.initial_cash),
        "source_pair_manifest": str(pair_path),
        "source_historical_root": historical["root"],
        "source_materialized_run": str(source_run_dir),
        "source_nav_file": str(source_nav),
        "source_nav_sha256": sha256_file(source_nav),
        "copied_materialized_files": copied_files,
        "fold_boundary_audit": str(out_root / "fold_boundary_audit.csv"),
        "comparison_csv": str(out_root / "r05_addon_backtest_comparison.csv"),
        "per_fold_runs": run_records,
        "duration_seconds": int(round(time.time() - started)),
    }
    manifest["all_ok"] = (
        manifest["independent_backtest_count"] == 6
        and int(audit["trading_gap_days"].sum()) == 0
        and len(copied_files) >= 3
    )
    manifest_path = out_root / "r05_addon_fold_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=bt.json_default,
        ),
        encoding="utf-8",
    )
    report = {
        "mode": manifest["mode"],
        "independent_backtest_count": manifest["independent_backtest_count"],
        "continuous_result_source": manifest["continuous_result_source"],
        "folds": list(EXPECTED_TARGET_FOLDS),
        "trading_gap_days_total": int(audit["trading_gap_days"].sum()),
        "duration_seconds": manifest["duration_seconds"],
        "all_ok": manifest["all_ok"],
        "manifest": str(manifest_path),
        "output_root": str(out_root),
    }
    (out_root / "r05_addon_fold_comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not manifest["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
