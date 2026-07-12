#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared historical-run and model-signal selection for AS1455.

Both fold0-forward backtests and plotting use this module, so the definition of
"best" cannot drift between the two workflows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_CANDIDATES = (
    "01_close_auction_daily_grid/02_summary/grid_summary_compact.csv",
    "01_close_auction_grid/02_summary/grid_summary_compact.csv",
    "01_close_auction_daily_grid/02_summary/grid_summary.csv",
    "01_close_auction_grid/02_summary/grid_summary.csv",
)

HIGHER_IS_BETTER = {
    "total_return",
    "annual_return",
    "sharpe",
    "calmar",
    "max_drawdown",
    "daily_win_rate",
    "monthly_win_rate",
    "trade_win_rate",
    "round_trip_win_rate",
}
LOWER_IS_BETTER = {
    "avg_turnover",
    "annualized_turnover",
    "gross_trade_amount",
    "total_fee",
    "fee_to_initial_cash",
    "n_orders",
    "n_rejections",
}


@dataclass(frozen=True)
class HistoricalSignalSelection:
    backtest_root: str
    summary_file: str
    rank_metric: str
    rank_metric_value: float
    run_name: str
    signal_name: str
    signal_cols: str
    signal_mode: str
    signal_spec: str
    required_top_n: int
    historical_max_positions: int | None
    historical_sell_rank: int | None
    historical_rebalance_every: int | None
    historical_rebalance_offset: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def find_summary_file(root: Path) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    for relative in SUMMARY_CANDIDATES:
        path = root / relative
        if path.exists():
            return path, path.parents[1]
    for filename in ("grid_summary_compact.csv", "grid_summary.csv"):
        matches = sorted(root.glob(f"**/02_summary/{filename}"))
        if matches:
            path = matches[0]
            return path, path.parents[1]
    raise FileNotFoundError(f"cannot find grid summary under {root}")


def select_best_run(summary: pd.DataFrame, metric: str) -> pd.Series:
    frame = summary.copy()
    if "status" in frame.columns:
        ok = frame["status"].astype(str).str.lower().eq("ok")
        if ok.any():
            frame = frame.loc[ok].copy()
    if metric not in frame.columns:
        raise RuntimeError(
            f"rank metric {metric!r} not found in summary columns: "
            f"{list(frame.columns)}"
        )
    if "run_name" not in frame.columns:
        raise RuntimeError("summary does not contain run_name column")
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=[metric])
    if frame.empty:
        raise RuntimeError(f"no valid rows for metric {metric!r}")
    ascending = metric in LOWER_IS_BETTER and metric not in HIGHER_IS_BETTER
    return frame.sort_values(metric, ascending=ascending, kind="mergesort").iloc[0]


def find_latest_target_backtest_root(
    *,
    base_root: Path,
    feature_preset: str,
    target_col: str,
    rebalance_every: int,
) -> Path:
    pattern = f"{feature_preset}_{target_col}_reb{rebalance_every}_*"
    candidates = sorted(
        path for path in base_root.expanduser().resolve().glob(pattern) if path.is_dir()
    )
    valid: list[Path] = []
    for path in candidates:
        try:
            find_summary_file(path)
        except FileNotFoundError:
            continue
        valid.append(path)
    if not valid:
        raise FileNotFoundError(
            "cannot find a completed historical target backtest; "
            f"base_root={base_root} pattern={pattern}"
        )
    return valid[-1]


def _normalize_signal_cols(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise RuntimeError(f"invalid signal_cols={value!r}")
    normalized: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        number = float(token)
        if not number.is_integer() or number < 0:
            raise RuntimeError(f"signal column must be a nonnegative integer: {token!r}")
        normalized.append(str(int(number)))
    if not normalized:
        raise RuntimeError(f"invalid signal_cols={value!r}")
    return ",".join(normalized)


def required_top_n_for_signal_cols(signal_cols: str) -> int:
    columns = [int(token) for token in _normalize_signal_cols(signal_cols).split(",")]
    return max(columns) + 1


def signal_spec_from_row(row: pd.Series) -> tuple[str, int]:
    required = ("signal_name", "signal_cols", "signal_mode")
    missing = [name for name in required if name not in row.index]
    if missing:
        raise RuntimeError(f"historical grid row missing signal fields: {missing}")
    signal_name = str(row["signal_name"]).strip()
    signal_mode = str(row["signal_mode"]).strip()
    signal_cols = _normalize_signal_cols(row["signal_cols"])
    if not signal_name or signal_name.lower() == "nan":
        raise RuntimeError(f"invalid signal_name={row['signal_name']!r}")
    if signal_mode not in {"single", "mean"}:
        raise RuntimeError(f"unsupported signal_mode={signal_mode!r}")
    return (
        f"{signal_name}:{signal_cols}:{signal_mode}",
        required_top_n_for_signal_cols(signal_cols),
    )


def _optional_int(row: pd.Series, name: str) -> int | None:
    if name not in row.index or pd.isna(row[name]):
        return None
    return int(row[name])


def select_historical_signal(
    *,
    backtest_root: Path,
    rank_metric: str = "sharpe",
) -> HistoricalSignalSelection:
    root = backtest_root.expanduser().resolve()
    summary_file, _grid_dir = find_summary_file(root)
    best = select_best_run(read_csv_auto(summary_file), rank_metric)
    signal_spec, required_top_n = signal_spec_from_row(best)
    return HistoricalSignalSelection(
        backtest_root=str(root),
        summary_file=str(summary_file),
        rank_metric=rank_metric,
        rank_metric_value=float(best[rank_metric]),
        run_name=str(best["run_name"]),
        signal_name=str(best["signal_name"]),
        signal_cols=_normalize_signal_cols(best["signal_cols"]),
        signal_mode=str(best["signal_mode"]),
        signal_spec=signal_spec,
        required_top_n=required_top_n,
        historical_max_positions=_optional_int(best, "max_positions"),
        historical_sell_rank=_optional_int(best, "sell_rank"),
        historical_rebalance_every=_optional_int(best, "rebalance_every"),
        historical_rebalance_offset=_optional_int(best, "rebalance_offset"),
    )


def select_corresponding_historical_signal(
    *,
    base_root: Path,
    feature_preset: str,
    target_col: str,
    rebalance_every: int,
    rank_metric: str = "sharpe",
    explicit_backtest_root: Path | None = None,
) -> HistoricalSignalSelection:
    root = explicit_backtest_root or find_latest_target_backtest_root(
        base_root=base_root,
        feature_preset=feature_preset,
        target_col=target_col,
        rebalance_every=rebalance_every,
    )
    return select_historical_signal(backtest_root=root, rank_metric=rank_metric)
