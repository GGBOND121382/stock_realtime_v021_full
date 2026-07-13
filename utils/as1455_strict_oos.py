#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict out-of-sample configuration helpers for AS1455 fold0-forward runs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from utils.as1455_model_selection import HistoricalSignalSelection


LEADERBOARD_SPECS = {
    "leaderboard_by_total_return.csv": ("total_return", False),
    "leaderboard_by_annual_return.csv": ("annual_return", False),
    "leaderboard_by_sharpe.csv": ("sharpe", False),
    "leaderboard_by_calmar.csv": ("calmar", False),
    "leaderboard_by_max_drawdown.csv": ("max_drawdown", False),
    "leaderboard_by_trade_win_rate.csv": ("trade_win_rate", False),
    "leaderboard_by_low_turnover.csv": ("avg_turnover", True),
    "leaderboard_by_fee_efficiency.csv": ("fee_to_initial_cash", True),
}


def historical_trading_config(
    selection: HistoricalSignalSelection,
    target_rebalance_every: int,
) -> dict[str, int]:
    fields = {
        "max_positions": selection.historical_max_positions,
        "sell_rank": selection.historical_sell_rank,
        "rebalance_every": selection.historical_rebalance_every,
        "rebalance_offset": selection.historical_rebalance_offset,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise RuntimeError(
            "strict_oos requires complete historical trading parameters; "
            f"missing={missing} run={selection.run_name}"
        )
    config = {name: int(value) for name, value in fields.items()}
    if config["rebalance_every"] != int(target_rebalance_every):
        raise RuntimeError(
            "historical rebalance_every does not match target protocol: "
            f"historical={config['rebalance_every']} "
            f"target={target_rebalance_every}"
        )
    if not 0 <= config["rebalance_offset"] < config["rebalance_every"]:
        raise RuntimeError(f"invalid historical rebalance_offset: {config}")
    if config["max_positions"] <= 0 or config["sell_rank"] <= 0:
        raise RuntimeError(f"invalid historical trading config: {config}")
    return config


def historical_phase_window(selection: HistoricalSignalSelection) -> dict[str, Any]:
    fields = {
        "historical_first_date": selection.historical_date_min,
        "historical_last_date": selection.historical_date_max,
        "historical_n_days": selection.historical_n_days,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise RuntimeError(
            "strict_oos phase alignment requires the selected historical run's "
            "date_min/date_max/n_days or a materialized close_auction_nav.csv; "
            f"missing={missing} run={selection.run_name}"
        )
    n_days = int(fields["historical_n_days"])
    if n_days <= 0:
        raise RuntimeError(f"historical_n_days must be positive: {n_days}")
    first = pd.Timestamp(fields["historical_first_date"]).normalize()
    last = pd.Timestamp(fields["historical_last_date"]).normalize()
    if first > last:
        raise RuntimeError(f"historical window is reversed: {first} > {last}")
    return {
        "historical_first_date": first.strftime("%Y-%m-%d"),
        "historical_last_date": last.strftime("%Y-%m-%d"),
        "historical_n_days": n_days,
    }


def apply_strict_oos_args(
    args: Any,
    selection: HistoricalSignalSelection,
) -> dict[str, int]:
    """Freeze non-phase parameters and defer offset translation to the grid.

    ``rebalance_offset`` is local to each v7 backtest window.  The historical
    integer therefore cannot be copied directly into fold0-forward.  We pass the
    historical phase window to the shared grid, which derives the forward-local
    effective offset from the full execution calendar before configs are built.
    """
    config = historical_trading_config(selection, int(args.rebalance_every))
    phase = historical_phase_window(selection)
    args.max_positions_list = str(config["max_positions"])
    args.sell_rank_list = str(config["sell_rank"])
    args.rebalance_every = int(config["rebalance_every"])
    args.offset_mode = "zero" if config["rebalance_every"] == 1 else "full"
    args.rebalance_offset_list = None
    args.rebalance_phase_history_offset = int(config["rebalance_offset"])
    args.rebalance_phase_history_first_date = phase["historical_first_date"]
    args.rebalance_phase_history_last_date = phase["historical_last_date"]
    args.rebalance_phase_history_n_days = int(phase["historical_n_days"])
    return config


def _read_summary(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def _match_strict_row(
    frame: pd.DataFrame,
    selection: HistoricalSignalSelection,
    config: dict[str, int],
) -> pd.DataFrame:
    required = {
        "run_name",
        "status",
        "signal_name",
        "signal_cols",
        "signal_mode",
        "max_positions",
        "sell_rank",
        "rebalance_every",
        "rebalance_offset",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"forward grid summary missing columns: {sorted(missing)}")

    signal_cols = frame["signal_cols"].astype(str).str.replace(r"\.0$", "", regex=True)
    mask = (
        frame["status"].astype(str).str.lower().eq("ok")
        & frame["signal_name"].astype(str).eq(selection.signal_name)
        & signal_cols.eq(selection.signal_cols)
        & frame["signal_mode"].astype(str).eq(selection.signal_mode)
        & pd.to_numeric(frame["max_positions"], errors="coerce").eq(
            config["max_positions"]
        )
        & pd.to_numeric(frame["sell_rank"], errors="coerce").eq(
            config["sell_rank"]
        )
        & pd.to_numeric(frame["rebalance_every"], errors="coerce").eq(
            config["rebalance_every"]
        )
        & pd.to_numeric(frame["rebalance_offset"], errors="coerce").eq(
            config["rebalance_offset"]
        )
    )
    selected = frame.loc[mask].copy()
    if len(selected) != 1:
        raise RuntimeError(
            "strict_oos expected exactly one successful phase-aligned run, "
            f"got={len(selected)} config={config} signal={selection.signal_spec}"
        )
    return selected


def _write_strict_summaries(
    summary_dir: Path,
    grid_root: Path,
    selected: pd.DataFrame,
) -> None:
    selected.to_csv(summary_dir / "grid_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(
        summary_dir / "grid_summary_compact.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(grid_root / "grid_summary.csv", index=False, encoding="utf-8-sig")

    for filename, (metric, ascending) in LEADERBOARD_SPECS.items():
        path = summary_dir / filename
        if metric not in selected.columns:
            path.unlink(missing_ok=True)
            continue
        ranked = selected.copy()
        ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        ranked.to_csv(path, index=False, encoding="utf-8-sig")

    if "signal_name" in selected.columns:
        for metric, ascending in (
            ("sharpe", False),
            ("total_return", False),
            ("calmar", False),
            ("max_drawdown", False),
        ):
            path = summary_dir / f"best_by_signal_{metric}.csv"
            if metric not in selected.columns:
                path.unlink(missing_ok=True)
                continue
            ranked = selected.copy()
            ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
            ranked = ranked.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
            ranked.to_csv(path, index=False, encoding="utf-8-sig")


def _phase_aligned_configs(
    grid_root: Path,
    selection: HistoricalSignalSelection,
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    engine_path = grid_root / "grid_engine_manifest.json"
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    alignment = engine.get("rebalance_phase_alignment")
    if not isinstance(alignment, dict):
        raise RuntimeError(
            "strict_oos grid manifest is missing rebalance_phase_alignment"
        )

    historical = historical_trading_config(
        selection,
        int(selection.historical_rebalance_every or -1),
    )
    effective_offset = alignment.get("effective_forward_offset")
    if effective_offset is None:
        raise RuntimeError(
            "rebalance_phase_alignment is missing effective_forward_offset"
        )
    forward = dict(historical)
    forward["rebalance_offset"] = int(effective_offset)
    if int(alignment.get("historical_offset", -1)) != historical["rebalance_offset"]:
        raise RuntimeError(
            "phase manifest historical_offset does not match selected history: "
            f"alignment={alignment} historical={historical}"
        )
    if int(alignment.get("rebalance_every", -1)) != historical["rebalance_every"]:
        raise RuntimeError(
            "phase manifest rebalance_every does not match selected history: "
            f"alignment={alignment} historical={historical}"
        )
    return historical, forward, alignment


def finalize_strict_oos_grid(
    out_root: Path,
    selection: HistoricalSignalSelection,
) -> dict[str, Any]:
    """Audit that forward contains one phase-aligned historical configuration."""
    out_root = out_root.expanduser().resolve()
    grid_root = out_root / "01_close_auction_grid"
    summary_dir = grid_root / "02_summary"
    summary_path = summary_dir / "grid_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    historical_config, forward_config, alignment = _phase_aligned_configs(
        grid_root, selection
    )
    generated = _read_summary(summary_path)
    generated.to_csv(
        summary_dir / "grid_summary_generated_before_strict_filter.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected = _match_strict_row(generated, selection, forward_config)
    run_name = str(selected.iloc[0]["run_name"])

    runs_root = grid_root / "01_runs"
    removed_runs: list[str] = []
    if runs_root.exists():
        for path in runs_root.iterdir():
            if path.is_dir() and path.name != run_name:
                shutil.rmtree(path)
                removed_runs.append(path.name)

    logs_root = grid_root / "04_logs"
    if logs_root.exists():
        for path in logs_root.glob("*.log"):
            if path.stem != run_name:
                path.unlink(missing_ok=True)

    _write_strict_summaries(summary_dir, grid_root, selected)
    payload = {
        "evaluation_mode": "strict_oos",
        "historical_trading_parameters_reused": True,
        "historical_rebalance_phase_reused": True,
        "historical_offset_numeric_reused": bool(
            historical_config["rebalance_offset"]
            == forward_config["rebalance_offset"]
        ),
        "historical_selection": selection.to_dict(),
        "historical_config": historical_config,
        "rebalance_phase_alignment": alignment,
        "retained_run_name": run_name,
        "retained_config": forward_config,
        "generated_config_count": int(len(generated)),
        "retained_config_count": 1,
        "removed_run_count": int(len(removed_runs)),
        "removed_runs": removed_runs,
    }
    (grid_root / "strict_oos_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    engine_manifest_path = grid_root / "grid_engine_manifest.json"
    if engine_manifest_path.exists():
        engine = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
        engine.update(payload)
        engine_manifest_path.write_text(
            json.dumps(engine, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload
