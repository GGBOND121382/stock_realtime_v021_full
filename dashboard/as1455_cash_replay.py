from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.as1455_cash_history import daily_cash_requirements
from scripts import run_as1455_live_nine_strategy_planner as planner
from scripts import run_as1455_live_strict_oos_monitor as live
from utils import as1455_paths
from utils.as1455_model_selection import select_corresponding_historical_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_BASE = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_prediction_cache"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = planner.read_json(path)
    return value if isinstance(value, dict) else {}


def _experiment_identity(experiment: str) -> dict[str, Any]:
    match = planner.EXPERIMENT_RE.fullmatch(str(experiment))
    if match is None:
        raise RuntimeError(f"unsupported AS1455 experiment: {experiment}")
    target = match.group("target")
    return {
        "target": target,
        "target_col": f"{target}_fwd",
        "signal": match.group("signal"),
        "rebalance_every": int(match.group("reb")),
        "fold_label": match.group("fold"),
    }


def _normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if not isinstance(work.index, pd.MultiIndex):
        if not {"symbol", "date"}.issubset(work.columns):
            raise RuntimeError("prediction frame requires symbol/date")
        work["symbol"] = work["symbol"].map(live.exchange_symbol)
        work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
        work = work.set_index(["symbol", "date"])
    else:
        names = list(work.index.names)
        if "symbol" not in names or "date" not in names:
            raise RuntimeError(f"unexpected prediction index names: {names}")
        symbols = work.index.get_level_values("symbol").map(live.exchange_symbol)
        dates = pd.to_datetime(
            work.index.get_level_values("date"), errors="raise"
        ).normalize()
        work.index = pd.MultiIndex.from_arrays(
            [symbols, dates], names=["symbol", "date"]
        )
    work.columns = [int(str(column)) if str(column).isdigit() else column for column in work.columns]
    missing = set(range(5)) - set(work.columns)
    if missing:
        raise RuntimeError(f"prediction frame lacks Top-5 columns: {sorted(missing)}")
    for column in range(5):
        work[column] = pd.to_numeric(work[column], errors="raise")
    work = work[list(range(5))].sort_index()
    if work.index.duplicated().any():
        raise RuntimeError("duplicate symbol/date rows in cash replay predictions")
    return work


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _normalize_predictions(pd.read_hdf(path, "predictions"))


def _score_predictions(frame: pd.DataFrame, signal_kind: str) -> pd.Series:
    if signal_kind == "best":
        score = frame[0]
    elif signal_kind == "first3":
        score = frame[[0, 1, 2]].mean(axis=1)
    elif signal_kind == "all5":
        score = frame[[0, 1, 2, 3, 4]].mean(axis=1)
    else:
        raise RuntimeError(f"unsupported signal kind: {signal_kind}")
    return score.rename("score")


def _cache_root_from_manifest(
    manifest: dict[str, Any],
    *,
    target_col: str,
    feature_preset: str,
) -> Path:
    source = str(manifest.get("prediction_source_root") or "").strip()
    prefix = "shared_prediction_cache:"
    if source.startswith(prefix):
        candidate = Path(source[len(prefix) :].strip()).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_dir():
            return candidate.resolve()
    return (
        DEFAULT_CACHE_BASE
        / f"{feature_preset}_{target_col}_top5"
    ).resolve()


def _prediction_context(
    matrix_root: Path,
    experiment: str,
    *,
    feature_preset: str,
) -> dict[str, Any]:
    identity = _experiment_identity(experiment)
    experiment_root = matrix_root / experiment
    manifest = _read_json(
        experiment_root / "global_fold0_to_fold5_forward_manifest.json"
    )
    history_root = planner.resolve_history_root(experiment_root, manifest)
    historical_prediction = history_root / "00_predictions" / "test_preds.h5"
    segments_file = history_root / "00_predictions" / "prediction_segments.csv"
    cache_root = _cache_root_from_manifest(
        manifest,
        target_col=identity["target_col"],
        feature_preset=feature_preset,
    )
    forward_prediction = (
        cache_root
        / "fold0_forward_latest"
        / "00_predictions"
        / "fold0_forward_preds.h5"
    )
    return {
        **identity,
        "experiment_root": experiment_root,
        "manifest": manifest,
        "history_root": history_root,
        "historical_prediction": historical_prediction,
        "forward_prediction": forward_prediction,
        "segments_file": segments_file,
        "cache_root": cache_root,
    }


def _load_segments(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(
            columns=["source_fold", "target_fold", "start", "end", "n_days"]
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"target_fold", "start", "end"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing segment columns: {sorted(missing)}")
    frame = frame.copy()
    frame["start"] = pd.to_datetime(frame["start"], errors="raise").dt.normalize()
    frame["end"] = pd.to_datetime(frame["end"], errors="raise").dt.normalize()
    frame["target_fold"] = pd.to_numeric(
        frame["target_fold"], errors="raise"
    ).astype(int)
    if "source_fold" in frame.columns:
        frame["source_fold"] = pd.to_numeric(
            frame["source_fold"], errors="raise"
        ).astype(int)
    return frame.sort_values("start").reset_index(drop=True)


def _combined_prediction_scores(context: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = _read_predictions(Path(context["historical_prediction"]))
    forward = _read_predictions(Path(context["forward_prediction"]))
    signal_kind = str(context["signal"])

    historical_score = _score_predictions(historical, signal_kind).reset_index()
    forward_score = _score_predictions(forward, signal_kind).reset_index()
    historical_score["date"] = pd.to_datetime(
        historical_score["date"], errors="raise"
    ).dt.normalize()
    forward_score["date"] = pd.to_datetime(
        forward_score["date"], errors="raise"
    ).dt.normalize()

    historical_last = pd.Timestamp(historical_score["date"].max()).normalize()
    manifest_forward_start = pd.to_datetime(
        context["manifest"].get("strict_forward_start"), errors="coerce"
    )
    if pd.isna(manifest_forward_start):
        forward_start = pd.Timestamp(forward_score["date"].min()).normalize()
    else:
        forward_start = pd.Timestamp(manifest_forward_start).normalize()
    forward_score = forward_score.loc[forward_score["date"].ge(forward_start)].copy()

    overlap = set(historical_score["date"]).intersection(set(forward_score["date"]))
    if overlap:
        first = min(overlap)
        last = max(overlap)
        raise RuntimeError(
            "historical and strict-forward prediction timelines overlap; "
            f"continuous cash replay would be ambiguous: {first:%Y-%m-%d}..{last:%Y-%m-%d}"
        )
    if not forward_score.empty and pd.Timestamp(forward_score["date"].min()) <= historical_last:
        raise RuntimeError("strict-forward predictions do not start after historical predictions")

    combined = pd.concat(
        [historical_score, forward_score], ignore_index=True, sort=False
    )
    combined["symbol"] = combined["symbol"].map(live.exchange_symbol)
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    if combined.duplicated(["symbol", "date"]).any():
        raise RuntimeError("duplicate symbol/date rows after historical/forward merge")

    dates = pd.DataFrame(
        {"date": pd.DatetimeIndex(combined["date"].unique()).sort_values()}
    )
    segments = _load_segments(Path(context["segments_file"]))
    dates["segment"] = "strict_forward"
    dates["source_fold"] = pd.NA
    dates["target_fold"] = pd.NA
    for row in segments.itertuples(index=False):
        mask = dates["date"].between(pd.Timestamp(row.start), pd.Timestamp(row.end))
        dates.loc[mask, "segment"] = f"fold{int(row.target_fold)}"
        dates.loc[mask, "target_fold"] = int(row.target_fold)
        if hasattr(row, "source_fold"):
            dates.loc[mask, "source_fold"] = int(row.source_fold)
    return combined, dates


def replay_date_catalog(
    matrix_root: str | Path,
    experiment: str,
    *,
    feature_preset: str = "rotation_addon_onehot",
) -> pd.DataFrame:
    context = _prediction_context(
        Path(matrix_root).expanduser().resolve(),
        experiment,
        feature_preset=feature_preset,
    )
    _combined, dates = _combined_prediction_scores(context)
    return dates


def _segment_summary(
    catalog: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment, dates in catalog.groupby("segment", sort=False):
        date_set = set(pd.to_datetime(dates["date"]).dt.normalize())
        part = daily.loc[pd.to_datetime(daily.get("date"), errors="coerce").isin(date_set)].copy() if not daily.empty else pd.DataFrame()
        buy_days = (
            pd.to_numeric(part.get("buy_amount"), errors="coerce").fillna(0).gt(0)
            if not part.empty
            else pd.Series(dtype=bool)
        )
        incomplete = (
            buy_days
            & ~part.get(
                "cash_requirement_complete",
                pd.Series(False, index=part.index),
            ).fillna(False).astype(bool)
            if not part.empty
            else pd.Series(dtype=bool)
        )
        complete = not bool(incomplete.any())
        cash = (
            pd.to_numeric(part.get("conservative_cash_required"), errors="coerce")
            if not part.empty
            else pd.Series(dtype=float)
        )
        if not complete:
            peak = np.nan
            peak_date = pd.NaT
        elif cash.notna().any():
            idx = cash.fillna(0.0).idxmax()
            peak = float(cash.fillna(0.0).loc[idx])
            peak_date = pd.Timestamp(part.loc[idx, "date"]).normalize()
        else:
            peak = 0.0
            peak_date = pd.NaT
        rows.append(
            {
                "segment": segment,
                "start_date": pd.Timestamp(dates["date"].min()).normalize(),
                "end_date": pd.Timestamp(dates["date"].max()).normalize(),
                "n_days": int(len(dates)),
                "n_buy_days": int(buy_days.sum()) if len(buy_days) else 0,
                "peak_cash_required": peak,
                "peak_date": peak_date,
                "complete": complete,
            }
        )
    return pd.DataFrame(rows)


def replay_cash_requirements(
    matrix_root: str | Path,
    experiment: str,
    start_date: str | pd.Timestamp,
    initial_cash: float,
    raw_daily_cache_dir: str | Path,
    *,
    live_root: str | Path | None = None,
    feature_preset: str = "rotation_addon_onehot",
) -> dict[str, Any]:
    """Replay a strategy from empty at an arbitrary historical/forward date.

    This is a diagnostic account replay for cash-buffer sizing. It reuses saved
    one-fold-lag historical predictions, saved fold0 strict-forward predictions,
    the selected frozen trading configuration, and raw-daily execution data. It
    does not train models, run Fold/Grid, or mutate canonical/tracking artifacts.

    The original historical rebalance phase is preserved. If ``start_date`` is
    not itself an available prediction/trading date, the account begins on the
    first available date after it and remains cash until the next scheduled
    rebalance. The executable tracking/live corporate-action approximation is
    used, so synthetic share-factor fractional holdings are not introduced.
    """
    matrix = Path(matrix_root).expanduser().resolve()
    raw_daily = Path(raw_daily_cache_dir).expanduser().resolve()
    if not raw_daily.is_dir():
        raise FileNotFoundError(raw_daily)
    cash = float(initial_cash)
    if not np.isfinite(cash) or cash <= 0:
        raise ValueError(f"initial_cash must be positive: {initial_cash}")

    context = _prediction_context(
        matrix,
        experiment,
        feature_preset=feature_preset,
    )
    predictions, catalog = _combined_prediction_scores(context)
    requested = pd.Timestamp(start_date).normalize()
    available = catalog.loc[catalog["date"].ge(requested), "date"]
    if available.empty:
        raise RuntimeError(
            f"no replay dates on or after {requested:%Y-%m-%d} for {experiment}"
        )
    effective_start = pd.Timestamp(available.iloc[0]).normalize()
    all_dates = pd.DatetimeIndex(catalog["date"]).normalize()
    start_index = int(np.flatnonzero(all_dates == effective_start)[0])
    replay_catalog = catalog.loc[catalog["date"].ge(effective_start)].copy().reset_index(drop=True)
    replay_predictions = predictions.loc[
        predictions["date"].ge(effective_start), ["symbol", "date", "score"]
    ].copy()
    if replay_predictions.empty:
        raise RuntimeError("cash replay has no predictions after selected start")

    selection = select_corresponding_historical_signal(
        base_root=Path(as1455_paths.TARGET_BACKTEST_ROOT),
        feature_preset=feature_preset,
        target_col=context["target_col"],
        rebalance_every=int(context["rebalance_every"]),
        rank_metric="sharpe",
        explicit_backtest_root=Path(context["history_root"]),
    )
    actual_spec = planner.selection_spec(selection)
    expected_spec = planner.EXPECTED_SIGNAL_SPEC[str(context["signal"])]
    if actual_spec != expected_spec:
        raise RuntimeError(
            f"fixed signal mismatch for {experiment}: {actual_spec} != {expected_spec}"
        )
    historical_config, config_path = live.load_historical_run_config(selection)

    v7 = live.load_v7_module()
    symbols = sorted(replay_predictions["symbol"].astype(str).unique())
    execution, _report = v7.build_execution_panel(
        symbols,
        raw_daily,
        pd.DataFrame(),
        set(),
        st_status=pd.DataFrame(),
        last5_panel=pd.DataFrame(),
        raw_5m_cache_dir=None,
    )
    execution["date"] = pd.to_datetime(
        execution["date"], errors="coerce"
    ).dt.normalize()
    replay_date_set = set(pd.DatetimeIndex(replay_catalog["date"]).normalize())
    execution = execution.loc[execution["date"].isin(replay_date_set)].copy()
    if execution.empty:
        raise RuntimeError("cash replay execution panel is empty")

    phase = {
        "effective_forward_offset": int(selection.historical_rebalance_offset or 0)
    }
    cfg = live.build_trade_config(
        v7,
        selection,
        historical_config,
        phase,
        cash,
        "none",
        0.05,
    )
    cfg = replace(
        cfg,
        corporate_action_mode=planner.synthetic_corporate_action_mode(
            historical_config
        ),
    )
    result = v7.backtest(
        replay_predictions,
        execution,
        cfg,
        corporate_actions=None,
        initial_positions=None,
        day_index_start=start_index,
        allow_single_date=True,
    )
    orders = result["orders"].copy()
    live_path = (
        Path(live_root).expanduser().resolve()
        if live_root is not None
        else PROJECT_ROOT / "saved_data" / "ashare_ml4t" / "live_as1455"
    )
    daily = daily_cash_requirements(
        orders,
        live_root=live_path,
        raw_daily_cache_dir=raw_daily,
    )
    segments = _segment_summary(replay_catalog, daily)

    if daily.empty:
        overall_peak = 0.0
        overall_peak_date = pd.NaT
        complete = True
    else:
        buy_days = pd.to_numeric(daily["buy_amount"], errors="coerce").fillna(0).gt(0)
        incomplete = buy_days & ~daily["cash_requirement_complete"].fillna(False).astype(bool)
        complete = not bool(incomplete.any())
        cash_required = pd.to_numeric(
            daily["conservative_cash_required"], errors="coerce"
        )
        if not complete:
            overall_peak = np.nan
            overall_peak_date = pd.NaT
        elif cash_required.notna().any():
            idx = cash_required.fillna(0.0).idxmax()
            overall_peak = float(cash_required.fillna(0.0).loc[idx])
            overall_peak_date = pd.Timestamp(daily.loc[idx, "date"]).normalize()
        else:
            overall_peak = 0.0
            overall_peak_date = pd.NaT

    peak_segment = None
    if pd.notna(overall_peak_date):
        match = replay_catalog.loc[
            replay_catalog["date"].eq(pd.Timestamp(overall_peak_date))
        ]
        if not match.empty:
            peak_segment = str(match.iloc[0]["segment"])

    final_nav = pd.to_numeric(
        pd.Series([result.get("final_state", {}).get("nav")]), errors="coerce"
    ).iloc[0]
    if pd.isna(final_nav) and not result["nav"].empty:
        final_nav = pd.to_numeric(
            pd.Series([result["nav"].iloc[-1].get("nav")]), errors="coerce"
        ).iloc[0]

    return {
        "experiment": experiment,
        "requested_start_date": requested,
        "effective_start_date": effective_start,
        "start_segment": str(replay_catalog.iloc[0]["segment"]),
        "end_date": pd.Timestamp(replay_catalog.iloc[-1]["date"]).normalize(),
        "initial_cash": cash,
        "final_nav": float(final_nav) if pd.notna(final_nav) else np.nan,
        "n_replay_days": int(len(replay_catalog)),
        "n_orders": int(len(orders)),
        "overall_peak": overall_peak,
        "overall_peak_date": overall_peak_date,
        "overall_peak_segment": peak_segment,
        "peak_to_initial_cash": (
            float(overall_peak) / cash if pd.notna(overall_peak) else np.nan
        ),
        "complete": complete,
        "daily": daily,
        "segments": segments,
        "catalog": replay_catalog,
        "historical_prediction_file": str(context["historical_prediction"]),
        "forward_prediction_file": str(context["forward_prediction"]),
        "historical_config_file": str(config_path),
        "corporate_action_mode": cfg.corporate_action_mode,
        "model_training": False,
        "grid_search": False,
        "canonical_artifacts_mutated": False,
    }
