#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-memory preparation of current-day AS1455 Chapter-17 inference features.

The historical feature matrix is only needed for the liquidity rolling context
used by compact add-ons. Rotation/breadth features are same-day cross-sectional
features. This module therefore scans historical model_data in chunks when the
HDF key is table-formatted and retains only compact aggregates/tails; fixed-HDF
files fall back to one full read but release it before any expanded feature
matrix or TensorFlow model is created.
"""
from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from utils import as1455_ch17_common as common

DUMMY_RE = re.compile(r"^sector_\d+$")
HISTORY_TAIL = 20


@dataclass
class LiveHistoryContext:
    stock_dollar_vol: pd.Series
    market_dollar_vol: pd.Series
    sector_dollar_vol: pd.Series
    symbol_sample: str
    report: dict[str, Any]


def base_feature_columns() -> list[str]:
    return [
        column
        for column in common.base.EXPECTED_MODEL_COLUMNS
        if column not in common.base.EXPECTED_OUTCOMES
    ]


def _normalize_chunk_index(frame: pd.DataFrame) -> pd.DataFrame:
    if list(frame.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected model_data index: {frame.index.names}")
    symbols = frame.index.get_level_values("symbol").astype(str)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("date"))).normalize()
    frame.index = pd.MultiIndex.from_arrays([symbols, dates], names=["symbol", "date"])
    return frame


def _complete_feature_mask(frame: pd.DataFrame, feature_cols: Iterable[str]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for column in feature_cols:
        if column not in frame.columns:
            raise RuntimeError(f"model_data missing feature column: {column}")
        mask &= frame[column].notna().to_numpy()
    return mask


def _merge_series_sum(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None or left.empty:
        return right.astype(float)
    return left.add(right, fill_value=0.0).astype(float)


def _stock_tail_merge(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None or left.empty:
        candidate = right
    else:
        candidate = pd.concat([left, right])
    if candidate.empty:
        return candidate.astype(float)
    candidate = candidate[~candidate.index.duplicated(keep="last")].sort_index()
    return candidate.groupby(level="symbol", sort=False).tail(HISTORY_TAIL).astype(float)


def _consume_history_chunks(
    chunks: Iterable[pd.DataFrame],
    trade_date: pd.Timestamp,
    mode: str,
) -> LiveHistoryContext:
    feature_cols = base_feature_columns()
    market: pd.Series | None = None
    sector: pd.Series | None = None
    stock: pd.Series | None = None
    rows_scanned = 0
    rows_complete = 0
    symbol_sample = ""

    for frame in chunks:
        if frame.empty:
            continue
        frame = _normalize_chunk_index(frame)
        if not symbol_sample and len(frame):
            symbol_sample = str(frame.index.get_level_values("symbol")[0])
        rows_scanned += int(len(frame))
        dates = pd.DatetimeIndex(frame.index.get_level_values("date"))
        before = dates < pd.Timestamp(trade_date).normalize()
        if not before.any():
            continue
        if not bool(before.all()):
            frame = frame.loc[before]
        mask = _complete_feature_mask(frame, feature_cols)
        if not mask.any():
            continue
        valid = frame.loc[mask, ["dollar_vol", "sector"]]
        rows_complete += int(len(valid))
        dv = pd.to_numeric(valid["dollar_vol"], errors="coerce")
        sectors = pd.to_numeric(valid["sector"], errors="coerce")
        finite = np.isfinite(dv.to_numpy(dtype=float)) & sectors.notna().to_numpy()
        valid = valid.loc[finite].copy()
        valid["dollar_vol"] = pd.to_numeric(valid["dollar_vol"], errors="raise").astype(float)
        valid["sector"] = pd.to_numeric(valid["sector"], errors="raise").astype(int)
        if valid.empty:
            continue

        d = pd.DatetimeIndex(valid.index.get_level_values("date"))
        m = valid.groupby(d, sort=False)["dollar_vol"].sum()
        m.index.name = "date"
        market = _merge_series_sum(market, m)

        temp = valid.reset_index()[["date", "sector", "dollar_vol"]]
        s = temp.groupby(["date", "sector"], sort=False)["dollar_vol"].sum()
        sector = _merge_series_sum(sector, s)

        stock = _stock_tail_merge(stock, valid["dollar_vol"])

    if not symbol_sample:
        raise RuntimeError("model_data contains no historical symbol rows")
    if market is None or market.empty or sector is None or sector.empty or stock is None or stock.empty:
        raise RuntimeError("no feature-complete historical rows available for low-memory live context")

    market = market.sort_index().tail(HISTORY_TAIL)
    sector_df = sector.rename("dollar_vol").reset_index().sort_values(["sector", "date"])
    sector_df = sector_df.groupby("sector", sort=False).tail(HISTORY_TAIL)
    sector = sector_df.set_index(["date", "sector"])["dollar_vol"].sort_index()
    stock = stock.sort_index()
    return LiveHistoryContext(
        stock_dollar_vol=stock,
        market_dollar_vol=market,
        sector_dollar_vol=sector,
        symbol_sample=symbol_sample,
        report={
            "history_read_mode": mode,
            "history_rows_scanned": rows_scanned,
            "history_feature_complete_rows": rows_complete,
            "stock_context_rows": int(len(stock)),
            "market_context_dates": int(len(market)),
            "sector_context_rows": int(len(sector)),
            "history_tail_observations": HISTORY_TAIL,
        },
    )


def load_live_history_context(
    model_data_path: Path,
    trade_date: pd.Timestamp,
    *,
    chunksize: int = 100_000,
) -> LiveHistoryContext:
    """Load only the historical context needed by current-day add-ons.

    Table HDF uses chunk iteration. Fixed HDF cannot be selected/chunked by
    PyTables, so it is read once, compacted immediately, and deleted before any
    expanded current-day feature matrix or TensorFlow checkpoint is loaded.
    """
    model_data_path = Path(model_data_path).expanduser().resolve()
    with pd.HDFStore(model_data_path, mode="r") as store:
        storer = store.get_storer("model_data")
        is_table = bool(getattr(storer, "is_table", False))
    if is_table:
        with pd.HDFStore(model_data_path, mode="r") as store:
            iterator = store.select(
                "model_data",
                columns=base_feature_columns(),
                chunksize=int(chunksize),
            )
            return _consume_history_chunks(iterator, trade_date, "hdf_table_chunks_columns")

    historical = pd.read_hdf(model_data_path, "model_data")
    try:
        context = _consume_history_chunks([historical], trade_date, "hdf_fixed_full_compact_then_free")
    finally:
        del historical
        gc.collect()
    return context


def context_from_frame(historical: pd.DataFrame, trade_date: pd.Timestamp) -> LiveHistoryContext:
    """Test/helper path with identical compaction semantics."""
    return _consume_history_chunks([historical], trade_date, "in_memory_test")


def _safe_ratio(current: float, history: pd.Series, *, window: int, min_periods: int) -> float:
    values = pd.to_numeric(history, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().tail(window)
    if len(values) < min_periods:
        return 1.0
    den = float(values.mean())
    if not np.isfinite(den) or den == 0 or not np.isfinite(current):
        return 1.0
    value = float(current) / den
    return value if np.isfinite(value) else 1.0


def _add_current_compact_addons(
    X_rot: pd.DataFrame,
    context: LiveHistoryContext,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Exact current-day slice of add_compact_addon_features()."""
    out = X_rot.copy()
    dates = pd.Index(out.index.get_level_values("date"), name="date")
    if dates.nunique() != 1:
        raise RuntimeError("low-memory live feature builder expects exactly one current date")
    sectors = out["sector"].astype(int)
    groups: dict[str, list[str]] = {
        "market_regime": [],
        "market_breadth": [],
        "sector_breadth": [],
        "sector_liquidity_addon": [],
        "stock_liquidity": [],
    }

    for column in common.addon.CORE_RETURN_COLS:
        values = pd.to_numeric(out[column], errors="coerce")
        mean = float(values.mean())
        std = float(values.std(ddof=0)) if len(values) else 0.0
        positive = float(values.gt(0).mean())
        cols = [
            f"market_{column}_mean",
            f"market_{column}_std",
            f"market_{column}_positive_rate",
        ]
        out[cols[0]] = mean
        out[cols[1]] = std
        out[cols[2]] = positive
        groups["market_regime"] += cols[:2]
        groups["market_breadth"].append(cols[2])

    current_market_dv = float(pd.to_numeric(out["dollar_vol"], errors="coerce").sum())
    out["market_dollar_vol_ratio_20"] = _safe_ratio(
        current_market_dv,
        context.market_dollar_vol,
        window=20,
        min_periods=5,
    )
    groups["market_regime"].append("market_dollar_vol_ratio_20")

    sector_count = sectors.value_counts().to_dict()
    for column in ["r01", "r05"]:
        positive = pd.to_numeric(out[column], errors="coerce").gt(0).astype(float)
        pos_sum = positive.groupby(sectors).transform("sum").to_numpy(dtype=float)
        counts = sectors.map(sector_count).to_numpy(dtype=float)
        self_pos = positive.to_numpy(dtype=float)
        full = np.divide(pos_sum, counts, out=np.zeros_like(pos_sum), where=counts != 0)
        ex_self = np.where(
            counts > 1,
            np.divide(pos_sum - self_pos, counts - 1, out=np.zeros_like(pos_sum), where=(counts - 1) != 0),
            full,
        )
        name = f"sector_{column}_positive_rate_ex_self"
        out[name] = ex_self
        groups["sector_breadth"].append(name)

    current_sector_dv = out.assign(__sector=sectors).groupby("__sector", sort=False)["dollar_vol"].sum()
    sector_hist = context.sector_dollar_vol.rename("dollar_vol").reset_index()
    ratios: dict[int, float] = {}
    for sector_value, current in current_sector_dv.items():
        hist = sector_hist.loc[sector_hist["sector"].astype(int).eq(int(sector_value)), "dollar_vol"]
        ratios[int(sector_value)] = _safe_ratio(float(current), hist, window=20, min_periods=5)
    out["sector_dollar_vol_ratio_20"] = sectors.map(ratios).astype(float)
    groups["sector_liquidity_addon"].append("sector_dollar_vol_ratio_20")

    stock_hist = context.stock_dollar_vol
    current_dv = pd.to_numeric(out["dollar_vol"], errors="coerce")
    symbols = out.index.get_level_values("symbol").astype(str)
    for window, min_periods in [(5, 3), (20, 5)]:
        ratio_values: list[float] = []
        for symbol, current in zip(symbols, current_dv.to_numpy(dtype=float)):
            try:
                hist = stock_hist.xs(str(symbol), level="symbol")
            except KeyError:
                hist = pd.Series(dtype=float)
            ratio_values.append(
                _safe_ratio(float(current), hist, window=window, min_periods=min_periods)
            )
        name = f"dollar_vol_ratio_{window}"
        out[name] = ratio_values
        groups["stock_liquidity"].append(name)

    new_cols = [column for columns in groups.values() for column in columns]
    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)
    if out[new_cols].isna().any().any():
        bad = out[new_cols].isna().sum()
        raise RuntimeError(f"NA in low-memory current add-on features: {bad[bad > 0].to_dict()}")
    return out, new_cols, groups


def build_current_day_inference_features(
    live_base: pd.DataFrame,
    context: LiveHistoryContext,
    feature_preset: str,
    *,
    required_feature_columns: list[str] | None = None,
) -> common.FeatureBuildResult:
    if feature_preset not in common.FEATURE_PRESETS:
        raise RuntimeError(f"bad feature_preset={feature_preset!r}; expected {common.FEATURE_PRESETS}")
    feature_cols = base_feature_columns()
    frame = live_base.copy()
    if list(frame.index.names) != ["symbol", "date"]:
        raise RuntimeError(f"unexpected live index names: {frame.index.names}")
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise RuntimeError(f"live base missing columns: {missing}")
    X_base = frame[feature_cols].dropna().sort_index()
    if X_base.empty:
        raise RuntimeError("no feature-complete current rows")
    if pd.DatetimeIndex(X_base.index.get_level_values("date")).nunique() != 1:
        raise RuntimeError("expected one current trade date")

    X_rot, rotation_cols = common.base.add_sector_rotation_features(X_base)
    addon_cols: list[str] = []
    feature_groups: dict[str, list[str]] = {}
    if feature_preset == "rotation_onehot":
        X_context = X_rot
    else:
        X_context, addon_cols, feature_groups = _add_current_compact_addons(X_rot, context)
    X_final, no_scale_cols, sector_onehot_cols = common.base.apply_sector_encoding(X_context, "onehot")

    if required_feature_columns is not None:
        required = list(required_feature_columns)
        missing_required = [column for column in required if column not in X_final.columns]
        illegal = [column for column in missing_required if not DUMMY_RE.fullmatch(column)]
        if illegal:
            raise RuntimeError(f"low-memory live matrix missing non-dummy model columns: {illegal[:20]}")
        for column in missing_required:
            X_final[column] = np.uint8(0)
        X_final = X_final[required]
        sector_onehot_cols = [column for column in required if DUMMY_RE.fullmatch(column)]
        no_scale_cols = list(sector_onehot_cols)

    y = pd.Series(np.nan, index=X_final.index, dtype=float)
    report = {
        **context.report,
        "row_mode": "low_memory_current_day_only",
        "rows_after_feature_dropna": int(len(X_final)),
        "feature_preset": feature_preset,
        "base_feature_count": int(X_base.shape[1]),
        "rotation_feature_count": int(len(rotation_cols)),
        "addon_feature_count": int(len(addon_cols)),
        "final_feature_count": int(X_final.shape[1]),
        "sector_encoding": "onehot",
        "sector_onehot_count": int(len(sector_onehot_cols)),
        "addon_feature_cols": list(addon_cols),
        "addon_feature_groups": feature_groups,
    }
    return common.FeatureBuildResult(
        X=X_final,
        y=y,
        no_scale_cols=list(no_scale_cols),
        rotation_cols=list(rotation_cols),
        addon_cols=list(addon_cols),
        feature_groups=feature_groups,
        sector_onehot_cols=list(sector_onehot_cols),
        report=report,
    )
