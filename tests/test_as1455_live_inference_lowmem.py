from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils import as1455_ch17_common as common
from utils.as1455_live_inference import build_inference_features_from_frame
from utils.as1455_live_inference_lowmem import (
    build_current_day_inference_features,
    context_from_frame,
    load_live_history_context,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    rng = np.random.default_rng(20260811)
    dates = pd.bdate_range("2026-06-15", periods=42)
    trade_date = pd.Timestamp(dates[-1]).normalize()
    symbols = [f"{i:06d}.SZ" for i in range(1, 81)]
    columns = list(common.base.EXPECTED_MODEL_COLUMNS)

    hist_index = pd.MultiIndex.from_product(
        [symbols, dates[:-1]], names=["symbol", "date"]
    )
    historical = pd.DataFrame(index=hist_index, columns=columns, dtype=float)
    for column in columns:
        historical[column] = rng.normal(size=len(historical))
    historical["dollar_vol"] = rng.lognormal(4.0, 0.5, size=len(historical))
    historical["sector"] = [int(symbol[:6]) % 5 for symbol, _ in historical.index]
    historical["year"] = [date.year for _, date in historical.index]
    historical["month"] = [date.month for _, date in historical.index]
    historical["weekday"] = [date.weekday() for _, date in historical.index]
    bad = rng.choice(len(historical), size=180, replace=False)
    historical.iloc[bad, historical.columns.get_loc("rsi")] = np.nan

    live_index = pd.MultiIndex.from_product(
        [symbols, [trade_date]], names=["symbol", "date"]
    )
    live = pd.DataFrame(index=live_index, columns=columns, dtype=float)
    for column in columns:
        live[column] = rng.normal(size=len(live))
    live["dollar_vol"] = rng.lognormal(4.0, 0.5, size=len(live))
    live["sector"] = [int(symbol[:6]) % 5 for symbol, _ in live.index]
    live["year"] = trade_date.year
    live["month"] = trade_date.month
    live["weekday"] = trade_date.weekday()
    for outcome in common.base.EXPECTED_OUTCOMES:
        live[outcome] = np.nan
    return historical, live, trade_date


def test_lowmem_current_day_features_match_full_history_builder() -> None:
    historical, live, trade_date = _frames()
    combined = pd.concat([historical, live]).sort_index()
    full = build_inference_features_from_frame(
        combined,
        "r01_fwd",
        "rotation_addon_onehot",
        "onehot",
        source_label="test-full",
    ).X
    dates = pd.DatetimeIndex(full.index.get_level_values("date")).normalize()
    expected = full.loc[dates == trade_date]

    context = context_from_frame(historical, trade_date)
    actual = build_current_day_inference_features(
        live,
        context,
        "rotation_addon_onehot",
        required_feature_columns=list(expected.columns),
    ).X.reindex(expected.index)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_hdf_chunk_context_matches_in_memory_context(tmp_path) -> None:
    pytest.importorskip("tables")
    historical, _live, trade_date = _frames()
    expected = context_from_frame(historical, trade_date)
    path = tmp_path / "model_data_table.h5"
    historical.to_hdf(path, key="model_data", format="table", mode="w")

    actual = load_live_history_context(path, trade_date, chunksize=97)

    pd.testing.assert_series_equal(actual.market_dollar_vol, expected.market_dollar_vol)
    pd.testing.assert_series_equal(actual.sector_dollar_vol, expected.sector_dollar_vol)
    pd.testing.assert_series_equal(actual.stock_dollar_vol, expected.stock_dollar_vol)
    assert actual.report["history_read_mode"] == "hdf_table_chunks_columns"
