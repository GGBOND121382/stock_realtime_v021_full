from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "live_monitor", ROOT / "scripts" / "run_as1455_live_strict_oos_monitor.py"
)
assert SPEC and SPEC.loader
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


def test_live_base_feature_contract_and_symbol_style(tmp_path: Path):
    row = {column: 1.0 for column in live.BASE_FEATURE_COLUMNS}
    row.update({"symbol": "600001.SH", "date": "2026-07-22"})
    path = tmp_path / "features.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    out = live.load_live_base_features(path, pd.Index(["600001"]), pd.Timestamp("2026-07-22"))
    assert out.index.names == ["symbol", "date"]
    assert out.index.get_level_values("symbol").tolist() == ["600001"]
    assert list(out.columns) == live.common.base.EXPECTED_MODEL_COLUMNS
    assert out[live.common.base.EXPECTED_OUTCOMES].isna().all().all()


def test_positions_and_marks_are_fail_closed(tmp_path: Path):
    positions_file = tmp_path / "positions.csv"
    pd.DataFrame([
        {"symbol": "600001.SH", "shares": 100, "buy_date": "2026-07-21", "avg_entry_price": 9.5}
    ]).to_csv(positions_file, index=False)
    positions = live.load_positions(positions_file, allow_missing_buy_date=False)
    assert positions["symbol"].tolist() == ["600001.SH"]

    execution = pd.DataFrame([
        {"date": "2026-07-22", "symbol": "600001.SH", "raw_close_1500": 10.0}
    ])
    assert np.isclose(live.current_marked_nav(500.0, positions, execution), 1500.0)

    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"symbol": "600001.SH", "shares": 100}]).to_csv(bad, index=False)
    try:
        live.load_positions(bad, allow_missing_buy_date=False)
    except RuntimeError as exc:
        assert "T+1" in str(exc)
    else:
        raise AssertionError("missing buy_date must fail")


def test_v7_single_date_accepts_actual_state_and_preserves_partial_positions():
    v7 = live.load_v7_module()
    date = pd.Timestamp("2026-07-22")
    preds = pd.DataFrame([
        {"date": date, "symbol": "600002.SH", "score": 3.0},
        {"date": date, "symbol": "600003.SH", "score": 2.0},
        {"date": date, "symbol": "600001.SH", "score": 1.0},
    ])
    execution = pd.DataFrame([
        {"date": date, "symbol": symbol, "raw_close_1500": 10.0,
         "qfq_close_1500": 10.0, "raw_preclose": 10.0,
         "prev_raw_close_1500": 10.0, "event_ratio": 1.0,
         "tradable": True, "is_st": False, "is_mainboard": True,
         "up_limit": 11.0, "down_limit": 9.0,
         "last5_volume": np.nan, "last5_amount": np.nan}
        for symbol in ["600001.SH", "600002.SH", "600003.SH"]
    ])
    cfg = v7.TradeConfig(
        max_positions=1, buy_candidate_rank=2, sell_rank=2,
        rebalance_every=1, rebalance_offset=0, initial_cash=1000.0,
        commission_rate=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0,
        slippage_bps=0.0, profile="close_auction_skip_limit",
        mainboard_only=True, min_price=0.0, limit_eps=1e-6,
        lot_size=100, min_commission=0.0, exclude_st=True,
        capacity_mode="none", participation_rate=0.05,
        corporate_action_mode="none", corporate_action_threshold=1e-3,
    )
    initial = pd.DataFrame([
        {"symbol": "600001.SH", "shares": 100, "buy_date": "2026-07-21", "avg_entry_price": 9.0}
    ])
    result = v7.backtest(
        preds, execution, cfg, initial_positions=initial,
        day_index_start=0, allow_single_date=True,
    )
    assert len(result["nav"]) == 1
    assert set(result["orders"]["side"]) == {"sell", "buy"}
    final_symbols = {row["symbol"] for row in result["final_state"]["positions"]}
    assert final_symbols == {"600002.SH"}
    assert result["final_state"]["n_dates"] == 1


def test_v7_default_multiday_behavior_is_unchanged():
    v7 = live.load_v7_module()
    dates = [pd.Timestamp("2026-07-21"), pd.Timestamp("2026-07-22")]
    preds = pd.DataFrame([
        {"date": date, "symbol": "600001.SH", "score": 1.0}
        for date in dates
    ])
    execution = pd.DataFrame([
        {"date": date, "symbol": "600001.SH", "raw_close_1500": 10.0,
         "qfq_close_1500": 10.0, "raw_preclose": 10.0,
         "prev_raw_close_1500": 10.0, "event_ratio": 1.0,
         "tradable": True, "is_st": False, "is_mainboard": True,
         "up_limit": 11.0, "down_limit": 9.0,
         "last5_volume": np.nan, "last5_amount": np.nan}
        for date in dates
    ])
    cfg = v7.TradeConfig(
        max_positions=1, buy_candidate_rank=1, sell_rank=1,
        rebalance_every=1, rebalance_offset=0, initial_cash=2000.0,
        commission_rate=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0,
        slippage_bps=0.0, profile="close_auction_skip_limit",
        mainboard_only=True, min_price=0.0, limit_eps=1e-6,
        lot_size=100, min_commission=0.0, exclude_st=True,
        capacity_mode="none", participation_rate=0.05,
        corporate_action_mode="none", corporate_action_threshold=1e-3,
    )
    old_style = v7.backtest(preds, execution, cfg)
    explicit_defaults = v7.backtest(
        preds, execution, cfg, initial_positions=None,
        day_index_start=0, allow_single_date=False,
    )
    pd.testing.assert_frame_equal(old_style["nav"], explicit_defaults["nav"])
    pd.testing.assert_frame_equal(old_style["orders"], explicit_defaults["orders"])


def test_capacity_mode_is_fail_closed():
    live.validate_live_capacity_mode("none")
    try:
        live.validate_live_capacity_mode("last5_amount")
    except RuntimeError as exc:
        assert "14:55-15:00 capacity" in str(exc)
    else:
        raise AssertionError("live capacity must fail closed before last-5min data exists")


def test_execution_calendar_appends_live_date_and_is_required(tmp_path: Path):
    path = tmp_path / "calendar.csv"
    pd.DataFrame({"date": ["2026-07-20", "2026-07-21"]}).to_csv(path, index=False)
    dates = live.load_execution_calendar(path, pd.Timestamp("2026-07-22"))
    assert dates.strftime("%Y-%m-%d").tolist() == [
        "2026-07-20", "2026-07-21", "2026-07-22"
    ]
    try:
        live.load_execution_calendar(tmp_path / "missing.csv", pd.Timestamp("2026-07-22"))
    except FileNotFoundError as exc:
        assert "rerun the live pre stage" in str(exc)
    else:
        raise AssertionError("missing execution calendar must fail")


def test_prepare_builds_raw_daily_execution_calendar(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "live_prepare", ROOT / "pipelines" / "as1455_live_prepare.py"
    )
    assert spec and spec.loader
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)

    raw_dir = tmp_path / "raw"
    as1455_dir = tmp_path / "as1455"
    raw_dir.mkdir(); as1455_dir.mkdir()
    symbols = ["600001.SH", "000001.SZ"]
    raw_dates = ["2026-07-17", "2026-07-20", "2026-07-21"]
    for symbol in symbols:
        code = symbol[:6]
        pd.DataFrame({
            "date": raw_dates,
            "close": [10.0, 10.2, 10.3],
            "preclose": [9.9, 10.0, 10.2],
        }).to_csv(raw_dir / f"{code}_daily_raw.csv", index=False)
        pd.DataFrame({
            "date": raw_dates,
            "symbol": [symbol] * 3,
            "raw_open_as1455": [10.0, 10.1, 10.2],
            "raw_high_as1455": [10.1, 10.2, 10.3],
            "raw_low_as1455": [9.9, 10.0, 10.1],
            "raw_close_as1455": [10.0, 10.2, 10.3],
            "raw_volume_as1455": [1_000_000, 1_100_000, 1_200_000],
        }).to_csv(as1455_dir / f"{code}_as1455_daily.csv", index=False)
    universe = pd.DataFrame({"symbol": symbols})
    events = pd.DataFrame({"symbol": symbols, "event_ratio": [1.0, 1.0]})
    _raw, qfq, report, calendar = prepare.build_history_tails(
        universe, events, raw_dir, as1455_dir, "2026-07-21", 252
    )
    assert len(qfq) == 6
    assert report["status"].eq("ok").all()
    assert calendar.strftime("%Y-%m-%d").tolist() == raw_dates


def test_fast_finalizer_emits_exact_base_contract(tmp_path: Path):
    import subprocess
    from features.as1455_live_common import EXPECTED_MODEL_COLUMNS

    live_dir = tmp_path / "20260722"
    live_dir.mkdir()
    n, t = 20, 100
    symbols = np.array([f"600{i:03d}.SH" for i in range(1, n + 1)], dtype=object)
    x = np.arange(t, dtype=float)
    close = np.vstack([10 + i * 0.2 + 0.01 * x + 0.2 * np.sin(x / 7 + i) for i in range(n)])
    high, low, open_ = close * 1.01, close * 0.99, close * 0.995
    volume = np.vstack([1_000_000 + 1000 * x + 10_000 * i for i in range(n)])
    np.savez_compressed(
        live_dir / "06_live_feature_state_fast.npz",
        symbols=symbols,
        sectors=np.array([i // 5 for i in range(n)]),
        dates_last=np.array(["2026-07-21"] * n, dtype=object),
        row_counts=np.array([t] * n),
        feature_columns=np.array(EXPECTED_MODEL_COLUMNS, dtype=object),
        trade_date=np.array(["20260722"], dtype=object),
        source_path=np.array(["synthetic"], dtype=object),
        open=open_, high=high, low=low, close=close, volume=volume,
    )
    raw = []
    for i, symbol in enumerate(symbols):
        price = float(close[i, -1] * (1.001 + i * 1e-5))
        raw.append({
            "symbol": symbol,
            "raw_open_as1455": price * 0.995,
            "raw_high_as1455": price * 1.01,
            "raw_low_as1455": price * 0.99,
            "raw_close_as1455": price,
            "raw_volume_as1455": float(volume[i, -1]),
            "raw_amount_as1455": price * volume[i, -1],
            "live_preclose": float(close[i, -1]),
        })
    pd.DataFrame(raw).to_csv(live_dir / "08_live_raw_row_as1455.csv", index=False)
    pd.DataFrame([
        {"symbol": symbol, "event_ratio": 1.0, "is_factor_event_today": False}
        for symbol in symbols
    ]).to_csv(live_dir / "03_adjustment_events.csv", index=False)

    subprocess.run([
        sys.executable,
        str(ROOT / "features" / "finalize_as1455_live_features_fast.py"),
        "--trade-date", "20260722",
        "--live-dir", str(live_dir),
        "--min-feature-rows", str(n),
        "--max-elapsed-seconds", "100",
        "--allow-indicator-fallback",
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    result = pd.read_csv(live_dir / "11_live_model_features_for_prediction.csv")
    assert len(result) == n
    assert result.columns.tolist() == ["date", "symbol", *EXPECTED_MODEL_COLUMNS]
    assert np.isfinite(result[EXPECTED_MODEL_COLUMNS].to_numpy(dtype=float)).all()
