from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.as1455_cash_replay import _score_predictions, _segment_summary


def test_cash_replay_signal_specs_match_formal_strategies() -> None:
    index = pd.MultiIndex.from_tuples(
        [("600000.SH", pd.Timestamp("2026-01-05"))],
        names=["symbol", "date"],
    )
    frame = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        index=index,
        columns=[0, 1, 2, 3, 4],
    )
    assert float(_score_predictions(frame, "best").iloc[0]) == 1.0
    assert float(_score_predictions(frame, "first3").iloc[0]) == 2.0
    assert float(_score_predictions(frame, "all5").iloc[0]) == 3.0


def test_segment_summary_reports_peak_and_fails_closed_only_for_affected_segment() -> None:
    catalog = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
            ),
            "segment": ["fold1", "fold1", "strict_forward", "strict_forward"],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-07", "2026-01-08"]),
            "buy_amount": [10_000.0, 20_000.0, 5_000.0],
            "conservative_cash_required": [11_000.0, np.nan, 5_500.0],
            "cash_requirement_complete": [True, False, True],
        }
    )
    summary = _segment_summary(catalog, daily).set_index("segment")
    assert float(summary.loc["fold1", "peak_cash_required"]) == 11_000.0
    assert bool(summary.loc["fold1", "complete"])
    assert pd.isna(summary.loc["strict_forward", "peak_cash_required"])
    assert not bool(summary.loc["strict_forward", "complete"])
