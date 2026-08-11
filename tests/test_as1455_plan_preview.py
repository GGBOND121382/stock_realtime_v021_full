from __future__ import annotations

import pandas as pd

from dashboard import as1455_plan_preview as preview


class FakeSelection:
    pass


def _predictions() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [
            ["000001.SZ", "000002.SZ", "000003.SZ"],
            pd.to_datetime(["2026-08-07", "2026-08-10", "2026-08-10"]),
        ],
        names=["symbol", "date"],
    )
    return pd.DataFrame(
        {
            0: [0.10, 0.30, 0.20],
            1: [0.0, 0.0, 0.0],
            2: [0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0],
        },
        index=index,
    )


def test_rank_table_filters_datetimeindex_with_vector_mask(monkeypatch) -> None:
    monkeypatch.setattr(
        preview.live,
        "score_predictions",
        lambda frame, selection: frame[0].rename("score"),
    )

    ranked = preview._rank_table(
        _predictions(), FakeSelection(), pd.Timestamp("2026-08-10")
    )

    assert ranked["rank"].tolist() == [1, 2]
    assert ranked["symbol"].tolist() == ["000002.SZ", "000003.SZ"]
    assert pd.to_datetime(ranked["date"]).dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-10",
        "2026-08-10",
    ]


def test_rank_table_missing_date_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        preview.live,
        "score_predictions",
        lambda frame, selection: frame[0].rename("score"),
    )

    ranked = preview._rank_table(
        _predictions(), FakeSelection(), pd.Timestamp("2026-08-11")
    )

    assert ranked.empty
    assert list(ranked.columns) == ["rank", "symbol", "date", "score"]
