from __future__ import annotations

import json
from pathlib import Path

from dashboard.as1455_monitor_config import (
    DEFAULT_PRODUCTION_EXPERIMENT,
    load_monitor_experiments,
    save_monitor_experiments,
)


def test_monitor_defaults_to_production_only(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    assert load_monitor_experiments(root) == [DEFAULT_PRODUCTION_EXPERIMENT]


def test_monitor_always_keeps_production_experiment(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    available = [
        "r01_best_reb1_fold0_5_forward",
        DEFAULT_PRODUCTION_EXPERIMENT,
    ]
    saved = save_monitor_experiments(
        root,
        ["r01_best_reb1_fold0_5_forward"],
        available,
    )
    assert saved == [
        DEFAULT_PRODUCTION_EXPERIMENT,
        "r01_best_reb1_fold0_5_forward",
    ]
    payload = json.loads(
        (root / ".dashboard" / "user_config.json").read_text(encoding="utf-8")
    )
    assert payload["monitor_experiments"] == saved


def test_monitor_filters_unknown_names(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    root.mkdir()
    available = [DEFAULT_PRODUCTION_EXPERIMENT]
    saved = save_monitor_experiments(
        root,
        ["does_not_exist", DEFAULT_PRODUCTION_EXPERIMENT],
        available,
    )
    assert saved == [DEFAULT_PRODUCTION_EXPERIMENT]
