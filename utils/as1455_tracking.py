from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

USER_CONFIG = Path(".dashboard") / "user_config.json"
TRACKING_SEMANTICS_VERSION = 4
DEFAULT_TRACKING_INITIAL_CASH = 120_000.0
TRACKING_MANIFEST = "tracking_forward_manifest.json"
TRACKING_RESULT = "tracking_forward_result.csv"
TRACKING_NAV = "tracking_forward_nav.csv"
TRACKING_ORDERS = "tracking_forward_orders.csv"
TRACKING_REJECTIONS = "tracking_forward_rejections.csv"
TRACKING_POSITIONS = "tracking_forward_positions.csv"
TRACKING_LATEST_STATE = "tracking_forward_latest_state.json"
TRACKING_LATEST_POSITIONS = "tracking_forward_latest_positions.csv"
TRACKING_MATRIX_SUMMARY = "tracking_matrix_summary.csv"
TRACKING_MATRIX_MANIFEST = "tracking_matrix_manifest.json"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def tracking_user_config(matrix_root: Path) -> dict[str, Any]:
    payload = read_json(Path(matrix_root) / USER_CONFIG, {}) or {}
    return payload if isinstance(payload, dict) else {}


def tracking_start_date(matrix_root: Path) -> pd.Timestamp | None:
    payload = tracking_user_config(matrix_root)
    value = payload.get("tracking_start_date")
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def tracking_initial_cash(
    matrix_root: Path,
    default: float = DEFAULT_TRACKING_INITIAL_CASH,
) -> float:
    """Return the user-configured strategy-account starting capital.

    This is deliberately separate from the immutable historical/strict-forward
    research run's ``initial_cash``.  Tracking/live accounts use this capital as
    their starting NAV, then compound naturally from their actual NAV: profits
    increase later position sizing and losses decrease it.
    """
    payload = tracking_user_config(matrix_root)
    raw = payload.get("tracking_initial_cash", default)
    value = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
    if pd.isna(value) or not float(value) > 0:
        raise ValueError(f"invalid tracking_initial_cash={raw!r}")
    return float(value)


def contiguous_tracking_dates(
    prediction_dates: pd.DatetimeIndex,
    execution_calendar: pd.DatetimeIndex,
    lower_bound: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return contiguous executable/predicted dates from ``lower_bound``.

    Dates before the first available prediction may be skipped (for example a
    weekend start). Once tracking starts, a missing prediction stops advancement
    so the account never jumps across an unprocessed trading day.
    """
    prediction_dates = pd.DatetimeIndex(prediction_dates).normalize().unique().sort_values()
    execution_calendar = pd.DatetimeIndex(execution_calendar).normalize().unique().sort_values()
    candidates = execution_calendar[execution_calendar >= pd.Timestamp(lower_bound).normalize()]
    if candidates.empty:
        return pd.DatetimeIndex([])
    available = set(prediction_dates)
    started = False
    selected: list[pd.Timestamp] = []
    for value in candidates:
        date = pd.Timestamp(value).normalize()
        if not started:
            if date not in available:
                continue
            started = True
            selected.append(date)
            continue
        if date not in available:
            break
        selected.append(date)
    return pd.DatetimeIndex(selected)


def experiment_tracking_paths(experiment_root: Path) -> dict[str, Path]:
    return {
        "manifest": experiment_root / TRACKING_MANIFEST,
        "result": experiment_root / TRACKING_RESULT,
        "nav": experiment_root / TRACKING_NAV,
        "orders": experiment_root / TRACKING_ORDERS,
        "rejections": experiment_root / TRACKING_REJECTIONS,
        "positions": experiment_root / TRACKING_POSITIONS,
        "latest_state": experiment_root / TRACKING_LATEST_STATE,
        "latest_positions": experiment_root / TRACKING_LATEST_POSITIONS,
    }


def resolve_initial_cash(
    experiment_root: Path,
    default: float = DEFAULT_TRACKING_INITIAL_CASH,
) -> float:
    """Resolve tracking/live starting capital from the dashboard account config.

    ``experiment_root`` is one strategy directory directly under the nine-strategy
    matrix root, so its parent owns ``.dashboard/user_config.json``.  Historical
    Fold/Grid and canonical strict-forward artifacts retain their original frozen
    research capital and are never rewritten by this helper.
    """
    return tracking_initial_cash(Path(experiment_root).parent, default=default)


def tracking_manifest_matches(experiment_root: Path, start: pd.Timestamp) -> bool:
    payload = read_json(experiment_root / TRACKING_MANIFEST, {}) or {}
    try:
        expected_cash = resolve_initial_cash(experiment_root)
        actual_cash = float(payload.get("initial_cash"))
    except (TypeError, ValueError):
        return False
    return (
        payload.get("status") == "ok"
        and payload.get("tracking_start_date") == start.strftime("%Y-%m-%d")
        and int(payload.get("tracking_semantics_version", 0) or 0)
        == TRACKING_SEMANTICS_VERSION
        and abs(actual_cash - expected_cash) <= 1e-6
    )


def load_latest_tracking_state(
    experiment_root: Path,
    start: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = experiment_tracking_paths(experiment_root)
    manifest = read_json(paths["manifest"], {}) or {}
    if manifest.get("tracking_start_date") != start.strftime("%Y-%m-%d"):
        raise RuntimeError(
            f"tracking start mismatch for {experiment_root.name}: "
            f"expected={start:%Y-%m-%d} actual={manifest.get('tracking_start_date')}"
        )
    if int(manifest.get("tracking_semantics_version", 0) or 0) != TRACKING_SEMANTICS_VERSION:
        raise RuntimeError(
            f"tracking semantics are stale for {experiment_root.name}: "
            f"expected={TRACKING_SEMANTICS_VERSION} "
            f"actual={manifest.get('tracking_semantics_version')}"
        )
    expected_cash = resolve_initial_cash(experiment_root)
    actual_cash = pd.to_numeric(
        pd.Series([manifest.get("initial_cash")]), errors="coerce"
    ).iloc[0]
    if pd.isna(actual_cash) or abs(float(actual_cash) - expected_cash) > 1e-6:
        raise RuntimeError(
            f"tracking initial cash is stale for {experiment_root.name}: "
            f"expected={expected_cash:.2f} actual={manifest.get('initial_cash')}"
        )
    state = read_json(paths["latest_state"], {}) or {}
    if not state:
        raise FileNotFoundError(paths["latest_state"])
    state_cash = pd.to_numeric(
        pd.Series([state.get("initial_cash")]), errors="coerce"
    ).iloc[0]
    if pd.isna(state_cash) or abs(float(state_cash) - expected_cash) > 1e-6:
        raise RuntimeError(
            f"tracking state initial cash is stale for {experiment_root.name}: "
            f"expected={expected_cash:.2f} actual={state.get('initial_cash')}"
        )
    if paths["latest_positions"].is_file():
        try:
            positions = pd.read_csv(paths["latest_positions"], encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            positions = pd.DataFrame()
    else:
        positions = pd.DataFrame()
    return state, positions
