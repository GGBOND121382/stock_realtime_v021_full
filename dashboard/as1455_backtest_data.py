#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only data access for the AS1455 nine-experiment dashboard."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

EXPERIMENT_RE = re.compile(
    r"^(?P<target>r\d{2})_(?P<signal>all5|first3|best)_"
    r"reb(?P<rebalance>\d+)_(?P<fold_label>fold0_[45])_forward$"
)
SIGNAL_LABELS = {
    "all5": "Top-5 等权 Ensemble",
    "first3": "Top-3 等权 Ensemble",
    "best": "最优单模型",
}
TARGET_LABELS = {"r01": "1日目标", "r05": "5日目标", "r21": "21日目标"}


@dataclass(frozen=True)
class ExperimentIdentity:
    name: str
    target: str
    target_col: str
    signal: str
    rebalance_every: int
    fold_label: str

    @property
    def target_label(self) -> str:
        return TARGET_LABELS.get(self.target, self.target)

    @property
    def signal_label(self) -> str:
        return SIGNAL_LABELS.get(self.signal, self.signal)

    @property
    def display_name(self) -> str:
        return f"{self.target_label} · {self.signal_label}"


def parse_experiment_name(name: str) -> ExperimentIdentity:
    match = EXPERIMENT_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"unsupported experiment directory name: {name}")
    target = match.group("target")
    return ExperimentIdentity(
        name=name,
        target=target,
        target_col=f"{target}_fwd",
        signal=match.group("signal"),
        rebalance_every=int(match.group("rebalance")),
        fold_label=match.group("fold_label"),
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)


def discover_experiment_names(matrix_root: Path) -> list[str]:
    expected_file = matrix_root / "expected_experiments.txt"
    if expected_file.is_file():
        names = [
            line.strip()
            for line in expected_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif matrix_root.is_dir():
        names = [
            item.name
            for item in matrix_root.iterdir()
            if item.is_dir() and EXPERIMENT_RE.fullmatch(item.name)
        ]
    else:
        names = []
    valid: list[str] = []
    for name in names:
        try:
            parse_experiment_name(name)
        except ValueError:
            continue
        if (matrix_root / name).is_dir():
            valid.append(name)
    signal_order = {"all5": 0, "first3": 1, "best": 2}
    return sorted(
        dict.fromkeys(valid),
        key=lambda name: (
            parse_experiment_name(name).rebalance_every,
            signal_order[parse_experiment_name(name).signal],
        ),
    )


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_matrix_summary(matrix_root: Path) -> pd.DataFrame:
    summary = read_csv(matrix_root / "fixed_signal_matrix_summary.csv")
    if summary.empty:
        rows: list[dict[str, Any]] = []
        for name in discover_experiment_names(matrix_root):
            result = read_csv(matrix_root / name / "strict_forward_result.csv")
            if result.empty:
                continue
            row = result.iloc[0].to_dict()
            row["experiment"] = name
            rows.append(row)
        summary = pd.DataFrame(rows)
    if summary.empty or "experiment" not in summary.columns:
        return pd.DataFrame()

    identities: list[ExperimentIdentity | None] = []
    for value in summary["experiment"].astype(str):
        try:
            identities.append(parse_experiment_name(value))
        except ValueError:
            identities.append(None)

    summary = summary.copy()
    summary["target"] = [x.target if x else None for x in identities]
    summary["target_label"] = [x.target_label if x else None for x in identities]
    summary["signal"] = [x.signal if x else None for x in identities]
    summary["signal_label"] = [x.signal_label if x else None for x in identities]
    summary["display_name"] = [
        x.display_name if x else str(name)
        for x, name in zip(identities, summary["experiment"])
    ]
    if "rebalance_every" not in summary.columns:
        summary["rebalance_every"] = [
            x.rebalance_every if x else None for x in identities
        ]
    if "forward_start" not in summary.columns and "strict_forward_start" in summary.columns:
        summary["forward_start"] = summary["strict_forward_start"]
    if "forward_end" not in summary.columns and "strict_forward_end" in summary.columns:
        summary["forward_end"] = summary["strict_forward_end"]
    summary = _coerce_numeric(
        summary,
        [
            "rebalance_every", "total_return", "annual_return", "sharpe",
            "max_drawdown", "final_nav", "forward_n_days", "max_positions",
            "sell_rank", "historical_offset", "effective_forward_offset",
        ],
    )
    summary["signal_order"] = summary["signal"].map(
        {"all5": 0, "first3": 1, "best": 2}
    )
    return summary.sort_values(
        ["rebalance_every", "signal_order"], kind="stable"
    ).drop(columns=["signal_order"]).reset_index(drop=True)


def _resolve_manifest_path(value: Any, experiment_root: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        project_root = experiment_root.parents[4] if len(experiment_root.parents) >= 5 else Path.cwd()
        candidates.extend([experiment_root / path, project_root / path])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists():
            return resolved
    return None


def find_forward_run_dir(experiment_root: Path, manifest: dict[str, Any]) -> Path | None:
    direct = _resolve_manifest_path(manifest.get("strict_forward_run_dir"), experiment_root)
    if direct and direct.is_dir():
        return direct
    strict_manifest_path = (
        experiment_root / "strict_oos_forward" / "01_close_auction_grid" / "strict_oos_manifest.json"
    )
    strict = read_json(strict_manifest_path, {}) or {}
    run_name = strict.get("retained_run_name")
    if run_name:
        candidate = strict_manifest_path.parent / "01_runs" / str(run_name)
        if candidate.is_dir():
            return candidate
        matches = list((experiment_root / "strict_oos_forward").glob(f"**/01_runs/{run_name}"))
        if matches:
            return matches[0]
    return None


def find_historical_run_dir(experiment_root: Path, manifest: dict[str, Any]) -> Path | None:
    direct = _resolve_manifest_path(manifest.get("historical_selected_run_dir"), experiment_root)
    return direct if direct and direct.is_dir() else None


def normalize_nav(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty or not {"date", "nav"}.issubset(nav.columns):
        return pd.DataFrame()
    work = nav.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["nav"] = pd.to_numeric(work["nav"], errors="coerce")
    work = (
        work.dropna(subset=["date", "nav"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if work.empty:
        return work
    initial = float(work.iloc[0]["nav"])
    if not math.isfinite(initial) or abs(initial) < 1e-12:
        return pd.DataFrame()
    work["cumulative_return"] = work["nav"] / initial - 1.0
    work["cumulative_return_pct"] = work["cumulative_return"] * 100.0
    work["daily_return"] = (
        pd.to_numeric(work["daily_return"], errors="coerce")
        if "daily_return" in work.columns else work["nav"].pct_change()
    )
    work["drawdown"] = work["nav"] / work["nav"].cummax() - 1.0
    return work


def load_experiment(matrix_root: Path, name: str) -> dict[str, Any]:
    identity = parse_experiment_name(name)
    root = matrix_root / name
    manifest = read_json(root / "global_fold0_to_fold5_forward_manifest.json", {}) or {}
    forward_run = find_forward_run_dir(root, manifest)
    historical_run = find_historical_run_dir(root, manifest)
    return {
        "identity": identity,
        "root": root,
        "manifest": manifest,
        "result": read_csv(root / "strict_forward_result.csv"),
        "forward_run": forward_run,
        "historical_run": historical_run,
        "forward_nav": normalize_nav(read_csv(forward_run / "close_auction_nav.csv") if forward_run else pd.DataFrame()),
        "historical_nav": normalize_nav(read_csv(historical_run / "close_auction_nav.csv") if historical_run else pd.DataFrame()),
        "fold_returns": read_csv(root / "historical_fold_segment_returns.csv"),
        "rebalance_dates": read_csv(root / "strict_forward_rebalance_dates.csv"),
        "positions": read_csv(root / "strict_forward_rebalance_positions.csv"),
        "positions_manifest": read_json(root / "strict_forward_rebalance_positions_manifest.json", {}) or {},
        "top20": read_csv(root / "historical_grid_top20.csv"),
    }


def build_forward_comparison(matrix_root: Path, experiment_names: Iterable[str]) -> pd.DataFrame:
    series: list[pd.Series] = []
    for name in experiment_names:
        item = load_experiment(matrix_root, name)
        nav = item["forward_nav"]
        if nav.empty:
            continue
        series.append(
            nav.set_index("date")["cumulative_return_pct"].rename(item["identity"].display_name)
        )
    return pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()


def load_refresh_status(matrix_root: Path) -> dict[str, Any]:
    status_file = matrix_root / ".dashboard" / "refresh_status.json"
    payload = read_json(status_file, {}) or {}
    payload["status_file"] = str(status_file)
    log_file = payload.get("log_file")
    if log_file:
        path = Path(str(log_file)).expanduser()
        if not path.is_absolute():
            path = matrix_root / ".dashboard" / path
        payload["resolved_log_file"] = str(path)
    return payload


def tail_text(path: Path, max_lines: int = 200, max_bytes: int = 256_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        text = stream.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-max_lines:])
