from __future__ import annotations

import json
import math
import re
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import fcntl
import numpy as np
import pandas as pd

from utils.as1455_tracking import (
    TRACKING_MATRIX_MANIFEST,
    TRACKING_MATRIX_SUMMARY,
    TRACKING_SEMANTICS_VERSION,
    experiment_tracking_paths,
    read_json,
    resolve_initial_cash,
    write_json,
)

DEFAULT_PRODUCTION_EXPERIMENT = "r21_best_reb21_fold0_4_forward"
CALIBRATION_ROOT = Path(".dashboard") / "manual_calibrations"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _normalize_symbol(value: object) -> str:
    match = re.search(r"(\d{6})", str(value))
    if match is None:
        raise ValueError(f"无法识别股票代码：{value!r}")
    code = match.group(1)
    return f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ"


def _calibration_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Shanghai"))


@contextmanager
def _exclusive_account_locks(matrix_root: Path, live_root: Path | None) -> Iterator[None]:
    lock_paths = [matrix_root / ".dashboard" / "refresh.lock"]
    if live_root is not None:
        lock_paths.append(Path(live_root) / ".dashboard" / "nine_strategy.lock")
    handles = []
    try:
        for path in sorted({p.resolve() for p in lock_paths}, key=str):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(
                    f"账户刷新或14:55任务正在运行，暂不能手动校准：{path}"
                ) from exc
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def load_manual_calibration_account(
    matrix_root: Path,
    experiment: str = DEFAULT_PRODUCTION_EXPERIMENT,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    root = Path(matrix_root).expanduser().resolve() / experiment
    paths = experiment_tracking_paths(root)
    state = read_json(paths["latest_state"], {}) or {}
    manifest = read_json(paths["manifest"], {}) or {}
    positions = _read_csv(paths["latest_positions"])
    return state, positions, manifest


def editable_positions(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "shares", "buy_date", "avg_entry_price"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    work = frame.copy()
    for column in columns:
        if column not in work.columns:
            work[column] = np.nan
    work = work[columns]
    work["buy_date"] = pd.to_datetime(work["buy_date"], errors="coerce").dt.date
    return work.reset_index(drop=True)


def _validated_positions(
    edited: pd.DataFrame,
    existing: pd.DataFrame,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    if edited is None:
        edited = pd.DataFrame()
    work = edited.copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "shares",
                "buy_date",
                "avg_entry_price",
                "entry_rank",
                "entry_score",
                "cost_basis_notional",
                "cost_basis_fee",
            ]
        )
    required = {"symbol", "shares", "buy_date", "avg_entry_price"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"持仓校准缺少字段：{sorted(missing)}")
    blank = work["symbol"].isna() | work["symbol"].astype(str).str.strip().eq("")
    if blank.any():
        raise ValueError("持仓表存在空股票代码；删除空白行后再保存")
    work["symbol"] = work["symbol"].map(_normalize_symbol)
    if work["symbol"].duplicated().any():
        dup = work.loc[work["symbol"].duplicated(), "symbol"].tolist()
        raise ValueError(f"持仓表存在重复股票：{dup}")
    shares = pd.to_numeric(work["shares"], errors="coerce")
    integer_mask = shares.map(lambda value: float(value).is_integer() if pd.notna(value) else False)
    if shares.isna().any() or (~shares.gt(0)).any() or (~integer_mask).any():
        raise ValueError("持仓 shares 必须全部为正整数；卖出零股可以是非100整数倍，但不能是小数股")
    work["shares"] = shares.astype(int)
    buy_dates = pd.to_datetime(work["buy_date"], errors="coerce").dt.normalize()
    if buy_dates.isna().any():
        raise ValueError("每个持仓都必须填写真实 buy_date，用于T+1约束")
    if buy_dates.gt(asof).any():
        raise ValueError("buy_date 不能晚于校准日期")
    work["buy_date"] = buy_dates.dt.strftime("%Y-%m-%d")
    prices = pd.to_numeric(work["avg_entry_price"], errors="coerce")
    if prices.isna().any() or (~prices.gt(0)).any():
        raise ValueError("每个持仓都必须填写正数 avg_entry_price")
    work["avg_entry_price"] = prices.astype(float)

    existing_map: dict[str, dict[str, Any]] = {}
    if existing is not None and not existing.empty and "symbol" in existing.columns:
        base = existing.copy()
        base["symbol"] = base["symbol"].map(_normalize_symbol)
        existing_map = {
            str(row["symbol"]): row.to_dict() for _, row in base.iterrows()
        }

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        symbol = str(row["symbol"])
        merged = dict(existing_map.get(symbol, {}))
        merged.update(
            {
                "symbol": symbol,
                "shares": int(row["shares"]),
                "buy_date": str(row["buy_date"]),
                "avg_entry_price": float(row["avg_entry_price"]),
            }
        )
        merged.setdefault("entry_rank", np.nan)
        merged.setdefault("entry_score", np.nan)
        merged["cost_basis_notional"] = float(row["shares"]) * float(row["avg_entry_price"])
        merged.setdefault("cost_basis_fee", 0.0)
        rows.append(merged)
    result = pd.DataFrame(rows)
    preferred = [
        "symbol",
        "shares",
        "buy_date",
        "avg_entry_price",
        "entry_rank",
        "entry_score",
        "cost_basis_notional",
        "cost_basis_fee",
    ]
    return result[[*preferred, *[c for c in result.columns if c not in preferred]]]


def _recompute_nav(nav: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    out = nav.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    values = pd.to_numeric(out["nav"], errors="raise")
    prior = values.shift(1)
    prior.iloc[0] = float(initial_cash)
    out["daily_return"] = values / prior - 1.0
    out["cumulative_return"] = values / float(initial_cash) - 1.0
    out["cumulative_return_pct"] = out["cumulative_return"] * 100.0
    out["drawdown"] = values / values.cummax() - 1.0
    return out.reset_index(drop=True)


def _summary_metrics(nav: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    if nav.empty:
        raise RuntimeError("tracking NAV为空，不能做手动校准")
    rets = pd.to_numeric(nav["daily_return"], errors="coerce").dropna()
    final_nav = float(pd.to_numeric(nav["nav"], errors="raise").iloc[-1])
    total_return = final_nav / float(initial_cash) - 1.0
    if total_return <= -1.0:
        annual_return = np.nan
    else:
        exponent = math.log1p(total_return) * 252.0 / max(len(nav), 1)
        annual_return = math.expm1(exponent) if exponent < 700 else np.inf
    vol = float(rets.std(ddof=0)) if len(rets) else np.nan
    return {
        "final_nav": final_nav,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(rets.mean() / vol * math.sqrt(252.0))
        if len(rets) and np.isfinite(vol) and vol > 0
        else np.nan,
        "max_drawdown": float(pd.to_numeric(nav["drawdown"], errors="coerce").min()),
        "forward_end": pd.Timestamp(nav["date"].iloc[-1]).strftime("%Y-%m-%d"),
        "forward_n_days": int(len(nav)),
        "initial_cash": float(initial_cash),
    }


def _patch_summary_file(path: Path, experiment: str, metrics: dict[str, Any]) -> None:
    frame = _read_csv(path)
    if frame.empty:
        return
    mask = (
        frame["experiment"].astype(str).eq(experiment)
        if "experiment" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    if not mask.any():
        return
    for key, value in metrics.items():
        if key in frame.columns:
            frame.loc[mask, key] = value
    _atomic_csv(frame, path)


def _archive_existing_ready(live_root: Path | None, experiment: str, calibration_id: str) -> str | None:
    if live_root is None:
        return None
    token = _calibration_now().strftime("%Y%m%d")
    nine_root = Path(live_root) / token / "nine_strategy"
    strategy_root = nine_root / "strategies" / experiment
    batch = strategy_root / "execution_batch.json"
    if not batch.is_file():
        return None
    archive = strategy_root / "_superseded_execution_batches"
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"{calibration_id}.json"
    batch.replace(target)

    manifest_file = strategy_root / "strategy_manifest.json"
    if manifest_file.is_file():
        payload = read_json(manifest_file, {}) or {}
        for key in ("execution_batch_file", "execution_batch_protocol", "execution_price_source"):
            payload.pop(key, None)
        payload["execution_batch_state"] = "invalidated_by_manual_calibration"
        write_json(manifest_file, payload)
    root_manifest_file = nine_root / "live_nine_strategy_manifest.json"
    if root_manifest_file.is_file():
        payload = read_json(root_manifest_file, {}) or {}
        payload.pop("execution_batch_protocol", None)
        payload["execution_batch_files"] = {}
        payload["execution_batch_state"] = "invalidated_by_manual_calibration"
        write_json(root_manifest_file, payload)
    return str(target)


def apply_manual_calibration(
    matrix_root: Path,
    *,
    experiment: str,
    asof_date: pd.Timestamp,
    strategy_cash: float,
    strategy_nav: float,
    positions: pd.DataFrame,
    note: str = "",
    live_root: Path | None = None,
) -> dict[str, Any]:
    matrix_root = Path(matrix_root).expanduser().resolve()
    live_root = Path(live_root).expanduser().resolve() if live_root is not None else None
    root = matrix_root / experiment
    paths = experiment_tracking_paths(root)
    expected_asof = pd.Timestamp(asof_date).normalize()
    cash = float(strategy_cash)
    nav_value = float(strategy_nav)
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("策略账户现金必须是非负有限数")
    if not math.isfinite(nav_value) or nav_value <= 0 or nav_value + 1e-6 < cash:
        raise ValueError("策略账户NAV必须为正，并且不能小于策略账户现金")

    with _exclusive_account_locks(matrix_root, live_root):
        state = read_json(paths["latest_state"], {}) or {}
        manifest = read_json(paths["manifest"], {}) or {}
        if manifest.get("status") != "ok" or state.get("status") != "ok":
            raise RuntimeError("r21_best tracking账户未就绪，不能手动校准")
        if int(manifest.get("tracking_semantics_version", 0) or 0) != TRACKING_SEMANTICS_VERSION:
            raise RuntimeError("tracking semantics已过期，请先完成账户重建")
        current_asof = pd.to_datetime(state.get("asof_date"), errors="coerce")
        if pd.isna(current_asof) or pd.Timestamp(current_asof).normalize() != expected_asof:
            raise RuntimeError(
                "只允许校准当前最新tracking日期："
                f"current={state.get('asof_date')} requested={expected_asof:%Y-%m-%d}"
            )
        initial_cash = resolve_initial_cash(root)
        state_initial_cash = pd.to_numeric(pd.Series([state.get("initial_cash")]), errors="coerce").iloc[0]
        if (
            pd.isna(state_initial_cash)
            or not math.isfinite(float(state_initial_cash))
            or abs(float(state_initial_cash) - initial_cash) > 1e-6
        ):
            raise RuntimeError("tracking初始资金口径已变化，请先重建账户再校准")

        existing_positions = _read_csv(paths["latest_positions"])
        calibrated_positions = _validated_positions(positions, existing_positions, expected_asof)
        if calibrated_positions.empty and abs(nav_value - cash) > 0.01:
            raise ValueError("空仓时策略NAV应与策略账户现金一致")

        nav = _read_csv(paths["nav"])
        if nav.empty or "date" not in nav.columns or "nav" not in nav.columns:
            raise RuntimeError("tracking NAV文件缺失或格式不完整")
        nav_dates = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
        mask = nav_dates.eq(expected_asof)
        if not mask.any():
            raise RuntimeError(f"tracking NAV中不存在校准日期 {expected_asof:%Y-%m-%d}")

        now = _calibration_now()
        calibration_id = now.strftime("cal_%Y%m%d_%H%M%S_%f")
        audit_dir = matrix_root / CALIBRATION_ROOT / experiment / calibration_id
        backup_dir = audit_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=False)
        backup_sources = [
            paths["latest_state"],
            paths["latest_positions"],
            paths["nav"],
            paths["positions"],
            paths["result"],
            paths["manifest"],
            matrix_root / TRACKING_MATRIX_SUMMARY,
            matrix_root / TRACKING_MATRIX_MANIFEST,
        ]
        for source in backup_sources:
            if source.is_file():
                shutil.copy2(source, backup_dir / source.name)

        # The account state is about to change. Any same-day READY order derived
        # from the previous state is invalid and must disappear before mutation.
        archived_batch = _archive_existing_ready(live_root, experiment, calibration_id)

        nav = nav.copy()
        nav.loc[mask, "nav"] = nav_value
        nav.loc[mask, "cash"] = cash
        nav.loc[mask, "n_positions"] = int(len(calibrated_positions))
        market_value = nav_value - cash
        for column in ("market_value", "positions_value", "stock_value"):
            if column in nav.columns:
                nav.loc[mask, column] = market_value
        nav = _recompute_nav(nav, initial_cash)

        position_history = _read_csv(paths["positions"])
        if not position_history.empty and "date" in position_history.columns:
            dates = pd.to_datetime(position_history["date"], errors="coerce").dt.normalize()
            position_history = position_history.loc[~dates.eq(expected_asof)].copy()
        else:
            position_history = pd.DataFrame()
        calibrated_history = calibrated_positions.copy()
        calibrated_history.insert(0, "date", expected_asof.strftime("%Y-%m-%d"))
        calibrated_history["tracking_bootstrap"] = False
        all_columns = list(dict.fromkeys([*position_history.columns, *calibrated_history.columns]))
        position_history = position_history.reindex(columns=all_columns)
        calibrated_history = calibrated_history.reindex(columns=all_columns)
        merged_positions = pd.concat([position_history, calibrated_history], ignore_index=True)
        if not merged_positions.empty and "date" in merged_positions.columns:
            merged_positions["date"] = pd.to_datetime(
                merged_positions["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        metrics = _summary_metrics(nav, initial_cash)
        state_before = dict(state)
        state.update(
            {
                "cash": cash,
                "nav": nav_value,
                "n_positions": int(len(calibrated_positions)),
                "account_state_source": "manual_broker_calibration",
                "manual_calibration_id": calibration_id,
                "manual_calibrated_at": now.isoformat(timespec="seconds"),
                "manual_calibration_note": str(note).strip() or None,
            }
        )
        manifest.update(
            {
                "last_manual_calibration_id": calibration_id,
                "last_manual_calibration_asof": expected_asof.strftime("%Y-%m-%d"),
                "last_manual_calibrated_at": now.isoformat(timespec="seconds"),
                "last_manual_calibration_note": str(note).strip() or None,
            }
        )

        _atomic_csv(nav, paths["nav"])
        _atomic_csv(calibrated_positions, paths["latest_positions"])
        _atomic_csv(merged_positions, paths["positions"])
        write_json(paths["latest_state"], state)
        write_json(paths["manifest"], manifest)
        _patch_summary_file(paths["result"], experiment, metrics)
        _patch_summary_file(matrix_root / TRACKING_MATRIX_SUMMARY, experiment, metrics)

        matrix_manifest_path = matrix_root / TRACKING_MATRIX_MANIFEST
        matrix_manifest = read_json(matrix_manifest_path, {}) or {}
        matrix_manifest["last_manual_calibration"] = {
            "experiment": experiment,
            "calibration_id": calibration_id,
            "asof_date": expected_asof.strftime("%Y-%m-%d"),
            "calibrated_at": now.isoformat(timespec="seconds"),
        }
        write_json(matrix_manifest_path, matrix_manifest)

        audit = {
            "status": "applied",
            "protocol": "as1455_manual_broker_calibration_v1",
            "calibration_id": calibration_id,
            "experiment": experiment,
            "asof_date": expected_asof.strftime("%Y-%m-%d"),
            "calibrated_at": now.isoformat(timespec="seconds"),
            "note": str(note).strip() or None,
            "initial_cash": initial_cash,
            "strategy_cash": cash,
            "strategy_nav": nav_value,
            "n_positions": int(len(calibrated_positions)),
            "state_before": state_before,
            "state_after": state,
            "archived_same_day_execution_batch": archived_batch,
            "backup_dir": str(backup_dir),
        }
        write_json(audit_dir / "calibration.json", audit)
        return audit


def calibration_history(
    matrix_root: Path,
    experiment: str = DEFAULT_PRODUCTION_EXPERIMENT,
    limit: int = 20,
) -> list[dict[str, Any]]:
    root = Path(matrix_root).expanduser().resolve() / CALIBRATION_ROOT / experiment
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for directory in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        payload = read_json(directory / "calibration.json", {}) or {}
        if payload:
            rows.append(payload)
        if len(rows) >= int(limit):
            break
    return rows
