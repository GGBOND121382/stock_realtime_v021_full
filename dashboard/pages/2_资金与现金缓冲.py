from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.as1455_backtest_data import (  # noqa: E402
    discover_experiment_names,
    load_experiment,
    load_tracking_matrix_summary,
    parse_experiment_name,
)
from dashboard.as1455_cash_history import (  # noqa: E402
    current_position_cash_estimate,
    strategy_cash_peak_report,
)
from utils.as1455_tracking import (  # noqa: E402
    DEFAULT_TRACKING_INITIAL_CASH,
    TRACKING_MATRIX_MANIFEST,
    TRACKING_SEMANTICS_VERSION,
    USER_CONFIG,
    read_json,
    tracking_initial_cash,
    write_json,
)

DEFAULT_MATRIX_ROOT = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_matrix"
    / "refresh_all_v1"
)
DEFAULT_LIVE_ROOT = PROJECT_ROOT / "saved_data" / "ashare_ml4t" / "live_as1455"
DEFAULT_RAW_DAILY = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch12_as1455"
    / "baostock_raw_daily_cache"
)
REFRESH_SCRIPT = PROJECT_ROOT / "scripts" / "run_as1455_dashboard_refresh.sh"
START_PLAN_MANIFEST = Path(".dashboard") / "start_date_plan_manifest.json"

st.set_page_config(page_title="AS1455 资金与现金缓冲", page_icon="💰", layout="wide")


def money(value: Any) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(parsed) else f"{float(parsed):,.0f} 元"


def _atomic_stale(path: Path, *, reason: str, initial_cash: float) -> None:
    payload = read_json(path, {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    payload.update(
        {
            "status": "stale_account_config",
            "stale_reason": reason,
            "expected_tracking_initial_cash": float(initial_cash),
            "invalidated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(path, payload)


def _start_rebuild(
    matrix_root: Path,
    live_root: Path,
    start: pd.Timestamp,
) -> tuple[bool, str]:
    env = os.environ.copy()
    env.update(
        {
            "MATRIX_ROOT": str(matrix_root),
            "LIVE_ROOT": str(live_root),
            "SKIP_DATA_REFRESH": "1",
            "TRACKING_MODE": "rebuild",
            "TRACKING_START_DATE": start.strftime("%Y-%m-%d"),
        }
    )
    try:
        subprocess.Popen(
            ["bash", str(REFRESH_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return False, str(exc)
    return True, "已提交后台重建；不重跑模型、Fold 或 Grid。"


matrix_root = Path(
    st.sidebar.text_input(
        "9组策略目录",
        value=os.environ.get("AS1455_MATRIX_ROOT", str(DEFAULT_MATRIX_ROOT)),
    )
).expanduser().resolve()
live_root = Path(
    st.sidebar.text_input(
        "盯盘目录",
        value=os.environ.get("AS1455_LIVE_ROOT", str(DEFAULT_LIVE_ROOT)),
    )
).expanduser().resolve()
raw_daily_root = Path(
    st.sidebar.text_input(
        "历史日线缓存",
        value=os.environ.get("AS1455_RAW_DAILY_CACHE_DIR", str(DEFAULT_RAW_DAILY)),
    )
).expanduser().resolve()

st.title("AS1455 策略资金与现金缓冲")
st.caption(
    "策略初始资金只定义 tracking/live 账户的起始 NAV；之后仓位按实际 NAV 自然复利，"
    "盈利扩大、亏损缩小。券商账户额外现金不进入策略 sizing。"
)

if not matrix_root.is_dir():
    st.error(f"策略目录不存在：{matrix_root}")
    st.stop()

config_file = matrix_root / USER_CONFIG
config = read_json(config_file, {}) or {}
raw_start = pd.to_datetime(config.get("tracking_start_date"), errors="coerce")
start = None if pd.isna(raw_start) else pd.Timestamp(raw_start).normalize()
try:
    configured_cash = tracking_initial_cash(matrix_root)
except ValueError:
    configured_cash = DEFAULT_TRACKING_INITIAL_CASH

st.subheader("策略初始资金")
if start is None:
    st.warning("尚未设置开始持仓日。请先在主看板设置开始持仓 / 收益起算日。")

with st.form("tracking_capital_form"):
    cash_input = st.number_input(
        "策略初始资金（元）",
        min_value=10_000.0,
        max_value=10_000_000.0,
        value=float(configured_cash),
        step=10_000.0,
        format="%.0f",
        help=(
            "仅用于9个 tracking/live 策略账户从空仓开始时的初始 NAV。"
            "之后全部按实际 NAV 复利，不设置盈利后的持仓金额上限。"
        ),
    )
    submitted = st.form_submit_button(
        "保存并按当前起算日重建9策略账户",
        type="primary",
        disabled=start is None,
    )

if submitted and start is not None:
    new_cash = float(cash_input)
    changed = abs(new_cash - float(configured_cash)) > 1e-6
    config = dict(config)
    config["tracking_initial_cash"] = new_cash
    config["tracking_semantics_version"] = TRACKING_SEMANTICS_VERSION
    config["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(config_file, config)
    if changed:
        _atomic_stale(
            matrix_root / TRACKING_MATRIX_MANIFEST,
            reason="tracking_initial_cash_changed",
            initial_cash=new_cash,
        )
        _atomic_stale(
            matrix_root / START_PLAN_MANIFEST,
            reason="tracking_initial_cash_changed",
            initial_cash=new_cash,
        )
    ok, message = _start_rebuild(matrix_root, live_root, start)
    (st.success if ok else st.error)(message)
    configured_cash = new_cash

tracking_summary, tracking_manifest = load_tracking_matrix_summary(matrix_root)
completed = int(tracking_manifest.get("completed_experiment_count", 0) or 0)
tracking_status = str(tracking_manifest.get("status", "missing"))
summary_cash = (
    pd.to_numeric(tracking_summary.get("initial_cash"), errors="coerce")
    if not tracking_summary.empty and "initial_cash" in tracking_summary.columns
    else pd.Series(dtype=float)
)
matching = int(
    np.isclose(summary_cash.dropna(), configured_cash, rtol=0, atol=1e-6).sum()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("当前配置", money(configured_cash))
c2.metric("Tracking 状态", tracking_status)
c3.metric("已完成策略", f"{completed}/9")
c4.metric("资金口径匹配", f"{matching}/9" if len(summary_cash) else "—")

st.caption(
    "修改资金后会重建 tracking 账户和已有14:55计划，但复用已有预测与交易参数；"
    "历史 Fold/Grid、模型 checkpoint、canonical strict-forward 研究结果保持不变。"
)

st.divider()
st.subheader("单策略历史 / Forward 涨停价最大占款")
experiment_names = discover_experiment_names(matrix_root)
if not experiment_names:
    st.info("未发现9个正式策略目录。")
    st.stop()

label_to_name = {
    parse_experiment_name(name).display_name: name for name in experiment_names
}
selected_label = st.selectbox("选择策略", list(label_to_name))
selected_name = label_to_name[selected_label]
item = load_experiment(matrix_root, selected_name)

with st.spinner("统计历史和 forward 每日涨停价占款……"):
    report = strategy_cash_peak_report(
        item,
        live_root=live_root,
        raw_daily_cache_dir=raw_daily_root,
    )
    current_est = current_position_cash_estimate(
        item,
        live_root=live_root,
        raw_daily_cache_dir=raw_daily_root,
        historical_orders=report["historical_orders"],
        forward_orders=report["forward_orders"],
    )

reference_values = [
    value
    for value in (
        report.get("overall_peak"),
        current_est.get("estimated_cash_required"),
    )
    if pd.notna(value)
]
reference_peak = max(map(float, reference_values)) if reference_values else np.nan

m1, m2, m3, m4 = st.columns(4)
m1.metric("历史最高涨停价占款", money(report.get("historical_peak")))
m2.metric("Strict Forward最高", money(report.get("forward_peak")))
m3.metric("当前持仓等股数估计", money(current_est.get("estimated_cash_required")))
m4.metric("历史/当前最高参考", money(reference_peak))

hist_date = report.get("historical_peak_date")
fwd_date = report.get("forward_peak_date")
st.caption(
    f"历史峰值日期={pd.Timestamp(hist_date).strftime('%Y-%m-%d') if pd.notna(hist_date) else '—'}；"
    f"Forward峰值日期={pd.Timestamp(fwd_date).strftime('%Y-%m-%d') if pd.notna(fwd_date) else '—'}；"
    f"当前持仓估计基准日={current_est.get('asof_date') or '—'}。"
)

segments = report["segments"].copy()
if not segments.empty:
    for column in ("start_date", "end_date", "peak_date"):
        if column in segments.columns:
            segments[column] = pd.to_datetime(
                segments[column], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    if "peak_cash_required" in segments.columns:
        segments["peak_cash_required"] = pd.to_numeric(
            segments["peak_cash_required"], errors="coerce"
        ).map(money)
    segments = segments.rename(
        columns={
            "segment": "区间",
            "start_date": "开始",
            "end_date": "结束",
            "n_days": "交易日",
            "n_buy_days": "有买单日",
            "peak_cash_required": "最高涨停价占款",
            "peak_date": "峰值日期",
            "complete": "数据完整",
        }
    )
    st.dataframe(segments, hide_index=True, use_container_width=True)

st.caption(
    "历史/forward指标按每天实际计划买入股数，以当日涨停价重新计价并加保守手续费预留，"
    "不抵扣同日卖出回款。旧日期没有14:55 sidecar时，从BaoStock raw-daily preclose还原主板非ST的涨停价。"
)
st.caption(
    "“当前持仓等股数估计”不是预测下一次一定会买哪些股票；它把当前持仓的相同股数按最新可得涨停价重新计价，"
    "手续费使用该策略历史/forward/tracking中观察到的最高有效买入费率，仅用于准备现金规模的保守参考。"
)

if st.button("重新读取", use_container_width=False):
    st.cache_data.clear()
    st.rerun()
