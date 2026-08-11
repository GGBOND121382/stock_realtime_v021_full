#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrated AS1455 dashboard: tracking accounts + 14:55 nine-strategy monitor."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.as1455_backtest_data import (  # noqa: E402
    SIGNAL_LABELS,
    TARGET_LABELS,
    discover_experiment_names,
    load_experiment,
    load_matrix_summary,
    load_refresh_status,
    load_tracking_matrix_summary,
    tail_text,
)
from dashboard.as1455_live_data import (  # noqa: E402
    discover_live_dates,
    load_job_status,
    load_live_day,
    load_strategy,
)
from dashboard.as1455_plan_preview import preview_nine_strategy_day  # noqa: E402
from utils.as1455_tracking import TRACKING_SEMANTICS_VERSION  # noqa: E402

DEFAULT_MATRIX_ROOT = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_matrix"
    / "refresh_all_v1"
)
DEFAULT_LIVE_ROOT = PROJECT_ROOT / "saved_data" / "ashare_ml4t" / "live_as1455"
REFRESH_SCRIPT = PROJECT_ROOT / "scripts" / "run_as1455_dashboard_refresh.sh"
LIVE_JOB_SCRIPT = PROJECT_ROOT / "scripts" / "run_as1455_live_nine_strategy_job.sh"

st.set_page_config(
    page_title="AS1455 策略看板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.15rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.status-running {padding:.65rem .9rem;border-left:4px solid #f39c12;background:rgba(243,156,18,.08);}
.status-failed {padding:.65rem .9rem;border-left:4px solid #c0392b;background:rgba(192,57,43,.08);}
.strategy-action {padding:.55rem .8rem;border-left:4px solid #2980b9;background:rgba(41,128,185,.08);}
</style>
""",
    unsafe_allow_html=True,
)


def pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if pd.isna(value) else f"{value * 100:+.2f}%"


def number(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def process_is_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
        return False


def effective_state(status: dict[str, Any]) -> str:
    state = str(status.get("status", "idle"))
    if state == "running" and not process_is_alive(status.get("pid")):
        return "stale"
    return state


def start_background(command: list[str], env_extra: dict[str, str]) -> tuple[bool, str]:
    env = os.environ.copy()
    env.update(env_extra)
    try:
        subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return False, str(exc)
    return True, "任务已提交到后台；页面不会阻塞。"


def strategy_labels(summary: pd.DataFrame) -> dict[str, str]:
    return {
        str(row["experiment"]): str(row.get("display_name", row["experiment"]))
        for _, row in summary.iterrows()
    }


def load_curve_bounds(matrix_root: Path, names: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    firsts: list[pd.Timestamp] = []
    lasts: list[pd.Timestamp] = []
    for name in names:
        nav = load_experiment(matrix_root, name)["forward_nav"]
        if not nav.empty:
            firsts.append(pd.Timestamp(nav["date"].min()).normalize())
            lasts.append(pd.Timestamp(nav["date"].max()).normalize())
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    return (min(firsts), max(max(lasts), now)) if firsts and lasts else (now, now)


def tracking_ready_for_start(
    summary: pd.DataFrame,
    manifest: dict[str, Any],
    start: pd.Timestamp,
) -> bool:
    return (
        manifest.get("status") == "ok"
        and manifest.get("tracking_start_date") == start.strftime("%Y-%m-%d")
        and int(manifest.get("tracking_semantics_version", 0) or 0)
        == TRACKING_SEMANTICS_VERSION
        and int(manifest.get("completed_experiment_count", 0)) == 9
        and len(summary) == 9
        and (
            "status" not in summary.columns
            or summary["status"].astype(str).eq("ok").all()
        )
    )


def summary_view(summary: pd.DataFrame) -> None:
    view = summary.copy()
    columns = {
        "display_name": "实验",
        "rebalance_every": "调仓周期",
        "historical_offset": "历史offset",
        "total_return": "跟踪收益",
        "annual_return": "年化收益",
        "sharpe": "Sharpe",
        "max_drawdown": "最大回撤",
        "forward_start": "账户起点",
        "first_entry_date": "首次建仓",
        "forward_end": "最新日期",
        "historical_result_reused": "历史Grid复用",
    }
    keep = [column for column in columns if column in view.columns]
    view = view[keep].rename(columns=columns)
    for column in ("跟踪收益", "年化收益", "最大回撤"):
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(pct)
    if "Sharpe" in view.columns:
        view["Sharpe"] = pd.to_numeric(view["Sharpe"], errors="coerce").map(number)
    if "调仓周期" in view.columns:
        view["调仓周期"] = pd.to_numeric(view["调仓周期"], errors="coerce").map(
            lambda x: f"每{int(x)}日" if pd.notna(x) else "—"
        )
    st.dataframe(view, hide_index=True, use_container_width=True)


def order_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    preferred = [
        "symbol",
        "side",
        "shares",
        "filled_shares",
        "raw_exec_price",
        "raw_close_1500",
        "notional",
        "rank",
        "score",
        "reason",
        "order_status",
        "partial_fill",
        "position_before",
        "position_after",
    ]
    keep = [column for column in preferred if column in frame.columns]
    return frame[keep].copy() if keep else frame.copy()


def tracking_date_sets(
    matrix_root: Path,
    names: list[str],
    start: pd.Timestamp,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    details: dict[str, dict[str, Any]] = {}
    date_sets: list[set[str]] = []
    for name in names:
        item = load_experiment(matrix_root, name)
        details[name] = item
        manifest = item["tracking_manifest"]
        nav = item["tracking_nav"]
        if (
            manifest.get("status") != "ok"
            or manifest.get("tracking_start_date") != start.strftime("%Y-%m-%d")
            or int(manifest.get("tracking_semantics_version", 0) or 0)
            != TRACKING_SEMANTICS_VERSION
            or nav.empty
        ):
            return set(), details
        date_sets.append(set(pd.to_datetime(nav["date"]).dt.strftime("%Y%m%d")))
    return (set.intersection(*date_sets) if date_sets else set()), details


def tracking_day_summary(
    details: dict[str, dict[str, Any]],
    names: list[str],
    labels: dict[str, str],
    date_token: str,
) -> pd.DataFrame:
    target_date = pd.to_datetime(date_token, format="%Y%m%d").normalize()
    rows: list[dict[str, Any]] = []
    for name in names:
        item = details[name]
        nav_day = item["tracking_nav"].loc[
            pd.to_datetime(item["tracking_nav"]["date"]).dt.normalize().eq(target_date)
        ]
        if nav_day.empty:
            continue
        nav_row = nav_day.iloc[-1]
        orders = item["tracking_orders"]
        if not orders.empty:
            orders = orders.loc[
                pd.to_datetime(orders["date"]).dt.normalize().eq(target_date)
            ]
        rejections = item["tracking_rejections"]
        if not rejections.empty:
            rejections = rejections.loc[
                pd.to_datetime(rejections["date"]).dt.normalize().eq(target_date)
            ]
        positions = item["tracking_positions"]
        if not positions.empty:
            positions = positions.loc[
                pd.to_datetime(positions["date"]).dt.normalize().eq(target_date)
            ]
        is_rebalance = as_bool(nav_row.get("is_rebalance_day", False))
        buys = (
            int(orders["side"].astype(str).str.lower().eq("buy").sum())
            if not orders.empty and "side" in orders
            else 0
        )
        sells = (
            int(orders["side"].astype(str).str.lower().eq("sell").sum())
            if not orders.empty and "side" in orders
            else 0
        )
        bootstrap = as_bool(nav_row.get("tracking_bootstrap", False))
        n_positions = int(nav_row.get("n_positions", 0) or 0)
        action = (
            "首次建仓"
            if bootstrap
            else "调仓"
            if is_rebalance and len(orders)
            else "调仓日·无需成交"
            if is_rebalance
            else "非调仓日·保持空仓"
            if n_positions == 0
            else "非调仓日·继续持有"
        )
        rows.append(
            {
                "experiment": name,
                "策略": labels.get(name, name),
                "action": action,
                "is_rebalance_day": is_rebalance,
                "是否调仓": "是" if is_rebalance else "否",
                "planned_orders": int(len(orders)),
                "planned_buys": buys,
                "planned_sells": sells,
                "rejections": int(len(rejections)),
                "target_positions": int(len(positions)),
                "planned_cash_after": nav_row.get("cash"),
                "tracking_bootstrap": bootstrap,
                "source": "当前起算日·收盘跟踪重放",
            }
        )
    return pd.DataFrame(rows)


def zero_before_start_summary(names: list[str], labels: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": name,
                "策略": labels.get(name, name),
                "action": "未开始持仓",
                "is_rebalance_day": False,
                "是否调仓": "否",
                "planned_orders": 0,
                "planned_buys": 0,
                "planned_sells": 0,
                "rejections": 0,
                "current_positions": 0,
                "target_positions": 0,
                "planned_cash_after": None,
                "tracking_bootstrap": False,
                "source": "起算日前空仓",
            }
            for name in names
        ]
    )


@st.cache_data(show_spinner=False, ttl=10)
def cached_plan_preview(
    matrix_root_text: str,
    live_root_text: str,
    start_text: str,
    date_token: str,
) -> dict[str, Any]:
    return preview_nine_strategy_day(
        Path(matrix_root_text),
        Path(live_root_text),
        pd.Timestamp(start_text),
        pd.to_datetime(date_token, format="%Y%m%d"),
    )


matrix_root = Path(
    st.sidebar.text_input(
        "9组回测目录",
        value=os.environ.get("AS1455_MATRIX_ROOT", str(DEFAULT_MATRIX_ROOT)),
    )
).expanduser().resolve()
live_root = Path(
    st.sidebar.text_input(
        "盯盘输出目录",
        value=os.environ.get("AS1455_LIVE_ROOT", str(DEFAULT_LIVE_ROOT)),
    )
).expanduser().resolve()

required_token = os.environ.get("AS1455_DASHBOARD_REFRESH_TOKEN", "")
supplied_token = (
    st.sidebar.text_input("操作口令", type="password") if required_token else ""
)
authorized = not required_token or secrets.compare_digest(
    supplied_token, required_token
)

st.title("AS1455 九策略收益与14:55盯盘")
st.caption("9组全局固定信号策略：r01/r05/r21 × Top-5、Top-3、最优单模型")

if not matrix_root.is_dir():
    st.error(f"回测目录不存在：{matrix_root}")
    st.stop()

summary_file = matrix_root / "fixed_signal_matrix_summary.csv"
summary_mtime = summary_file.stat().st_mtime_ns if summary_file.is_file() else 0


@st.cache_data(show_spinner=False, ttl=30)
def cached_summary(root: str, token: int) -> pd.DataFrame:
    del token
    return load_matrix_summary(Path(root))


canonical_summary = cached_summary(str(matrix_root), summary_mtime)
experiment_names = discover_experiment_names(matrix_root)
if canonical_summary.empty or len(experiment_names) != 9:
    st.error(
        f"需要完整9组冻结策略；当前summary={len(canonical_summary)} "
        f"experiments={len(experiment_names)}"
    )
    st.stop()
labels = strategy_labels(canonical_summary)

refresh_status = load_refresh_status(matrix_root)
refresh_state = effective_state(refresh_status)
pre_status = load_job_status(live_root, "pre")
post_status = load_job_status(live_root, "post")
pre_state = effective_state(pre_status)
post_state = effective_state(post_status)

config_file = matrix_root / ".dashboard" / "user_config.json"
user_config = read_json(config_file, {}) or {}
curve_min, curve_max = load_curve_bounds(matrix_root, experiment_names)
live_date_tokens = discover_live_dates(live_root)
if live_date_tokens:
    curve_max = max(
        curve_max,
        pd.to_datetime(max(live_date_tokens), format="%Y%m%d").normalize(),
    )
stored = pd.to_datetime(user_config.get("tracking_start_date"), errors="coerce")
if pd.isna(stored):
    stored = curve_min
stored = min(max(pd.Timestamp(stored).normalize(), curve_min), curve_max)
selected_start_date = st.sidebar.date_input(
    "开始持仓 / 收益起算日",
    value=stored.date(),
    min_value=curve_min.date(),
    max_value=curve_max.date(),
    disabled=refresh_state == "running",
    help=(
        "起算日前账户为空；起算日不会重置任何策略的调仓offset。"
        "如果起算日不是该策略原定调仓日，账户继续保持现金，直到下一个原定调仓日才首次建仓。"
        "因此首次真正建仓时只能买入，不可能卖出。"
    ),
)
selected_start = pd.Timestamp(selected_start_date).normalize()
selected_start_text = selected_start.strftime("%Y-%m-%d")
start_semantics_changed = (
    user_config.get("tracking_start_date") != selected_start_text
    or int(user_config.get("tracking_semantics_version", 0) or 0)
    != TRACKING_SEMANTICS_VERSION
)
start_rebuild_message: tuple[bool, str] | None = None
if start_semantics_changed and refresh_state != "running":
    user_config["tracking_start_date"] = selected_start_text
    user_config["tracking_semantics_version"] = TRACKING_SEMANTICS_VERSION
    user_config["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(config_file, user_config)
    start_rebuild_message = start_background(
        ["bash", str(REFRESH_SCRIPT)],
        {
            "MATRIX_ROOT": str(matrix_root),
            "LIVE_ROOT": str(live_root),
            "SKIP_DATA_REFRESH": "1",
            "TRACKING_MODE": "rebuild",
            "TRACKING_START_DATE": selected_start_text,
        },
    )
    if start_rebuild_message[0]:
        refresh_state = "running"

tracking_summary, tracking_manifest = load_tracking_matrix_summary(matrix_root)
tracking_ready = tracking_ready_for_start(
    tracking_summary, tracking_manifest, selected_start
)
active_summary = tracking_summary if tracking_ready else canonical_summary

st.sidebar.divider()
st.sidebar.subheader("后台任务")
st.sidebar.caption(f"20:20市场数据 + 增量账户刷新：{refresh_state}")
if start_rebuild_message is not None:
    ok, msg = start_rebuild_message
    (st.sidebar.success if ok else st.sidebar.error)(
        ("起算日已更新；" if ok else "起算日已保存，但重建启动失败：") + msg
    )
if st.sidebar.button(
    "更新市场数据并增量推进9组账户",
    type="primary",
    disabled=refresh_state == "running" or not authorized,
    use_container_width=True,
):
    ok, msg = start_background(
        ["bash", str(REFRESH_SCRIPT)],
        {
            "MATRIX_ROOT": str(matrix_root),
            "LIVE_ROOT": str(live_root),
            "SKIP_DATA_REFRESH": "0",
            "TRACKING_MODE": "incremental",
        },
    )
    (st.sidebar.success if ok else st.sidebar.error)(msg)
if st.sidebar.button(
    "按当前起算日重建9组账户和盯盘计划",
    disabled=refresh_state == "running" or not authorized,
    use_container_width=True,
):
    ok, msg = start_background(
        ["bash", str(REFRESH_SCRIPT)],
        {
            "MATRIX_ROOT": str(matrix_root),
            "LIVE_ROOT": str(live_root),
            "SKIP_DATA_REFRESH": "1",
            "TRACKING_MODE": "rebuild",
            "TRACKING_START_DATE": selected_start_text,
        },
    )
    (st.sidebar.success if ok else st.sidebar.error)(msg)

st.sidebar.caption(f"14:55盯盘预处理：{pre_state}；计划生成：{post_state}")
col_a, col_b = st.sidebar.columns(2)
if col_a.button(
    "盯盘预处理",
    disabled=pre_state == "running" or not authorized,
    use_container_width=True,
):
    ok, msg = start_background(
        ["bash", str(LIVE_JOB_SCRIPT), "pre"],
        {"MATRIX_ROOT": str(matrix_root), "OUT_ROOT": str(live_root)},
    )
    (st.sidebar.success if ok else st.sidebar.error)(msg)
if col_b.button(
    "生成今日策略",
    disabled=post_state == "running" or not authorized,
    use_container_width=True,
):
    ok, msg = start_background(
        ["bash", str(LIVE_JOB_SCRIPT), "post"],
        {"MATRIX_ROOT": str(matrix_root), "OUT_ROOT": str(live_root)},
    )
    (st.sidebar.success if ok else st.sidebar.error)(msg)

if required_token and not authorized:
    st.sidebar.caption("输入正确口令后才能启动后台任务。")
if st.sidebar.button("重新读取页面", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

latest_end = (
    active_summary["forward_end"].dropna().astype(str).max()
    if "forward_end" in active_summary and not active_summary.empty
    else "—"
)
best_sharpe = None
if tracking_ready and "sharpe" in active_summary:
    values = pd.to_numeric(active_summary["sharpe"], errors="coerce")
    if values.notna().any():
        best_sharpe = active_summary.loc[values.idxmax()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("策略数", f"{len(canonical_summary)}/9")
m2.metric("开始持仓", selected_start_text)
m3.metric("跟踪最新日期", latest_end if tracking_ready else "重建中/未就绪")
m4.metric(
    "跟踪最高Sharpe",
    number(best_sharpe.get("sharpe")) if best_sharpe is not None else "—",
    best_sharpe.get("display_name") if best_sharpe is not None else None,
)

if refresh_state == "running":
    st.markdown(
        '<div class="status-running">当前起算日的9组收益账户和已有14:55盯盘计划正在后台统一刷新；完成后页面只读缓存，不再现场重放。</div>',
        unsafe_allow_html=True,
    )
elif refresh_state in {"failed", "stale"}:
    st.markdown(
        '<div class="status-failed">最近一次账户刷新未正常完成；请查看任务日志。冻结历史Grid不会被后台任务改写。</div>',
        unsafe_allow_html=True,
    )
if not tracking_ready:
    st.warning(
        "当前起算日的9个跟踪收益账户和盯盘缓存尚未全部就绪；请等待一次后台统一重建完成。"
        "页面查看本身不会再现场重放9个策略。"
    )

curve_tab, overview_tab, detail_tab, live_tab, task_tab = st.tabs(
    ["9策略收益曲线", "回测总览", "单策略详情", "每日14:55盯盘", "自动任务与日志"]
)

with curve_tab:
    st.subheader("从空仓开始的9策略跟踪收益")
    st.caption(
        "起算日前账户为空，但起算日不改变原调仓相位。若当天不是该策略原定调仓日，"
        "该账户继续持有现金；到下一个原定调仓日才从空仓首次建仓。"
    )
    c1, c2 = st.columns(2)
    target_filter = c1.multiselect(
        "目标周期",
        options=list(TARGET_LABELS.values()),
        default=list(TARGET_LABELS.values()),
    )
    signal_filter = c2.multiselect(
        "固定模型",
        options=list(SIGNAL_LABELS.values()),
        default=list(SIGNAL_LABELS.values()),
    )
    if not tracking_ready:
        st.info("等待当前起算日账户重建完成。")
    else:
        selected_names = active_summary.loc[
            active_summary["target_label"].isin(target_filter)
            & active_summary["signal_label"].isin(signal_filter),
            "experiment",
        ].astype(str).tolist()
        series: list[pd.Series] = []
        rows: list[dict[str, Any]] = []
        for name in selected_names:
            item = load_experiment(matrix_root, name)
            nav = item["tracking_nav"]
            if nav.empty:
                continue
            series.append(
                nav.set_index("date")["cumulative_return_pct"].rename(
                    labels.get(name, name)
                )
            )
            result = (
                item["tracking_result"].iloc[0].to_dict()
                if not item["tracking_result"].empty
                else {}
            )
            rows.append(
                {
                    "实验": labels.get(name, name),
                    "账户起点": selected_start_text,
                    "历史offset": result.get("historical_offset"),
                    "首次实际建仓": item["tracking_manifest"].get(
                        "first_entry_date"
                    )
                    or "尚未建仓",
                    "最新收益": result.get("total_return"),
                }
            )
        curves = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
        if curves.empty:
            st.info("当前起算日之后还没有已完成的市场日。")
        else:
            st.line_chart(curves, use_container_width=True, height=560)
            table = pd.DataFrame(rows)
            if not table.empty:
                table["最新收益"] = table["最新收益"].map(pct)
                st.dataframe(table, hide_index=True, use_container_width=True)

with overview_tab:
    st.subheader("9组当前跟踪账户指标")
    if tracking_ready:
        summary_view(active_summary)
        chart = active_summary[["display_name", "total_return"]].copy()
        chart["跟踪收益率（%）"] = (
            pd.to_numeric(chart["total_return"], errors="coerce") * 100
        )
        st.bar_chart(
            chart.set_index("display_name")[["跟踪收益率（%）"]],
            use_container_width=True,
        )
    else:
        st.info(
            "等待当前起算日账户重建完成；历史Fold/Grid仍保持冻结，可在单策略详情中查看。"
        )

with detail_tab:
    label_to_name = {labels[name]: name for name in experiment_names}
    selected_label = st.selectbox("选择策略", list(label_to_name))
    selected_name = label_to_name[selected_label]
    item = load_experiment(matrix_root, selected_name)
    item_tracking_ready = (
        item["tracking_manifest"].get("status") == "ok"
        and item["tracking_manifest"].get("tracking_start_date")
        == selected_start_text
        and int(item["tracking_manifest"].get("tracking_semantics_version", 0) or 0)
        == TRACKING_SEMANTICS_VERSION
        and not item["tracking_nav"].empty
    )
    result = (
        item["tracking_result"].iloc[0].to_dict()
        if item_tracking_ready and not item["tracking_result"].empty
        else {}
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("跟踪收益", pct(result.get("total_return")))
    d2.metric("年化收益", pct(result.get("annual_return")))
    d3.metric("Sharpe", number(result.get("sharpe")))
    d4.metric("最大回撤", pct(result.get("max_drawdown")))
    if item_tracking_ready:
        curve = item["tracking_nav"].set_index("date")[[
            "cumulative_return_pct"
        ]].rename(columns={"cumulative_return_pct": "累计收益率（%）"})
        st.line_chart(curve, use_container_width=True, height=420)
        st.caption(
            f"历史调仓周期={result.get('rebalance_every', '—')}，"
            f"历史offset={result.get('historical_offset', '—')}，"
            f"首次实际建仓={item['tracking_manifest'].get('first_entry_date') or '尚未建仓'}。"
        )
    else:
        st.info("该策略当前起算日跟踪账户尚未就绪。")

    sub_actions, sub_positions, sub_fold, sub_grid = st.tabs(
        ["每日买卖动作", "调仓持仓", "历史Fold", "历史Grid Top20"]
    )
    with sub_actions:
        if not item_tracking_ready:
            st.info("跟踪账户就绪后，这里会逐日展示买入、卖出以及无交易日。")
        else:
            nav = item["tracking_nav"].copy()
            nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
            available = sorted(
                nav["date"].dropna().dt.strftime("%Y-%m-%d").unique(),
                reverse=True,
            )
            selected_action_date = st.selectbox(
                "交易日", available, key="detail_action_date"
            )
            target_date = pd.Timestamp(selected_action_date).normalize()
            nav_row = nav.loc[nav["date"].eq(target_date)].iloc[-1]
            orders = item["tracking_orders"].copy()
            if not orders.empty:
                orders = orders.loc[
                    pd.to_datetime(orders["date"]).dt.normalize().eq(target_date)
                ]
            buys = (
                orders.loc[orders["side"].astype(str).str.lower().eq("buy")]
                if not orders.empty and "side" in orders
                else pd.DataFrame()
            )
            sells = (
                orders.loc[orders["side"].astype(str).str.lower().eq("sell")]
                if not orders.empty and "side" in orders
                else pd.DataFrame()
            )
            is_rebalance = as_bool(nav_row.get("is_rebalance_day", False))
            bootstrap = as_bool(nav_row.get("tracking_bootstrap", False))
            n_positions = int(nav_row.get("n_positions", 0) or 0)
            action = (
                "首次建仓"
                if bootstrap
                else "调仓"
                if is_rebalance and len(orders)
                else "调仓日·无需成交"
                if is_rebalance
                else "非调仓日·保持空仓"
                if n_positions == 0
                else "非调仓日·继续持有"
            )
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("动作", action)
            a2.metric("买入", str(len(buys)))
            a3.metric("卖出", str(len(sells)))
            a4.metric("收盘持仓", str(n_positions))
            if bootstrap:
                st.caption(
                    "这是该策略按原历史offset遇到的首次实际建仓日。此前账户一直为空，"
                    "因此这里只可能买入，不可能卖出。"
                )
            buy_col, sell_col = st.columns(2)
            with buy_col:
                st.markdown("**买入动作**")
                if buys.empty:
                    st.info("无买入")
                else:
                    st.dataframe(
                        order_view(buys), hide_index=True, use_container_width=True
                    )
            with sell_col:
                st.markdown("**卖出动作**")
                if sells.empty:
                    st.info("无卖出")
                else:
                    st.dataframe(
                        order_view(sells), hide_index=True, use_container_width=True
                    )
    with sub_positions:
        if not item_tracking_ready:
            st.info("跟踪账户尚未就绪。")
        else:
            dates = item["tracking_rebalance_dates"].copy()
            positions = item["tracking_positions"].copy()
            if dates.empty:
                st.info("当前跟踪区间还没有调仓日。")
            else:
                dates["date"] = pd.to_datetime(
                    dates["date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                preferred = [
                    "date",
                    "is_rebalance_day",
                    "tracking_bootstrap",
                    "nav",
                    "cash",
                    "n_positions",
                    "orders",
                    "buy_orders",
                    "sell_orders",
                    "turnover",
                ]
                st.dataframe(
                    dates[[c for c in preferred if c in dates.columns]].sort_values(
                        "date", ascending=False
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                available = sorted(dates["date"].dropna().unique(), reverse=True)
                if available:
                    selected_rebalance_date = st.selectbox(
                        "调仓日持仓", available, key="detail_position_date"
                    )
                    if not positions.empty:
                        positions["date"] = pd.to_datetime(
                            positions["date"], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
                        held = positions.loc[
                            positions["date"].eq(selected_rebalance_date)
                        ]
                        if held.empty:
                            st.info("该调仓日收盘后为空仓。")
                        else:
                            st.dataframe(
                                held, hide_index=True, use_container_width=True
                            )
    with sub_fold:
        st.caption("历史Fold用于冻结模型/交易参数，不随开始持仓日改变。")
        st.dataframe(item["fold_returns"], hide_index=True, use_container_width=True)
    with sub_grid:
        st.caption("历史Grid只用于历史窗口参数选择；每日刷新不会重跑。")
        st.dataframe(item["top20"], hide_index=True, use_container_width=True)

with live_tab:
    st.subheader("每日14:55九策略盯盘")
    st.caption(
        "修改开始持仓日时，系统会在后台一次性重建9个跟踪账户，并把已有14:55日期按新起算日统一重算后落盘。"
        "这里切换日期或策略时只读取已保存结果，不运行模型、不重跑Fold/Grid，也不现场重放账户。"
    )
    tracking_dates, tracking_details = tracking_date_sets(
        matrix_root, experiment_names, selected_start
    )
    original_live_dates = set(discover_live_dates(live_root))
    available_tokens = sorted(original_live_dates | tracking_dates, reverse=True)
    if not available_tokens:
        st.info("还没有可展示的盯盘/跟踪日期。")
    else:
        selected_live_date = st.selectbox("盯盘日期", available_tokens, index=0)
        selected_live_ts = pd.to_datetime(
            selected_live_date, format="%Y%m%d"
        ).normalize()
        source_mode = "none"
        preview: dict[str, Any] | None = None
        live_summary = pd.DataFrame()

        if selected_live_ts < selected_start:
            live_summary = zero_before_start_summary(experiment_names, labels)
            source_mode = "before_start"
        elif selected_live_date in original_live_dates:
            try:
                preview = cached_plan_preview(
                    str(matrix_root),
                    str(live_root),
                    selected_start_text,
                    selected_live_date,
                )
                live_summary = preview["summary"].copy()
                if not live_summary.empty:
                    live_summary["策略"] = live_summary["experiment"].map(
                        labels
                    ).fillna(live_summary["experiment"])
                    live_summary["是否调仓"] = live_summary[
                        "is_rebalance_day"
                    ].map(as_bool).map({True: "是", False: "否"})
                source_mode = "preview"
            except Exception as exc:
                st.warning(
                    "当前起算日的预计算盯盘缓存尚未就绪，尝试回退到已完成的同起算日跟踪结果。"
                    f" 原因：{type(exc).__name__}: {exc}"
                )
                if selected_live_date in tracking_dates:
                    live_summary = tracking_day_summary(
                        tracking_details,
                        experiment_names,
                        labels,
                        selected_live_date,
                    )
                    source_mode = "tracking"
        elif selected_live_date in tracking_dates:
            live_summary = tracking_day_summary(
                tracking_details, experiment_names, labels, selected_live_date
            )
            source_mode = "tracking"

        if source_mode == "before_start":
            st.info(
                f"{selected_live_ts:%Y-%m-%d} 早于开始持仓日 {selected_start_text}："
                "9个账户均为空仓，不产生买卖。"
            )
        elif source_mode == "preview" and preview is not None:
            fallback_dates = preview.get("raw_daily_fallback_dates") or []
            st.caption(
                "数据源：起算日变更时统一预计算缓存 / 当日正式计划；页面即时计算=否，模型推理=否，历史Grid=否。"
                f"执行数据={preview.get('execution_source', '—')}。"
            )
            if fallback_dates:
                st.caption(
                    "部分旧日期缺少保存的14:55 sidecar，统一重建时使用raw-daily执行数据回退："
                    + "、".join(fallback_dates)
                )
        elif source_mode == "tracking":
            st.caption("数据源：当前起算日已完成的收盘跟踪重放。")

        if live_summary.empty:
            st.info("该日期没有可用的完整9策略结果。")
        else:
            preferred = [
                "策略",
                "action",
                "是否调仓",
                "rebalance_every",
                "historical_offset",
                "effective_preview_offset",
                "planned_buys",
                "planned_sells",
                "rejections",
                "current_positions",
                "target_positions",
                "planned_cash_after",
                "tracking_bootstrap",
                "source",
            ]
            st.dataframe(
                live_summary[[c for c in preferred if c in live_summary.columns]],
                hide_index=True,
                use_container_width=True,
            )
            rebalance_today = (
                live_summary.loc[live_summary["is_rebalance_day"].map(as_bool)]
                if "is_rebalance_day" in live_summary
                else pd.DataFrame()
            )
            st.metric("该日需要调仓的策略", f"{len(rebalance_today)}/9")
            if not rebalance_today.empty:
                names = "、".join(rebalance_today["策略"].astype(str).tolist())
                st.markdown(
                    f'<div class="strategy-action"><b>该日调仓：</b>{names}</div>',
                    unsafe_allow_html=True,
                )

            strategy_label_map = {
                labels.get(str(row["experiment"]), str(row["experiment"])): str(
                    row["experiment"]
                )
                for _, row in live_summary.iterrows()
                if str(row.get("status", "ok")) == "ok"
            }
            if not strategy_label_map:
                st.info("没有可展开的策略计划。")
            else:
                selected_strategy_label = st.selectbox(
                    "查看策略计划", list(strategy_label_map)
                )
                selected_experiment = strategy_label_map[selected_strategy_label]
                orders = pd.DataFrame()
                target_positions = pd.DataFrame()
                current_positions = pd.DataFrame()
                rejections = pd.DataFrame()
                rank = pd.DataFrame()
                action = "—"

                if source_mode == "preview" and preview is not None:
                    detail = preview["details"][selected_experiment]
                    manifest = detail["manifest"]
                    action = str(manifest.get("action", "—"))
                    orders = detail["orders"]
                    target_positions = detail["target_positions"]
                    current_positions = detail["current_positions"]
                    rejections = detail["rejections"]
                    rank = detail["rank"]
                elif source_mode == "tracking":
                    detail = tracking_details[selected_experiment]
                    target_date = selected_live_ts
                    nav_day = detail["tracking_nav"].loc[
                        pd.to_datetime(detail["tracking_nav"]["date"])
                        .dt.normalize()
                        .eq(target_date)
                    ]
                    nav_row = nav_day.iloc[-1]
                    action = (
                        "首次建仓"
                        if as_bool(nav_row.get("tracking_bootstrap", False))
                        else "调仓"
                        if as_bool(nav_row.get("is_rebalance_day", False))
                        and int(nav_row.get("orders", 0))
                        else "调仓日·无需成交"
                        if as_bool(nav_row.get("is_rebalance_day", False))
                        else "非调仓日·保持空仓"
                        if int(nav_row.get("n_positions", 0) or 0) == 0
                        else "非调仓日·继续持有"
                    )
                    orders = detail["tracking_orders"]
                    if not orders.empty:
                        orders = orders.loc[
                            pd.to_datetime(orders["date"])
                            .dt.normalize()
                            .eq(target_date)
                        ]
                    target_positions = detail["tracking_positions"]
                    if not target_positions.empty:
                        target_positions = target_positions.loc[
                            pd.to_datetime(target_positions["date"])
                            .dt.normalize()
                            .eq(target_date)
                        ]
                    rejections = detail["tracking_rejections"]
                    if not rejections.empty:
                        rejections = rejections.loc[
                            pd.to_datetime(rejections["date"])
                            .dt.normalize()
                            .eq(target_date)
                        ]
                    if selected_live_date in original_live_dates:
                        rank = load_strategy(
                            load_live_day(live_root, selected_live_date),
                            selected_experiment,
                        )["rank"]

                buy_count = (
                    int(orders["side"].astype(str).str.lower().eq("buy").sum())
                    if not orders.empty and "side" in orders
                    else 0
                )
                sell_count = (
                    int(orders["side"].astype(str).str.lower().eq("sell").sum())
                    if not orders.empty and "side" in orders
                    else 0
                )
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("动作", action)
                q2.metric("计划买入", str(buy_count))
                q3.metric("计划卖出", str(sell_count))
                q4.metric("目标持仓数", str(len(target_positions)))
                orders_tab, target_tab, current_tab, reject_tab, rank_tab = st.tabs(
                    [
                        "计划订单",
                        "调仓后目标组合",
                        "调仓前持仓",
                        "未成交/拒单",
                        "当日模型排名",
                    ]
                )
                with orders_tab:
                    if orders.empty:
                        st.info("该日没有计划成交。")
                    else:
                        st.dataframe(
                            order_view(orders),
                            hide_index=True,
                            use_container_width=True,
                        )
                with target_tab:
                    st.dataframe(
                        target_positions, hide_index=True, use_container_width=True
                    )
                with current_tab:
                    if current_positions.empty:
                        st.info("调仓前为空仓。")
                    else:
                        st.dataframe(
                            current_positions,
                            hide_index=True,
                            use_container_width=True,
                        )
                with reject_tab:
                    st.dataframe(
                        rejections, hide_index=True, use_container_width=True
                    )
                with rank_tab:
                    if rank.empty:
                        st.info("该日期没有可用模型排名。")
                    else:
                        st.dataframe(
                            rank.head(100), hide_index=True, use_container_width=True
                        )

with task_tab:
    st.subheader("自动任务")
    st.code(
        "09:35  工作日：准备T-1历史、preclose和紧凑特征状态\n"
        "14:50  工作日：启动实时采集，14:55冻结快照并生成9策略预测/rank\n"
        "20:20  工作日：探测BaoStock当日数据；可用则更新到T，否则保持T-1；随后只增量推进9个跟踪账户",
        language="text",
    )
    st.markdown(
        "安装定时任务（需要 root）：\n"
        "```bash\n"
        "sudo bash scripts/install_as1455_live_nine_strategy_cron.sh\n"
        "sudo bash scripts/install_as1455_dashboard_daily_refresh_cron.sh\n"
        "```"
    )
    st.caption(
        "每日任务不重跑历史Fold/Grid，也不重新计算旧Forward窗口；正常情况下只更新市场缓存并追加新日期。"
        "开始持仓日改变时，9个收益账户和已有14:55计划会在后台统一重建一次；之后页面只读缓存。"
    )
    statuses = {
        "20:20市场数据/账户刷新": refresh_status,
        "09:35盯盘预处理": pre_status,
        "14:55盯盘计划": post_status,
    }
    for title, status in statuses.items():
        with st.expander(
            title,
            expanded=effective_state(status)
            in {"failed", "stale", "running"},
        ):
            st.json(status)
            log = status.get("resolved_log_file")
            if log:
                st.code(
                    tail_text(Path(str(log)), max_lines=120) or "日志尚无内容",
                    language="text",
                )

st.caption(f"页面读取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
