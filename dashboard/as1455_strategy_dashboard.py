#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrated AS1455 dashboard: nine backtests + 14:55 nine-strategy monitor."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from datetime import date, datetime
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
    tail_text,
)
from dashboard.as1455_live_data import (  # noqa: E402
    discover_live_dates,
    load_job_status,
    load_live_day,
    load_strategy,
)

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
.status-success {padding:.65rem .9rem;border-left:4px solid #27ae60;background:rgba(39,174,96,.08);}
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
    labels: dict[str, str] = {}
    for _, row in summary.iterrows():
        labels[str(row["experiment"])] = str(row.get("display_name", row["experiment"]))
    return labels


def load_curve_bounds(matrix_root: Path, experiment_names: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    firsts: list[pd.Timestamp] = []
    lasts: list[pd.Timestamp] = []
    for name in experiment_names:
        nav = load_experiment(matrix_root, name)["forward_nav"]
        if nav.empty:
            continue
        firsts.append(pd.Timestamp(nav["date"].min()).normalize())
        lasts.append(pd.Timestamp(nav["date"].max()).normalize())
    if not firsts or not lasts:
        today = pd.Timestamp.today().normalize()
        return today, today
    return min(firsts), max(lasts)


def build_rebased_curves(
    matrix_root: Path,
    names: list[str],
    labels: dict[str, str],
    start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series: list[pd.Series] = []
    effective_rows: list[dict[str, Any]] = []
    for name in names:
        nav = load_experiment(matrix_root, name)["forward_nav"].copy()
        if nav.empty:
            continue
        nav["date"] = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
        nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
        nav = nav.dropna(subset=["date", "nav"]).sort_values("date")
        part = nav.loc[nav["date"].ge(start)].copy()
        if part.empty:
            continue
        baseline = float(part.iloc[0]["nav"])
        if baseline <= 0:
            continue
        effective = pd.Timestamp(part.iloc[0]["date"]).normalize()
        part["return_pct"] = (part["nav"] / baseline - 1.0) * 100.0
        series.append(part.set_index("date")["return_pct"].rename(labels.get(name, name)))
        effective_rows.append(
            {
                "实验": labels.get(name, name),
                "选择起点": start.strftime("%Y-%m-%d"),
                "实际起算交易日": effective.strftime("%Y-%m-%d"),
                "起算NAV": baseline,
                "最新收益": float(part.iloc[-1]["return_pct"]) / 100.0,
            }
        )
    curves = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
    return curves, pd.DataFrame(effective_rows)


def summary_view(summary: pd.DataFrame) -> None:
    view = summary.copy()
    columns = {
        "display_name": "实验",
        "rebalance_every": "调仓周期",
        "total_return": "Forward收益",
        "annual_return": "年化收益",
        "sharpe": "Sharpe",
        "max_drawdown": "最大回撤",
        "forward_start": "Forward开始",
        "forward_end": "最新日期",
        "historical_result_reused": "历史Grid复用",
    }
    keep = [column for column in columns if column in view.columns]
    view = view[keep].rename(columns=columns)
    for column in ("Forward收益", "年化收益", "最大回撤"):
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(pct)
    if "Sharpe" in view.columns:
        view["Sharpe"] = pd.to_numeric(view["Sharpe"], errors="coerce").map(number)
    if "调仓周期" in view.columns:
        view["调仓周期"] = pd.to_numeric(view["调仓周期"], errors="coerce").map(
            lambda x: f"每{int(x)}日" if pd.notna(x) else "—"
        )
    st.dataframe(view, hide_index=True, use_container_width=True)


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
supplied_token = st.sidebar.text_input("操作口令", type="password") if required_token else ""
authorized = not required_token or secrets.compare_digest(supplied_token, required_token)

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


summary = cached_summary(str(matrix_root), summary_mtime)
experiment_names = discover_experiment_names(matrix_root)
if summary.empty or len(experiment_names) != 9:
    st.error(f"需要完整9组结果；当前summary={len(summary)} experiments={len(experiment_names)}")
    st.stop()
labels = strategy_labels(summary)

# Persist the performance start date so the page returns to the user's chosen
# synchronization date after Streamlit/server restarts.
config_file = matrix_root / ".dashboard" / "user_config.json"
user_config = read_json(config_file, {}) or {}
curve_min, curve_max = load_curve_bounds(matrix_root, experiment_names)
stored = pd.to_datetime(user_config.get("tracking_start_date"), errors="coerce")
if pd.isna(stored):
    stored = curve_min
stored = min(max(pd.Timestamp(stored).normalize(), curve_min), curve_max)
selected_start_date = st.sidebar.date_input(
    "开始持仓 / 收益起算日",
    value=stored.date(),
    min_value=curve_min.date(),
    max_value=curve_max.date(),
    help=(
        "按所选日期或之后第一个交易日的策略NAV重新归零。"
        "这是同步加入既有策略的收益起点，不重跑历史Grid，也不改变调仓相位。"
    ),
)
selected_start = pd.Timestamp(selected_start_date).normalize()
if user_config.get("tracking_start_date") != selected_start.strftime("%Y-%m-%d"):
    user_config["tracking_start_date"] = selected_start.strftime("%Y-%m-%d")
    user_config["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(config_file, user_config)

refresh_status = load_refresh_status(matrix_root)
refresh_state = effective_state(refresh_status)
pre_status = load_job_status(live_root, "pre")
post_status = load_job_status(live_root, "post")
pre_state = effective_state(pre_status)
post_state = effective_state(post_status)

st.sidebar.divider()
st.sidebar.subheader("后台任务")
st.sidebar.caption(f"20:00回测刷新：{refresh_state}")
if st.sidebar.button(
    "立即更新数据并刷新9组回测",
    type="primary",
    disabled=refresh_state == "running" or not authorized,
    use_container_width=True,
):
    ok, msg = start_background(
        ["bash", str(REFRESH_SCRIPT)],
        {
            "MATRIX_ROOT": str(matrix_root),
            "SKIP_DATA_REFRESH": "0",
            "REQUIRE_HISTORICAL_REUSE": "1",
            "FORCE_HISTORICAL_GRID": "0",
        },
    )
    (st.sidebar.success if ok else st.sidebar.error)(msg)

st.sidebar.caption(f"14:55盯盘预处理：{pre_state}；计划生成：{post_state}")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("盯盘预处理", disabled=pre_state == "running" or not authorized, use_container_width=True):
    ok, msg = start_background(
        ["bash", str(LIVE_JOB_SCRIPT), "pre"],
        {"MATRIX_ROOT": str(matrix_root), "OUT_ROOT": str(live_root)},
    )
    (st.sidebar.success if ok else st.sidebar.error)(msg)
if col_b.button("生成今日策略", disabled=post_state == "running" or not authorized, use_container_width=True):
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

latest_end = summary["forward_end"].dropna().astype(str).max() if "forward_end" in summary else "—"
best_sharpe = None
if "sharpe" in summary:
    values = pd.to_numeric(summary["sharpe"], errors="coerce")
    if values.notna().any():
        best_sharpe = summary.loc[values.idxmax()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("策略数", f"{len(summary)}/9")
m2.metric("收益起算日", selected_start.strftime("%Y-%m-%d"))
m3.metric("回测最新日期", latest_end)
m4.metric(
    "Forward最高Sharpe",
    number(best_sharpe.get("sharpe")) if best_sharpe is not None else "—",
    best_sharpe.get("display_name") if best_sharpe is not None else None,
)

if refresh_state == "running":
    st.markdown('<div class="status-running">20:00收益刷新正在后台运行，页面暂时展示上一版完整结果。</div>', unsafe_allow_html=True)
elif refresh_state in {"failed", "stale"}:
    st.markdown('<div class="status-failed">最近一次收益刷新未正常完成；旧结果仍保留，请查看任务日志。</div>', unsafe_allow_html=True)

curve_tab, overview_tab, detail_tab, live_tab, task_tab = st.tabs(
    ["9策略收益曲线", "回测总览", "单策略详情", "每日14:55盯盘", "自动任务与日志"]
)

with curve_tab:
    st.subheader("从指定开始持仓日同步跟随9个策略")
    st.caption(
        "曲线在所选日期或之后第一个可用交易日重新归零；不补计首次同步建仓成本，后续收益来自原 strict-forward 策略账户。"
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
    selected_names = summary.loc[
        summary["target_label"].isin(target_filter)
        & summary["signal_label"].isin(signal_filter),
        "experiment",
    ].astype(str).tolist()
    curves, effective = build_rebased_curves(
        matrix_root, selected_names, labels, selected_start
    )
    if curves.empty:
        st.info("所选开始日期之后没有可用Forward NAV。")
    else:
        st.line_chart(curves, use_container_width=True, height=560)
        if not effective.empty:
            table = effective.copy()
            table["最新收益"] = table["最新收益"].map(pct)
            st.dataframe(table, hide_index=True, use_container_width=True)

with overview_tab:
    st.subheader("9组统一回测指标")
    summary_view(summary)
    chart = summary[["display_name", "total_return"]].copy()
    chart["Forward收益率（%）"] = pd.to_numeric(chart["total_return"], errors="coerce") * 100
    st.bar_chart(chart.set_index("display_name")[["Forward收益率（%）"]], use_container_width=True)

with detail_tab:
    label_to_name = {labels[name]: name for name in experiment_names}
    selected_label = st.selectbox("选择策略", list(label_to_name))
    item = load_experiment(matrix_root, label_to_name[selected_label])
    result = item["result"].iloc[0].to_dict() if not item["result"].empty else {}
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Forward收益", pct(result.get("total_return")))
    d2.metric("年化收益", pct(result.get("annual_return")))
    d3.metric("Sharpe", number(result.get("sharpe")))
    d4.metric("最大回撤", pct(result.get("max_drawdown")))
    nav = item["forward_nav"]
    if not nav.empty:
        curve = nav.set_index("date")[["cumulative_return_pct"]].rename(
            columns={"cumulative_return_pct": "累计收益率（%）"}
        )
        st.line_chart(curve, use_container_width=True, height=420)
    sub1, sub2, sub3 = st.tabs(["历史Fold", "调仓持仓", "历史Grid Top20"])
    with sub1:
        st.dataframe(item["fold_returns"], hide_index=True, use_container_width=True)
    with sub2:
        dates = item["rebalance_dates"].copy()
        positions = item["positions"].copy()
        if dates.empty:
            st.info("尚无调仓持仓审计文件。")
        else:
            dates["date"] = pd.to_datetime(dates["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.dataframe(dates.sort_values("date", ascending=False), hide_index=True, use_container_width=True)
            available = sorted(dates["date"].dropna().unique(), reverse=True)
            if available and not positions.empty:
                selected_date = st.selectbox("调仓日", available)
                positions["date"] = pd.to_datetime(positions["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                st.dataframe(
                    positions.loc[positions["date"].eq(selected_date)],
                    hide_index=True,
                    use_container_width=True,
                )
    with sub3:
        st.dataframe(item["top20"], hide_index=True, use_container_width=True)

with live_tab:
    st.subheader("每日14:55九策略盯盘")
    st.caption(
        "09:35准备历史/特征状态；14:50开始采集；14:55冻结最新有效快照后生成计划。"
        "这里只输出 planned_not_submitted 策略，不调用券商API。"
    )
    live_dates = discover_live_dates(live_root)
    if not live_dates:
        st.info("还没有九策略盯盘结果。安装定时任务或手动运行预处理/生成今日策略后会出现在这里。")
    else:
        selected_live_date = st.selectbox("盯盘日期", live_dates, index=0)
        live_day = load_live_day(live_root, selected_live_date)
        live_summary = live_day["summary"].copy()
        if live_summary.empty:
            st.warning("该日期manifest存在，但summary为空。")
        else:
            live_summary["策略"] = live_summary["experiment"].map(labels).fillna(live_summary["experiment"])
            live_summary["是否调仓"] = live_summary["is_rebalance_day"].astype(bool).map({True: "是", False: "否"})
            preferred = [
                "策略", "action", "是否调仓", "planned_buys", "planned_sells",
                "rejections", "current_positions", "target_positions",
                "state_asof_date", "planned_cash_after", "fixed_signal_spec",
            ]
            st.dataframe(
                live_summary[[c for c in preferred if c in live_summary.columns]],
                hide_index=True,
                use_container_width=True,
            )
            rebalance_today = live_summary.loc[live_summary["is_rebalance_day"].astype(bool)]
            st.metric("今日需要调仓的策略", f"{len(rebalance_today)}/9")
            if not rebalance_today.empty:
                names = "、".join(rebalance_today["策略"].astype(str).tolist())
                st.markdown(
                    f'<div class="strategy-action"><b>今日调仓：</b>{names}</div>',
                    unsafe_allow_html=True,
                )
            strategy_label_map = {
                labels.get(str(row["experiment"]), str(row["experiment"])): str(row["experiment"])
                for _, row in live_summary.iterrows()
            }
            selected_strategy_label = st.selectbox("查看策略计划", list(strategy_label_map))
            selected_experiment = strategy_label_map[selected_strategy_label]
            detail = load_strategy(live_day, selected_experiment)
            manifest = detail["manifest"]
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("动作", str(manifest.get("action", "—")))
            q2.metric("计划买入", str(manifest.get("planned_buy_count", 0)))
            q3.metric("计划卖出", str(manifest.get("planned_sell_count", 0)))
            q4.metric("目标持仓数", str(manifest.get("target_position_count", 0)))
            orders_tab, target_tab, reject_tab, rank_tab = st.tabs(
                ["计划订单", "调仓后目标组合", "未成交/拒单", "当日模型排名"]
            )
            with orders_tab:
                if detail["orders"].empty:
                    st.info("今日没有计划成交；若不是调仓日，这是正常状态。")
                else:
                    st.dataframe(detail["orders"], hide_index=True, use_container_width=True)
            with target_tab:
                st.dataframe(detail["target_positions"], hide_index=True, use_container_width=True)
            with reject_tab:
                st.dataframe(detail["rejections"], hide_index=True, use_container_width=True)
            with rank_tab:
                st.dataframe(detail["rank"].head(100), hide_index=True, use_container_width=True)

with task_tab:
    st.subheader("自动任务")
    st.code(
        "09:35  工作日：准备T-1历史、preclose和紧凑特征状态\n"
        "14:50  工作日：启动实时采集，14:55冻结快照并生成9策略计划\n"
        "20:00  工作日：更新每日数据、刷新9组Forward收益曲线和T日最终账户状态",
        language="text",
    )
    st.markdown(
        "安装定时任务：\n"
        "```bash\n"
        "bash scripts/install_as1455_live_nine_strategy_cron.sh\n"
        "bash scripts/install_as1455_dashboard_daily_refresh_cron.sh\n"
        "```"
    )
    statuses = {
        "20:00收益刷新": refresh_status,
        "09:35盯盘预处理": pre_status,
        "14:55盯盘计划": post_status,
    }
    for title, status in statuses.items():
        with st.expander(title, expanded=effective_state(status) in {"failed", "stale", "running"}):
            st.json(status)
            log = status.get("resolved_log_file")
            if log:
                st.code(tail_text(Path(str(log)), max_lines=120) or "日志尚无内容", language="text")
    st.caption(
        "前端/定时刷新均要求复用9组已验证历史Grid；缺失任何历史结果都会失败关闭，不会静默重新跑Grid。"
    )

st.caption(f"页面读取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
