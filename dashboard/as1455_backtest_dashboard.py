#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit dashboard for the nine AS1455 global fixed-signal backtests."""
from __future__ import annotations

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
    build_forward_comparison,
    discover_experiment_names,
    load_experiment,
    load_matrix_summary,
    load_refresh_status,
    tail_text,
)

DEFAULT_MATRIX_ROOT = PROJECT_ROOT / "saved_data" / "ashare_ml4t" / "ch17_as1455_global_fixed_signal_matrix" / "refresh_all_v1"
REFRESH_SCRIPT = PROJECT_ROOT / "scripts" / "run_as1455_dashboard_refresh.sh"

st.set_page_config(
    page_title="AS1455 回测看板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.block-container {padding-top: 1.35rem; padding-bottom: 2rem;}
.status-running {padding: .7rem 1rem; border-left: 4px solid #f39c12; background: rgba(243,156,18,.08);}
.status-success {padding: .7rem 1rem; border-left: 4px solid #27ae60; background: rgba(39,174,96,.08);}
.status-failed {padding: .7rem 1rem; border-left: 4px solid #c0392b; background: rgba(192,57,43,.08);}
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


def path_input() -> Path:
    configured = os.environ.get("AS1455_MATRIX_ROOT", str(DEFAULT_MATRIX_ROOT))
    return Path(st.sidebar.text_input("回测结果目录", value=configured)).expanduser().resolve()


def start_refresh(matrix_root: Path, skip_data_refresh: bool) -> tuple[bool, str]:
    if not REFRESH_SCRIPT.is_file():
        return False, f"刷新脚本不存在：{REFRESH_SCRIPT}"
    env = os.environ.copy()
    env.update(
        {
            "MATRIX_ROOT": str(matrix_root),
            "SKIP_DATA_REFRESH": "1" if skip_data_refresh else "0",
            "FORCE_HISTORICAL_GRID": "0",
            "FORCE_HISTORICAL_PREDICTIONS": "0",
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
    return True, "刷新任务已提交；页面不会阻塞，可稍后点击“重新读取”。"


def process_is_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError, PermissionError, OSError):
        return False
    return True


def effective_state(status: dict[str, Any]) -> str:
    state = str(status.get("status", "idle"))
    if state == "running" and not process_is_alive(status.get("pid")):
        return "stale"
    return state


def render_refresh_panel(matrix_root: Path) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("每日刷新")
    status = load_refresh_status(matrix_root)
    state = effective_state(status)
    labels = {
        "idle": "尚未从前端启动",
        "running": "正在刷新",
        "success": "最近一次成功",
        "failed": "最近一次失败",
        "blocked": "已有刷新任务运行",
        "stale": "上次任务异常终止，可重新启动",
    }
    st.sidebar.caption(labels.get(state, state))
    if status.get("started_at"):
        st.sidebar.caption(f"开始：{status['started_at']}")
    if status.get("finished_at"):
        st.sidebar.caption(f"结束：{status['finished_at']}")

    required_token = os.environ.get("AS1455_DASHBOARD_REFRESH_TOKEN", "")
    supplied_token = st.sidebar.text_input("刷新口令", type="password") if required_token else ""
    authorized = not required_token or secrets.compare_digest(supplied_token, required_token)
    full = st.sidebar.button(
        "更新每日数据并刷新9组回测",
        type="primary",
        disabled=state == "running" or not authorized,
        use_container_width=True,
    )
    backtest_only = st.sidebar.button(
        "复用现有数据，仅刷新9组回测",
        disabled=state == "running" or not authorized,
        use_container_width=True,
    )
    if required_token and not authorized:
        st.sidebar.caption("输入正确口令后才可启动刷新。")
    if full or backtest_only:
        ok, message = start_refresh(matrix_root, skip_data_refresh=backtest_only)
        (st.sidebar.success if ok else st.sidebar.error)(message)
    if st.sidebar.button("重新读取结果与状态", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("刷新前先验证9组历史 Grid；任一缺失都会停止，不会静默重跑 Grid。")

    log_value = status.get("resolved_log_file")
    if log_value:
        with st.sidebar.expander("最近刷新日志", expanded=state in {"running", "failed", "stale"}):
            st.code(tail_text(Path(str(log_value)), max_lines=100) or "日志尚无内容", language="text")


def summary_view(summary: pd.DataFrame) -> None:
    view = summary.copy()
    columns = {
        "display_name": "实验",
        "rebalance_every": "调仓周期",
        "total_return": "Forward收益",
        "annual_return": "年化收益",
        "sharpe": "Sharpe",
        "max_drawdown": "最大回撤",
        "forward_start": "开始日期",
        "forward_end": "结束日期",
        "historical_result_reused": "历史Grid复用",
    }
    view = view[[column for column in columns if column in view.columns]].rename(columns=columns)
    for column in ("Forward收益", "年化收益", "最大回撤"):
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce").map(pct)
    if "Sharpe" in view.columns:
        view["Sharpe"] = pd.to_numeric(view["Sharpe"], errors="coerce").map(number)
    if "调仓周期" in view.columns:
        view["调仓周期"] = pd.to_numeric(view["调仓周期"], errors="coerce").map(
            lambda value: f"每{int(value)}日" if pd.notna(value) else "—"
        )
    st.dataframe(view, hide_index=True, use_container_width=True)


matrix_root = path_input()
render_refresh_panel(matrix_root)
st.title("AS1455 九模型回测看板")
st.caption(f"结果目录：`{matrix_root}`")

if not matrix_root.is_dir():
    st.error("结果目录不存在。请先运行统一9组回测，或在左侧修改目录。")
    st.stop()

summary_file = matrix_root / "fixed_signal_matrix_summary.csv"
summary_mtime = summary_file.stat().st_mtime_ns if summary_file.is_file() else 0


@st.cache_data(show_spinner=False, ttl=30)
def cached_summary(root: str, token: int) -> pd.DataFrame:
    del token
    return load_matrix_summary(Path(root))


summary = cached_summary(str(matrix_root), summary_mtime)
experiment_names = discover_experiment_names(matrix_root)
if summary.empty or not experiment_names:
    st.warning("尚未发现完整的9组结果。可在左侧启动刷新，或检查 fixed_signal_matrix_summary.csv。")
    st.stop()

latest_end = summary["forward_end"].dropna().astype(str).max() if "forward_end" in summary else "—"
best_sharpe_row = None
best_return_row = None
if "sharpe" in summary.columns:
    values = pd.to_numeric(summary["sharpe"], errors="coerce")
    if values.notna().any():
        best_sharpe_row = summary.loc[values.idxmax()]
if "total_return" in summary.columns:
    values = pd.to_numeric(summary["total_return"], errors="coerce")
    if values.notna().any():
        best_return_row = summary.loc[values.idxmax()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("已完成实验", f"{len(summary)}/9")
m2.metric("最新 Forward 日期", latest_end)
m3.metric(
    "最高 Sharpe",
    number(best_sharpe_row.get("sharpe")) if best_sharpe_row is not None else "—",
    best_sharpe_row.get("display_name") if best_sharpe_row is not None else None,
)
m4.metric(
    "最高 Forward 收益",
    pct(best_return_row.get("total_return")) if best_return_row is not None else "—",
    best_return_row.get("display_name") if best_return_row is not None else None,
)

status = load_refresh_status(matrix_root)
state = effective_state(status)
if state == "running":
    st.markdown('<div class="status-running">刷新任务正在后台运行。当前页面展示的是最近一次完整结果。</div>', unsafe_allow_html=True)
elif state in {"failed", "stale"}:
    st.markdown('<div class="status-failed">最近一次刷新未正常完成。请在左侧查看日志；旧结果未被删除。</div>', unsafe_allow_html=True)
elif state == "success":
    st.markdown('<div class="status-success">最近一次前端刷新成功。</div>', unsafe_allow_html=True)

overview_tab, compare_tab, detail_tab, refresh_tab = st.tabs(
    ["九组总览", "Forward曲线对比", "单组详情", "刷新与运行说明"]
)

with overview_tab:
    st.subheader("统一指标")
    summary_view(summary)
    if "total_return" in summary.columns:
        chart = summary[["display_name", "total_return"]].copy()
        chart["Forward收益率（%）"] = pd.to_numeric(chart["total_return"], errors="coerce") * 100
        st.bar_chart(chart.set_index("display_name")[["Forward收益率（%）"]], use_container_width=True)

with compare_tab:
    c1, c2 = st.columns(2)
    selected_target = c1.selectbox("目标周期", ["全部"] + list(TARGET_LABELS.values()))
    selected_signal = c2.selectbox("固定模型", ["全部"] + list(SIGNAL_LABELS.values()))
    filtered = summary.copy()
    if selected_target != "全部":
        filtered = filtered[filtered["target_label"] == selected_target]
    if selected_signal != "全部":
        filtered = filtered[filtered["signal_label"] == selected_signal]
    comparison = build_forward_comparison(matrix_root, filtered["experiment"].astype(str).tolist())
    if comparison.empty:
        st.info("所选实验没有可读取的 close_auction_nav.csv。")
    else:
        st.line_chart(comparison, use_container_width=True, height=520)
        st.caption("所有曲线均从各自 Forward 首日净值归一化为0%，仅用于横向比较。")

with detail_tab:
    label_to_name = {row["display_name"]: row["experiment"] for _, row in summary.iterrows()}
    selected_label = st.selectbox("选择实验", list(label_to_name))
    item = load_experiment(matrix_root, str(label_to_name[selected_label]))
    result = item["result"].iloc[0].to_dict() if not item["result"].empty else {}
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Forward收益", pct(result.get("total_return")))
    d2.metric("年化收益", pct(result.get("annual_return")))
    d3.metric("Sharpe", number(result.get("sharpe")))
    d4.metric("最大回撤", pct(result.get("max_drawdown")))
    d5.metric("Forward天数", str(result.get("forward_n_days", "—")))

    forward_tab, fold_tab, holdings_tab, grid_tab, files_tab = st.tabs(
        ["Forward", "历史Fold", "调仓与持仓", "历史Grid Top20", "文件与口径"]
    )
    with forward_tab:
        nav = item["forward_nav"]
        if nav.empty:
            st.info("未找到 Forward NAV。")
        else:
            curve = nav.set_index("date")[["cumulative_return_pct"]].rename(columns={"cumulative_return_pct": "累计收益率（%）"})
            st.line_chart(curve, use_container_width=True, height=430)
            drawdown = nav.set_index("date")[["drawdown"]].copy() * 100
            drawdown.columns = ["回撤（%）"]
            st.area_chart(drawdown, use_container_width=True, height=220)
            st.dataframe(nav.tail(30), hide_index=True, use_container_width=True)
    with fold_tab:
        folds = item["fold_returns"]
        if folds.empty:
            st.info("未找到 historical_fold_segment_returns.csv。")
        else:
            if "segment_return_pct" in folds.columns:
                st.bar_chart(folds.set_index("segment")[["segment_return_pct"]], use_container_width=True)
            st.dataframe(folds, hide_index=True, use_container_width=True)
    with holdings_tab:
        dates = item["rebalance_dates"].copy()
        positions = item["positions"].copy()
        if dates.empty:
            st.info("尚未导出 strict_forward_rebalance_dates.csv。请重新运行统一刷新。")
        else:
            dates["date"] = pd.to_datetime(dates["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.dataframe(dates.sort_values("date", ascending=False), hide_index=True, use_container_width=True)
            available = sorted(dates["date"].dropna().unique(), reverse=True)
            selected_date = st.selectbox("查看调仓后持仓", available) if available else None
            if not available:
                st.warning("调仓日期文件没有有效日期。")
            elif positions.empty:
                st.warning("该实验没有持仓明细文件。")
            else:
                positions["date"] = pd.to_datetime(positions["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                selected = positions[positions["date"] == selected_date].copy()
                preferred = [
                    "position_ordinal", "symbol", "rank", "score", "shares",
                    "raw_close_1500", "value", "weight", "buy_date",
                    "holding_days", "entry_rank", "avg_entry_price",
                ]
                columns = [column for column in preferred if column in selected.columns]
                st.dataframe(selected[columns], hide_index=True, use_container_width=True)
    with grid_tab:
        st.info("未找到 historical_grid_top20.csv。") if item["top20"].empty else st.dataframe(item["top20"], hide_index=True, use_container_width=True)
    with files_tab:
        manifest = item["manifest"]
        st.json(
            {
                "experiment_root": str(item["root"]),
                "forward_run": str(item["forward_run"]) if item["forward_run"] else None,
                "historical_run": str(item["historical_run"]) if item["historical_run"] else None,
                "target_col": manifest.get("target_col"),
                "fixed_signal_kind": manifest.get("fixed_signal_kind"),
                "fixed_signal_spec": manifest.get("fixed_signal_spec"),
                "historical_target_folds": manifest.get("historical_target_folds"),
                "historical_result_reused": manifest.get("historical_result_reused"),
                "strict_forward_start": manifest.get("strict_forward_start"),
                "strict_forward_end": manifest.get("strict_forward_end"),
            }
        )

with refresh_tab:
    st.subheader("刷新流程")
    st.code(
        "更新行情与 forward model_data 一次\n"
        "→ 为 r01/r05/r21 各生成一套最新 Top-5 Forward 预测\n"
        "→ 严格复用9组已完成的历史 Grid 冠军\n"
        "→ 重跑9次冻结参数的 strict-forward\n"
        "→ 导出每个调仓日的调仓后持仓\n"
        "→ 更新汇总、图表与前端",
        language="text",
    )
    st.markdown(
        "命令行全量刷新：\n"
        "```bash\n"
        "time bash scripts/run_ch17_as1455_full_rebuild.sh refresh-all-fixed-signals\n"
        "```\n"
        "前端按钮使用带文件锁的后台包装器，并在刷新前验证9组历史 Grid，因而不会意外重跑历史搜索。"
    )
    if status:
        st.json(status)

st.caption(f"页面读取时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
