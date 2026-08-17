from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.as1455_manual_calibration import (  # noqa: E402
    DEFAULT_PRODUCTION_EXPERIMENT,
    apply_manual_calibration,
    calibration_history,
    editable_positions,
    load_manual_calibration_account,
)

DEFAULT_MATRIX_ROOT = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_matrix"
    / "refresh_all_v1"
)
DEFAULT_LIVE_ROOT = PROJECT_ROOT / "saved_data" / "ashare_ml4t" / "live_as1455"

st.set_page_config(page_title="AS1455 实盘账户手动校准", page_icon="🧭", layout="wide")


def money(value: Any) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(parsed) else f"{float(parsed):,.0f} 元"


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
experiment = st.sidebar.text_input(
    "生产策略",
    value=os.environ.get("AS1455_PRODUCTION_EXPERIMENT", DEFAULT_PRODUCTION_EXPERIMENT),
    disabled=True,
)

st.title("AS1455 实盘账户手动校准")
st.caption(
    "这个页面用于把 r21_best 的服务器 tracking 状态人工校准到券商真实状态。"
    "校准后，后续 tracking/live 会从校准后的现金和持仓继续演化；不自动读取券商账户。"
)

st.warning(
    "公司行动仍使用近似处理。若出现现金分红、送股/转增、配股、拆并股，或者任何部分成交、拒单、"
    "人工撤单导致服务器持仓/现金与券商不一致，请以券商实际状态为准在这里手动校准。"
)
st.info(
    "这里填写的是“策略子账户”口径，不是券商总账户。比如券商总资金约26万元、策略初始资金14万元时，"
    "不要把26万元全部填成策略现金；额外执行缓冲资金不参与策略 sizing。"
)

if not matrix_root.is_dir():
    st.error(f"策略目录不存在：{matrix_root}")
    st.stop()

try:
    state, positions, manifest = load_manual_calibration_account(matrix_root, experiment)
except Exception as exc:
    st.error(f"读取当前生产账户失败：{type(exc).__name__}: {exc}")
    st.stop()

if not state or state.get("status") != "ok":
    st.error("当前 r21_best tracking 最新状态未就绪，请先完成账户重建/增量刷新。")
    st.stop()

asof = pd.to_datetime(state.get("asof_date"), errors="coerce")
if pd.isna(asof):
    st.error("当前 tracking state 缺少有效 asof_date，不能校准。")
    st.stop()
asof = pd.Timestamp(asof).normalize()

m1, m2, m3, m4 = st.columns(4)
m1.metric("当前状态日期", asof.strftime("%Y-%m-%d"))
m2.metric("服务器策略NAV", money(state.get("nav")))
m3.metric("服务器策略现金", money(state.get("cash")))
m4.metric("服务器持仓数", str(int(state.get("n_positions", 0) or 0)))

source = state.get("account_state_source") or state.get("tracking_state_source") or "tracking_simulation"
last_id = state.get("manual_calibration_id") or manifest.get("last_manual_calibration_id")
st.caption(
    f"当前账户状态来源：{source}"
    + (f"；最近校准ID：{last_id}" if last_id else "；尚未记录手动校准")
)

st.subheader("录入券商真实的策略子账户状态")
st.caption(
    "校准日期固定为当前最新 tracking 日期，不允许从页面回改更早日期；这样可以避免破坏既有交易历史。"
    "如当日已经生成 execution_batch，保存校准时会自动撤销/归档该 READY 文件，之后需要重新生成今日策略。"
)

with st.form("manual_broker_calibration_form"):
    c1, c2, c3 = st.columns(3)
    c1.date_input("校准日期", value=asof.date(), disabled=True)
    strategy_nav = c2.number_input(
        "策略子账户 NAV / 总权益（元）",
        min_value=0.01,
        max_value=100_000_000.0,
        value=float(state.get("nav", 0.0) or 0.0),
        step=100.0,
        format="%.2f",
        help="只填写策略口径的总权益，不含你额外留作执行缓冲、且不希望参与策略 sizing 的资金。",
    )
    strategy_cash = c3.number_input(
        "策略子账户现金（元）",
        min_value=0.0,
        max_value=100_000_000.0,
        value=float(state.get("cash", 0.0) or 0.0),
        step=100.0,
        format="%.2f",
        help="不是券商总可用现金。应填写属于该策略账户口径的现金部分。",
    )

    st.markdown("**真实持仓**")
    edited = st.data_editor(
        editable_positions(positions),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "symbol": st.column_config.TextColumn("股票代码", help="例如 600000.SH / 000001.SZ"),
            "shares": st.column_config.NumberColumn("实际股数", min_value=1, step=1, format="%d"),
            "buy_date": st.column_config.DateColumn("真实买入日", format="YYYY-MM-DD"),
            "avg_entry_price": st.column_config.NumberColumn("券商成本价", min_value=0.001, step=0.01, format="%.4f"),
        },
        key="manual_calibration_positions",
    )
    note = st.text_input(
        "校准原因/备注",
        placeholder="例如：部分成交；送转后股数变化；人工撤单；例行对账",
    )
    confirm = st.checkbox(
        "我已确认：这里填写的是策略子账户状态，不是券商总账户26万元资金；持仓和股数以券商实际显示为准。"
    )
    submitted = st.form_submit_button(
        "保存手动校准并作为后续生产状态",
        type="primary",
        disabled=not confirm,
    )

if submitted:
    try:
        audit = apply_manual_calibration(
            matrix_root,
            experiment=experiment,
            asof_date=asof,
            strategy_cash=float(strategy_cash),
            strategy_nav=float(strategy_nav),
            positions=pd.DataFrame(edited),
            note=note,
            live_root=live_root,
        )
    except Exception as exc:
        st.error(f"校准失败，原状态未作为新生产状态提交：{type(exc).__name__}: {exc}")
    else:
        st.success(
            "校准已应用。后续 tracking/live 将从这个状态继续；"
            "如果今天已有 READY 订单且被归档，请重新生成今日策略。"
        )
        st.json(
            {
                "calibration_id": audit.get("calibration_id"),
                "asof_date": audit.get("asof_date"),
                "strategy_nav": audit.get("strategy_nav"),
                "strategy_cash": audit.get("strategy_cash"),
                "n_positions": audit.get("n_positions"),
                "archived_same_day_execution_batch": audit.get("archived_same_day_execution_batch"),
                "backup_dir": audit.get("backup_dir"),
            }
        )
        st.cache_data.clear()

st.divider()
st.subheader("最近手动校准记录")
history = calibration_history(matrix_root, experiment, limit=20)
if not history:
    st.info("尚无手动校准记录。")
else:
    rows = [
        {
            "校准ID": item.get("calibration_id"),
            "状态日期": item.get("asof_date"),
            "校准时间": item.get("calibrated_at"),
            "策略NAV": item.get("strategy_nav"),
            "策略现金": item.get("strategy_cash"),
            "持仓数": item.get("n_positions"),
            "备注": item.get("note"),
        }
        for item in history
    ]
    table = pd.DataFrame(rows)
    for column in ("策略NAV", "策略现金"):
        table[column] = pd.to_numeric(table[column], errors="coerce").map(money)
    st.dataframe(table, hide_index=True, use_container_width=True)
