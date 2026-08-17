from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.as1455_rebalance_calendar import (  # noqa: E402
    DEFAULT_PRODUCTION_EXPERIMENT,
    build_rebalance_schedule,
    daily_rebalance_summary,
    query_baostock_trade_dates,
)

DEFAULT_MATRIX_ROOT = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_matrix"
    / "refresh_all_v1"
)

st.set_page_config(page_title="AS1455 调仓日历", page_icon="📅", layout="wide")


def _weekday_cn(value: pd.Timestamp) -> str:
    return "一二三四五六日"[pd.Timestamp(value).weekday()]


def _fmt_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "—" if pd.isna(parsed) else pd.Timestamp(parsed).strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _cached_trade_calendar(start_text: str, end_text: str) -> pd.DatetimeIndex:
    return query_baostock_trade_dates(start_text, end_text)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_schedule(
    matrix_root_text: str,
    trade_dates_tuple: tuple[str, ...],
    production_experiment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(trade_dates_tuple))).normalize()
    return build_rebalance_schedule(
        Path(matrix_root_text),
        dates,
        production_experiment=production_experiment,
    )


matrix_root = Path(
    st.sidebar.text_input(
        "9组策略目录",
        value=os.environ.get("AS1455_MATRIX_ROOT", str(DEFAULT_MATRIX_ROOT)),
    )
).expanduser().resolve()
production_experiment = os.environ.get(
    "AS1455_PRODUCTION_EXPERIMENT", DEFAULT_PRODUCTION_EXPERIMENT
)
horizon_sessions = st.sidebar.slider(
    "显示未来交易日",
    min_value=10,
    max_value=80,
    value=40,
    step=5,
)

st.title("AS1455 调仓日历")
st.caption(
    "调仓日期按每条 tracking 账户最后保存的 V7 相位继续推演，不从今天重新计数。"
    "未来交易日使用 BaoStock 交易日历；查询失败时不使用简单工作日近似。"
)
st.info(
    "标记为“实盘”的 r21_best 是当前14:55真实生产策略；其它8条继续作为研究/盯盘账户更新。"
)

if not matrix_root.is_dir():
    st.error(f"策略目录不存在：{matrix_root}")
    st.stop()

today = pd.Timestamp.now(tz=ZoneInfo("Asia/Shanghai")).tz_localize(None).normalize()
calendar_start = today - pd.Timedelta(days=730)
calendar_end = today + pd.Timedelta(days=240)

try:
    with st.spinner("读取 BaoStock A股交易日历……"):
        trade_dates = _cached_trade_calendar(
            calendar_start.strftime("%Y-%m-%d"),
            calendar_end.strftime("%Y-%m-%d"),
        )
except Exception as exc:
    st.error(
        "无法取得权威未来交易日历，因此没有生成调仓日期，避免节假日造成相位错位。"
        f"\n\n{type(exc).__name__}: {exc}"
    )
    st.stop()

try:
    schedule, status = _cached_schedule(
        str(matrix_root),
        tuple(pd.Timestamp(value).strftime("%Y-%m-%d") for value in trade_dates),
        production_experiment,
    )
except Exception as exc:
    st.error(f"调仓日历计算失败：{type(exc).__name__}: {exc}")
    st.stop()

if status.empty:
    st.error("没有发现可用于调仓日历的正式策略。")
    st.stop()

errors = status.loc[status["status"].astype(str).ne("ok")]
if not errors.empty:
    st.error(
        "以下策略 tracking 相位无法验证，因此没有为它们猜测调仓日期：\n\n"
        + "\n".join(
            f"- {row['display_name']}: {row['error']}" for _, row in errors.iterrows()
        )
    )

ok_status = status.loc[status["status"].astype(str).eq("ok")].copy()
if ok_status.empty or schedule.empty:
    st.stop()

# Surface stale tracking explicitly. The projection still counts every exchange
# trading date in the bridge, so its phase remains continuous.
prior_sessions = trade_dates[trade_dates < today]
expected_prior = pd.Timestamp(prior_sessions[-1]).normalize() if len(prior_sessions) else None
if expected_prior is not None:
    stale = ok_status.loc[
        pd.to_datetime(ok_status["latest_tracking_date"], errors="coerce").dt.normalize()
        < expected_prior
    ]
    if not stale.empty:
        st.warning(
            f"有 {len(stale)} 条策略尚未更新到最近完成交易日 {expected_prior:%Y-%m-%d}。"
            "日历仍按完整交易日序列延续相位，但建议检查9策略日更状态。"
        )

summary = daily_rebalance_summary(schedule)
summary = summary.loc[summary["date"].ge(today)].copy()
future_market_dates = trade_dates[trade_dates >= today][:horizon_sessions]
summary = summary.loc[summary["date"].isin(future_market_dates)].copy()

is_today_session = today in set(trade_dates)
next_market_date = (
    pd.Timestamp(future_market_dates[0]).normalize() if len(future_market_dates) else None
)

def _due_for(day: pd.Timestamp | None) -> pd.DataFrame:
    if day is None:
        return pd.DataFrame()
    return schedule.loc[
        schedule["date"].eq(pd.Timestamp(day).normalize())
        & schedule["is_rebalance_day"].astype(bool)
    ].copy()


today_due = _due_for(today) if is_today_session else pd.DataFrame()
next_due = _due_for(next_market_date)
production_today = bool(today_due["is_production"].astype(bool).any()) if not today_due.empty else False
production_next = bool(next_due["is_production"].astype(bool).any()) if not next_due.empty else False

m1, m2, m3, m4 = st.columns(4)
m1.metric("今天", f"{today:%Y-%m-%d} 周{_weekday_cn(today)}")
m2.metric(
    "今日需调仓",
    f"{len(today_due)} 条" if is_today_session else "非交易日",
)
m3.metric(
    "r21_best 今日",
    "需要调仓" if production_today else "无需调仓" if is_today_session else "非交易日",
)
m4.metric(
    "下一交易日",
    f"{_fmt_date(next_market_date)} · {len(next_due)}条" if next_market_date is not None else "—",
)

if production_today:
    st.success("今天 r21_best 是调仓日：14:55 生产链应生成该策略的调仓计划。")
elif is_today_session:
    st.caption("今天 r21_best 不是调仓日；14:55 仍可生成“继续持有/无需成交”的生产计划。")

st.subheader("未来交易日调仓表")
calendar_rows: list[dict[str, object]] = []
for _, row in summary.iterrows():
    day = pd.Timestamp(row["date"]).normalize()
    due_codes = list(row.get("rebalance_codes") or [])
    due_names = list(row.get("rebalance_experiments") or [])
    tagged: list[str] = []
    for code, name in zip(due_codes, due_names):
        tagged.append(
            ("[实盘] " if code == production_experiment else "[监控] ") + str(name)
        )
    calendar_rows.append(
        {
            "日期": day.strftime("%Y-%m-%d"),
            "星期": f"周{_weekday_cn(day)}",
            "需调仓策略数": int(row["rebalance_count"]),
            "需要调仓": "；".join(tagged) if tagged else "—",
            "r21_best": "调仓" if bool(row["production_rebalance"]) else "—",
        }
    )
st.dataframe(pd.DataFrame(calendar_rows), hide_index=True, use_container_width=True)

st.subheader("9策略下一次调仓")
status_view = ok_status.copy()
status_view["latest_tracking_date"] = status_view["latest_tracking_date"].map(_fmt_date)
status_view["next_rebalance_date"] = status_view["next_rebalance_date"].map(_fmt_date)
status_view["类型"] = status_view["is_production"].map(
    lambda value: "实盘" if bool(value) else "监控"
)
status_view = status_view.rename(
    columns={
        "display_name": "策略",
        "latest_tracking_date": "Tracking最新日",
        "rebalance_every": "周期(交易日)",
        "rebalance_offset": "当前相位offset",
        "next_rebalance_date": "下一调仓日",
    }
)[["类型", "策略", "Tracking最新日", "周期(交易日)", "当前相位offset", "下一调仓日"]]
st.dataframe(status_view, hide_index=True, use_container_width=True)

st.subheader("单策略未来调仓日期")
label_map = {
    ("[实盘] " if bool(row["is_production"]) else "[监控] ") + str(row["display_name"]): str(row["experiment"])
    for _, row in ok_status.iterrows()
}
selected_label = st.selectbox("选择策略", list(label_map))
selected_experiment = label_map[selected_label]
selected_dates = schedule.loc[
    schedule["experiment"].astype(str).eq(selected_experiment)
    & schedule["date"].ge(today)
    & schedule["is_rebalance_day"].astype(bool),
    "date",
].head(12)
if selected_dates.empty:
    st.info("当前查询范围内没有找到该策略后续调仓日。")
else:
    future_table = pd.DataFrame(
        {
            "调仓日期": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in selected_dates],
            "星期": [f"周{_weekday_cn(pd.Timestamp(value))}" for value in selected_dates],
        }
    )
    st.dataframe(future_table, hide_index=True, use_container_width=False)

st.caption(
    "这是调仓“计划日历”，表示策略在该交易日会执行一次排名检查/调仓逻辑；"
    "调仓日不保证一定产生买卖订单，因为现有持仓可能仍满足 sell-rank 条件且没有空仓位需要补。"
)
