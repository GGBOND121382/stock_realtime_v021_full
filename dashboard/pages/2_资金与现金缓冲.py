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
from dashboard.as1455_cash_replay import (  # noqa: E402
    replay_cash_requirements,
    replay_date_catalog,
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


def pct(value: Any) -> str:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(parsed) else f"{float(parsed) * 100:.1f}%"


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


@st.cache_data(show_spinner=False, ttl=300)
def _cached_replay_catalog(matrix_root_text: str, experiment: str) -> pd.DataFrame:
    return replay_date_catalog(Path(matrix_root_text), experiment)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_cash_replay(
    matrix_root_text: str,
    experiment: str,
    start_text: str,
    initial_cash: float,
    raw_daily_text: str,
    live_root_text: str,
) -> dict[str, Any]:
    return replay_cash_requirements(
        Path(matrix_root_text),
        experiment,
        pd.Timestamp(start_text),
        float(initial_cash),
        Path(raw_daily_text),
        live_root=Path(live_root_text),
    )


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
st.subheader("单策略现金缓冲诊断")
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

# Canonical Fold/Grid runs intentionally use compact retention by default, so
# order-level audit data may be absent even though NAV/summary artifacts exist.
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
historical_orders_retained = not report["historical_orders"].empty
forward_orders_retained = not report["forward_orders"].empty

s1, s2, s3 = st.columns(3)
s1.metric("当前持仓等股数涨停价估计", money(current_est.get("estimated_cash_required")))
s2.metric(
    "Canonical历史订单",
    "已留存" if historical_orders_retained else "compact未留存",
)
s3.metric(
    "Canonical Forward订单",
    "已留存" if forward_orders_retained else "compact未留存",
)

if not historical_orders_retained or not forward_orders_retained:
    st.info(
        "正式 Fold/Grid 与 strict-forward 默认使用 compact 输出：收益、NAV、回撤等仍完整，"
        "但逐笔 orders 可能被有意省略。因此不能从 canonical 文件直接反推出历史最高涨停价占款。"
        "下面使用已保存预测和冻结交易参数做独立账户 replay；不会重跑模型、Fold 或 Grid。"
    )
else:
    a1, a2 = st.columns(2)
    a1.metric("Canonical历史最高", money(report.get("historical_peak")))
    a2.metric("Canonical Strict Forward最高", money(report.get("forward_peak")))

st.subheader("按任意起算日重新回放涨停价占款")
st.caption(
    "这里的起算日只用于现金缓冲诊断，不会修改主看板真正的 tracking 起算日。"
    "回放从空仓和当前策略初始资金开始，保留冻结策略原有调仓 phase；"
    "若所选日期不是有效交易/预测日，则从其后的第一个有效日期开始，并等待原定调仓日才建仓。"
)

try:
    catalog = _cached_replay_catalog(str(matrix_root), selected_name)
except Exception as exc:
    st.error(f"无法解析该策略的历史/Forward预测区间：{type(exc).__name__}: {exc}")
    catalog = pd.DataFrame()

if not catalog.empty:
    catalog = catalog.copy()
    catalog["date"] = pd.to_datetime(catalog["date"], errors="raise").dt.normalize()
    min_date = pd.Timestamp(catalog["date"].min()).normalize()
    max_date = pd.Timestamp(catalog["date"].max()).normalize()
    default_date = start if start is not None else min_date
    default_date = min(max(pd.Timestamp(default_date).normalize(), min_date), max_date)

    segment_ranges = (
        catalog.groupby("segment", sort=False)["date"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    segment_ranges["范围"] = segment_ranges.apply(
        lambda row: f"{pd.Timestamp(row['min']):%Y-%m-%d} ~ {pd.Timestamp(row['max']):%Y-%m-%d}",
        axis=1,
    )
    segment_ranges = segment_ranges.rename(
        columns={"segment": "区间", "count": "交易日"}
    )[["区间", "范围", "交易日"]]
    with st.expander("可选历史 Fold / Forward 日期范围", expanded=False):
        st.dataframe(segment_ranges, hide_index=True, use_container_width=True)

    replay_start_date = st.date_input(
        "现金回放起算日（可选择任意 Fold 内日期）",
        value=default_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date(),
        key=f"cash_replay_start_{selected_name}",
    )
    requested_start = pd.Timestamp(replay_start_date).normalize()
    effective_rows = catalog.loc[catalog["date"].ge(requested_start)]
    if effective_rows.empty:
        st.warning("所选日期之后没有可回放数据。")
    else:
        effective_row = effective_rows.iloc[0]
        effective_preview = pd.Timestamp(effective_row["date"]).normalize()
        st.caption(
            f"请求起算日={requested_start:%Y-%m-%d}；"
            f"预计有效起算日={effective_preview:%Y-%m-%d}；"
            f"所在区间={effective_row['segment']}；"
            f"回放资金={money(configured_cash)}。"
        )

        request_key = (
            selected_name,
            requested_start.strftime("%Y-%m-%d"),
            float(configured_cash),
            str(raw_daily_root),
        )
        if st.button(
            "计算该起算日后的涨停价占款",
            type="primary",
            key="run_cash_replay",
        ):
            st.session_state["as1455_cash_replay_request"] = request_key

        if st.session_state.get("as1455_cash_replay_request") == request_key:
            try:
                with st.spinner("读取已有预测和日线，回放交易账户；不运行 TensorFlow / Fold / Grid……"):
                    replay = _cached_cash_replay(
                        str(matrix_root),
                        selected_name,
                        requested_start.strftime("%Y-%m-%d"),
                        float(configured_cash),
                        str(raw_daily_root),
                        str(live_root),
                    )
            except Exception as exc:
                st.error(f"现金回放失败：{type(exc).__name__}: {exc}")
            else:
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric(
                    "实际回放起点",
                    pd.Timestamp(replay["effective_start_date"]).strftime("%Y-%m-%d"),
                    replay.get("start_segment"),
                )
                r2.metric("期间最高涨停价占款", money(replay.get("overall_peak")))
                peak_date = replay.get("overall_peak_date")
                r3.metric(
                    "峰值日期",
                    pd.Timestamp(peak_date).strftime("%Y-%m-%d") if pd.notna(peak_date) else "—",
                    replay.get("overall_peak_segment") or None,
                )
                r4.metric("峰值 / 初始资金", pct(replay.get("peak_to_initial_cash")))
                r5.metric("回放末NAV", money(replay.get("final_nav")))

                if not bool(replay.get("complete", False)):
                    st.error(
                        "至少一个有买单的日期无法恢复完整涨停价；总峰值已 fail-closed，不应据此确定现金缓冲。"
                    )

                segments = replay["segments"].copy()
                if not segments.empty:
                    for column in ("start_date", "end_date", "peak_date"):
                        segments[column] = pd.to_datetime(
                            segments[column], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
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

                daily = replay["daily"].copy()
                if daily.empty:
                    st.info("所选区间没有产生买入订单，因此额外买入占款为0。")
                else:
                    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
                    curve = catalog.loc[catalog["date"].ge(pd.Timestamp(replay["effective_start_date"]))][
                        ["date", "segment"]
                    ].copy()
                    curve = curve.merge(
                        daily[["date", "conservative_cash_required"]],
                        on="date",
                        how="left",
                    )
                    curve["conservative_cash_required"] = pd.to_numeric(
                        curve["conservative_cash_required"], errors="coerce"
                    ).fillna(0.0)
                    st.line_chart(
                        curve.set_index("date")[["conservative_cash_required"]].rename(
                            columns={"conservative_cash_required": "涨停价最大占款（元）"}
                        ),
                        use_container_width=True,
                        height=360,
                    )

                    top = daily.copy()
                    top["conservative_cash_required"] = pd.to_numeric(
                        top["conservative_cash_required"], errors="coerce"
                    )
                    top = top.sort_values(
                        "conservative_cash_required", ascending=False
                    ).head(20)
                    label_map = catalog.set_index("date")["segment"].to_dict()
                    top["区间"] = top["date"].map(label_map)
                    top["日期"] = top["date"].dt.strftime("%Y-%m-%d")
                    top["涨停价最大占款"] = top["conservative_cash_required"].map(money)
                    top["买入金额"] = pd.to_numeric(
                        top["buy_amount"], errors="coerce"
                    ).map(money)
                    top["数据完整"] = top["cash_requirement_complete"].astype(bool)
                    st.markdown("**占款最高的20个交易日**")
                    st.dataframe(
                        top[["日期", "区间", "涨停价最大占款", "买入金额", "数据完整"]],
                        hide_index=True,
                        use_container_width=True,
                    )

                st.caption(
                    "诊断回放使用已保存的 one-fold-lag 历史预测 + fold0 strict-forward预测、"
                    "冻结交易参数和 BaoStock raw daily；起始账户为空仓，资金采用当前策略初始资金。"
                    "不训练模型、不重新做参数Grid，也不写回 canonical/tracking 文件。"
                )
                st.caption(
                    f"公司行动回放口径={replay.get('corporate_action_mode')}；"
                    "该口径用于保持实盘可执行的整数持仓语义。"
                )

st.divider()
st.subheader("当前持仓现金参考")
st.metric("当前持仓等股数涨停价估计", money(current_est.get("estimated_cash_required")))
st.caption(
    f"估计基准日={current_est.get('asof_date') or '—'}。该指标不是预测下一次一定会买哪些股票；"
    "它把当前持仓的相同股数按最新可得涨停价重新计价，手续费使用历史/forward/tracking中"
    "观察到的最高有效买入费率，仅用于准备现金规模的保守参考。"
)

if st.button("重新读取", use_container_width=False):
    st.cache_data.clear()
    st.session_state.pop("as1455_cash_replay_request", None)
    st.rerun()
