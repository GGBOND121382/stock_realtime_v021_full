#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS1455 production model generation and rolling-period dashboard page."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.as1455_backtest_data import (  # noqa: E402
    discover_experiment_names,
    load_matrix_summary,
)
from dashboard.as1455_model_status import (  # noqa: E402
    attach_current_model_columns,
    load_dashboard_registry,
)
from utils.as1455_model_registry import DEFAULT_REGISTRY_ROOT  # noqa: E402

DEFAULT_MATRIX_ROOT = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_matrix"
    / "refresh_all_v1"
)

st.set_page_config(page_title="AS1455 模型与滚动更新", page_icon="🧠", layout="wide")
st.title("AS1455 模型与滚动更新")
st.caption(
    "历史 fold0..fold6 继续表示原时序交叉验证切分；生产模型使用 genNNN，"
    "每个 gen 服务一个 periodNNN，默认累计 63 个成功 live 交易日后滚动重训。"
)

matrix_root = Path(
    st.sidebar.text_input(
        "9组回测目录",
        value=os.environ.get("AS1455_MATRIX_ROOT", str(DEFAULT_MATRIX_ROOT)),
    )
).expanduser().resolve()
registry_root = Path(
    st.sidebar.text_input(
        "模型注册目录",
        value=os.environ.get("AS1455_MODEL_REGISTRY_ROOT", str(DEFAULT_REGISTRY_ROOT)),
    )
).expanduser().resolve()

if not matrix_root.is_dir():
    st.error(f"回测目录不存在：{matrix_root}")
    st.stop()

try:
    registry = load_dashboard_registry(registry_root)
except Exception as exc:
    st.error(f"模型注册表读取失败：{type(exc).__name__}: {exc}")
    st.stop()

summary = load_matrix_summary(matrix_root)
experiments = discover_experiment_names(matrix_root)
if summary.empty or len(experiments) != 9:
    st.error(f"需要完整9策略；summary={len(summary)} experiments={len(experiments)}")
    st.stop()

summary = attach_current_model_columns(summary, registry)
period = dict(registry.get("current_period") or {})
legacy_initialized = bool(period.get("legacy_cache_initialized")) or str(
    registry.get("active_generation")
) != "gen000"
observed = int(period.get("observed_days", 0) or 0)
required = int(period.get("required_days", registry.get("period_length", 63)) or 63)
remaining = max(0, required - observed)

m1, m2, m3, m4 = st.columns(4)
m1.metric("当前模型代", str(registry.get("active_generation") or "—"))
m2.metric("当前服务周期", str(period.get("period_id") or "—"))
m3.metric(
    "Forward进度",
    f"{observed}/{required} 交易日" if legacy_initialized else "待初始化",
)
m4.metric(
    "距离下次模型更新",
    ("已到期" if remaining == 0 else f"{remaining} 交易日")
    if legacy_initialized
    else "待初始化",
)
if not legacy_initialized:
    st.info(
        "gen000 的现有 strict-forward 历史尚未一次性并入 period000。"
        "运行滚动状态检查后，系统会读取既有 r01/r05/r21 fold0-forward 预测日期的交集，"
        "恢复真实进度和 gen000 的首次生产使用日期；不会训练模型。"
    )

st.subheader("9策略当前模型")
view = summary.copy()
if "display_name" not in view.columns:
    view["display_name"] = view["experiment"].astype(str)
source_labels = {
    "legacy_cv_fold": "历史 fold0 兼容模型",
    "legacy_explicit_model_dir": "历史 fold0 显式模型",
    "rolling_refit": "63日滚动重训",
}
if "model_source_type" in view.columns:
    view["model_source_type"] = view["model_source_type"].map(
        lambda value: source_labels.get(str(value), str(value) if pd.notna(value) else "—")
    )
columns = {
    "display_name": "策略",
    "target_label": "目标周期",
    "signal_label": "固定信号",
    "model_generation": "模型版本",
    "model_updated_date": "模型更新日期",
    "model_source_type": "模型来源",
    "model_train_start": "训练起点",
    "model_train_end": "训练截止",
    "rebalance_every": "调仓周期",
}
keep = [column for column in columns if column in view.columns]
view = view[keep].rename(columns=columns)
if "调仓周期" in view.columns:
    view["调仓周期"] = pd.to_numeric(view["调仓周期"], errors="coerce").map(
        lambda value: f"每{int(value)}日" if pd.notna(value) else "—"
    )
for column in ("模型版本", "模型更新日期", "训练起点", "训练截止"):
    if column in view.columns:
        view[column] = view[column].fillna("待初始化" if column == "模型更新日期" else "—").astype(str)
st.dataframe(view, hide_index=True, use_container_width=True)

st.caption(
    "模型更新日期=该 generation 首次实际用于生产交易信号的交易日。gen000 的日期从既有 strict-forward "
    "首个公共预测交易日恢复，不拿 fold0 的 test_end 冒充；gen001、gen002… 则在第一次成功 live 使用时记录。"
    "九策略 experiment 名、历史 Fold/Grid 和交易参数均不因此重命名。"
)

st.subheader("当前 period")
p1, p2, p3, p4 = st.columns(4)
p1.metric("开始日期", str(period.get("start_date") or ("待初始化" if not legacy_initialized else "—")))
p2.metric("最近成功 live 日", str(period.get("last_observed_date") or "—"))
p3.metric("累计成功日", str(observed) if legacy_initialized else "待初始化")
p4.metric("要求长度", str(required))
if observed and legacy_initialized:
    st.progress(min(1.0, observed / max(required, 1)))

st.subheader("模型代历史")
generation_rows: list[dict[str, Any]] = []
for generation in registry.get("generations", []):
    if not isinstance(generation, dict):
        continue
    targets = generation.get("targets") or {}
    generation_rows.append(
        {
            "模型版本": generation.get("generation_id"),
            "来源": source_labels.get(
                str(generation.get("source_type")), generation.get("source_type")
            ),
            "首次生产使用": generation.get("model_updated_date") or "待初始化",
            "来源周期": generation.get("source_period") or "—",
            "上一模型": generation.get("source_generation") or "—",
            "r01训练截止": (targets.get("r01_fwd") or {}).get("train_end") or "—",
            "r05训练截止": (targets.get("r05_fwd") or {}).get("train_end") or "—",
            "r21训练截止": (targets.get("r21_fwd") or {}).get("train_end") or "—",
            "训练完成": generation.get("trained_at") or "—",
        }
    )
if generation_rows:
    st.dataframe(pd.DataFrame(generation_rows), hide_index=True, use_container_width=True)
else:
    st.info("尚无模型代记录。")

st.subheader("滚动训练任务")
status_file = registry_root / ".dashboard" / "rollover_status.json"
if status_file.is_file():
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
        st.json(status)
        log_file = status.get("log_file")
        if log_file:
            log_path = Path(str(log_file))
            if not log_path.is_absolute():
                log_path = PROJECT_ROOT / log_path
            if log_path.is_file():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                st.code("\n".join(lines[-120:]), language="text")
    except Exception as exc:
        st.warning(f"滚动训练状态读取失败：{type(exc).__name__}: {exc}")
else:
    st.info("尚未运行滚动训练检查任务。")
