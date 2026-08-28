#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.as1455_backtest_data import discover_experiment_names, parse_experiment_name
from dashboard.as1455_monitor_config import (
    DEFAULT_PRODUCTION_EXPERIMENT,
    load_monitor_experiments,
    save_monitor_experiments,
)

DEFAULT_MATRIX_ROOT = (
    PROJECT_ROOT
    / "saved_data"
    / "ashare_ml4t"
    / "ch17_as1455_global_fixed_signal_matrix"
    / "refresh_all_v1"
)

st.set_page_config(page_title="AS1455 盯盘模型配置", page_icon="👁️", layout="wide")
st.title("AS1455 盯盘模型配置")
st.caption(
    "这里仅控制前端展示哪些冻结策略，不改变14:55实盘生产策略。"
    "实盘生产始终固定为 21日目标 · 最优单模型（r21_best）。"
)

matrix_root = Path(
    st.sidebar.text_input(
        "9组回测目录",
        value=os.environ.get("AS1455_MATRIX_ROOT", str(DEFAULT_MATRIX_ROOT)),
    )
).expanduser().resolve()

if not matrix_root.is_dir():
    st.error(f"回测目录不存在：{matrix_root}")
    st.stop()

names = discover_experiment_names(matrix_root)
if not names:
    st.error("没有发现可配置的冻结策略。")
    st.stop()

labels: dict[str, str] = {}
for name in names:
    identity = parse_experiment_name(name)
    labels[name] = identity.display_name

current = load_monitor_experiments(matrix_root, names)
production_label = labels.get(DEFAULT_PRODUCTION_EXPERIMENT, DEFAULT_PRODUCTION_EXPERIMENT)

st.info(f"固定实盘策略：{production_label}。该策略始终保留在盯盘列表中，不能通过本页关闭。")

extra_names = [name for name in names if name != DEFAULT_PRODUCTION_EXPERIMENT]
current_extra = [name for name in current if name != DEFAULT_PRODUCTION_EXPERIMENT]
selected_extra = st.multiselect(
    "额外盯盘模型",
    options=extra_names,
    default=current_extra,
    format_func=lambda name: labels.get(name, name),
    help=(
        "默认不额外选择，因此每日14:55盯盘页只显示 r21_best。"
        "增加研究模型只改变页面希望展示的模型集合，不会改变实盘下单模型。"
    ),
)

selected = [DEFAULT_PRODUCTION_EXPERIMENT, *selected_extra]

c1, c2, c3 = st.columns(3)
c1.metric("当前盯盘模型", str(len(selected)))
c2.metric("固定实盘模型", "1")
c3.metric("额外研究模型", str(len(selected_extra)))

st.markdown("**保存后盯盘列表：**")
for name in selected:
    prefix = "[实盘]" if name == DEFAULT_PRODUCTION_EXPERIMENT else "[监控]"
    st.write(f"{prefix} {labels.get(name, name)}")

if st.button("保存盯盘模型", type="primary"):
    saved = save_monitor_experiments(matrix_root, selected, names)
    st.success(f"已保存 {len(saved)} 个盯盘模型。刷新主看板后生效。")

st.caption(
    "说明：当天14:55关键链只生成 r21_best 正式计划。若额外选择的研究模型当日尚无已落盘计划，"
    "主看板应只展示当前已存在的模型，并把缺失模型视为“尚未生成”，而不是把整个日期判失败。"
)
