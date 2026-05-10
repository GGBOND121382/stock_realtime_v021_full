# 交给 Codex 的任务说明

你正在帮助我在本地 A 股 5 分钟数据上搜索一个最优模型。请直接在本地运行代码、读取数据、保存结果，不要只做理论分析。

## 背景

股票：`002714`

已有数据目录：

```text
dual_opp_out_002714_v12/
  signal_samples.csv
  intraday_features.csv
  daily_features.csv
  artifacts/
```

主输入优先使用：

```text
dual_opp_out_002714_v12/signal_samples.csv
```

如果路径不存在，请在当前工作区搜索 `signal_samples.csv`。

## 目标

预测当前 5 分钟 bar 之后未来 6 根 5 分钟 bar 的平均 `bar_vwap`：

```text
target_price(t) = mean(bar_vwap[t+1 : t+6])
```

也可以训练残差：

```text
delta(t) = target_price(t) - current_bar_vwap(t)
pred_target_price = current_bar_vwap(t) + pred_delta(t)
```

无论训练 price 还是 delta，最终都必须在 `target_price` 上评估误差。

## 成功标准

核心要求：

```text
valid_MAE < 0.05 元/股
test_MAE  < 0.05 元/股
```

辅助要求：

```text
test_p90_AE < 0.10
test_pct_abs_error_le_0.05 >= 65%
必须显著优于 baseline：pred=current_bar_vwap/current_close
```

如果找不到，请明确输出“当前特征和过滤范围下未找到达标模型”，并给出最优模型结果。

## 时间过滤限制

最多只能排除上午 10:00 前、下午 14:30 后。禁止只保留尾盘。

允许过滤：

```text
F0_all_day: 全日
F1_start1000: 10:00 之后
F2_end1430: 14:30 之前
F3_1000_1430: 10:00~14:30
F4_0945_1430: 09:45~14:30
F5_0950_1430: 09:50~14:30
F6_0955_1430: 09:55~14:30
```

## 请先运行这个脚本

包里有：

```text
src/future_h6_codex_search.py
```

推荐运行：

```bash
python src/future_h6_codex_search.py prepare --signal-samples dual_opp_out_002714_v12/signal_samples.csv --out-dir future_h6_codex_out
python src/future_h6_codex_search.py run-baselines --out-dir future_h6_codex_out
python src/future_h6_codex_search.py run-batch --out-dir future_h6_codex_out --batch et_time_filters
python src/future_h6_codex_search.py run-batch --out-dir future_h6_codex_out --batch lgbm_delta
python src/future_h6_codex_search.py run-batch --out-dir future_h6_codex_out --batch xgb_delta
python src/future_h6_codex_search.py run-batch --out-dir future_h6_codex_out --batch segments
python src/future_h6_codex_search.py leaderboard --out-dir future_h6_codex_out
```

如果脚本有问题，请修它，不要重写一堆不落盘的临时代码。

## 必须保存的结果

每个配置都要追加保存，不要等整个 batch 结束：

```text
future_h6_codex_out/results/*.csv
future_h6_codex_out/leaderboard_all.csv
future_h6_codex_out/best_config.json
future_h6_codex_out/predictions/best_valid_predictions.csv
future_h6_codex_out/predictions/best_test_predictions.csv
```

每行结果必须包含：

```text
config_id
filter
feature_set
target_kind
model
params
valid_MAE
test_MAE
valid_median_AE
test_median_AE
valid_p75_AE
test_p75_AE
valid_p90_AE
test_p90_AE
valid_p95_AE
test_p95_AE
valid_max_AE
test_max_AE
valid_pct_abs_error_le_0.05
test_pct_abs_error_le_0.05
valid_pct_abs_error_le_0.10
test_pct_abs_error_le_0.10
elapsed_seconds
```

## 当前已知结果，不要重复浪费时间

已知多组模型卡在 `test_MAE≈0.088`：

```text
baseline current_close: ~0.08823
ExtraTrees d8 leaf20 delta F1: ~0.08836
ExtraTrees d10 leaf10 delta F1: 0.08842
ExtraTrees d12 leaf10 delta F1: 0.08856
ExtraTrees d14 leaf5 delta F1: 0.08861
LightGBM leaves7 n20 delta F1: 0.08844
Ridge/BayesianRidge delta F1: ~0.08875
```

优先尝试：

```text
1. 同一模型换 F3/F4/F5/F6 时间过滤
2. LightGBM/XGB 小网格 delta
3. 分时段模型：10:00~11:30、13:00~14:30
4. ensemble
```

不要只看 `rank_ic`。本任务只看预测误差，特别是 MAE、p90、max AE。

## 最终输出

请最终给出：

```text
1. 是否找到 valid/test MAE < 0.05 的模型
2. 最优模型配置
3. valid/test 全部误差指标
4. 与 baseline 的提升幅度
5. 若未达标，判断瓶颈：特征不足/目标噪声/时间段差异/模型不足
6. 可复现实验命令
```

