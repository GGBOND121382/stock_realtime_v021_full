# 多模型打分方案 B 补丁说明

## 目的

原交易日流水线默认是 `preferred`：每个标的只选择一个 artifact 打分。现在我们要让同一标的下的多个好模型都参与打分，例如：

- 平高：`vwap_low close` + `all_days close`
- 紫金：`vwap_low close` + `all_days close`
- 应流：external close + sector-only close
- 海大/云天化：close + hit80 辅助
- 恒瑞：hit80
- 牧原：观察 close

因此需要交易日流水线支持：

```bash
--model-policy all
```

## 覆盖文件

```text
pipelines/run_trading_day_signal_pipeline.py
scripts/run_trading_day_signal_pipeline_all_models.sh
scripts/run_trading_day_signal_and_portfolio_all_models.sh
README_PATCH_MODEL_POLICY_ALL.md
```

## 关键改动

`pipelines/run_trading_day_signal_pipeline.py` 新增参数：

```bash
--model-policy preferred|all
```

并将其传给：

```text
run_intraday_nextday_signals.py plan
collect_realtime_context.py collect-loop
collect_realtime_context.py build-features
run_intraday_nextday_signals.py score-now
```

这样：

```text
plan 阶段：按所有 artifact 生成 runtime_feature_requirements.csv 和 effective_watchlist.txt
context 阶段：按所有 artifact 的 feature_columns 采集上下文并构造 context_features_asof.csv
score 阶段：所有 artifact 都打分，all_scores.csv / buy_signals.csv 中可出现同一标的多行
portfolio 阶段：再由 PySCIPOpt 组合确认程序聚合同一标的多模型并选最终订单
```

## 使用方式

在项目根目录解压：

```bash
unzip -o trading_day_model_policy_all_patch.zip -d .
```

语法检查：

```bash
python3 -m compileall -q pipelines scripts
```

只跑多模型交易日打分：

```bash
python3 pipelines/run_trading_day_signal_pipeline.py \
  --watchlist selected_watchlist.txt \
  --models-dir saved_models \
  --model-policy all \
  --context-config configs/realtime_context_sources.toml \
  --cutoff-time 14:55 \
  --stock-collect-until 14:52 \
  --context-collect-until 14:52 \
  --build-time 14:52 \
  --score-time 14:54 \
  --spot-source-priority sina,ths,em,xq \
  --required-fields close,open,high,low,volume,amount \
  --xq-max-symbols-per-round 10 \
  --xq-per-symbol-timeout-seconds 2 \
  --stock-collect-wait-timeout-seconds 45 \
  --context-collect-wait-timeout-seconds 45 \
  --max-missing-features 5 \
  --min-amount-yuan 50000000
```

等价脚本：

```bash
chmod +x scripts/run_trading_day_signal_pipeline_all_models.sh
PYTHON=python3 ./scripts/run_trading_day_signal_pipeline_all_models.sh
```

## 与组合确认程序串行

如果你已经放好了：

```text
daily_portfolio_confirm_pyscipopt.py
portfolio_confirm_from_buy_signals.py
account.json
history_close.csv
```

可以直接运行：

```bash
chmod +x scripts/run_trading_day_signal_and_portfolio_all_models.sh
PYTHON=python3 ./scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

可选启用协方差风险惩罚：

```bash
USE_COVARIANCE_PENALTY=1 COV_RISK_AVERSION=3.0 PYTHON=python3 ./scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

## 输出变化

`--model-policy all` 后，`buy_signals.csv` 可能出现同一股票多行，例如：

```text
600312.SH / vwap_low close model
600312.SH / all_days close model
```

这是正常的。之后由：

```bash
portfolio_confirm_from_buy_signals.py
```

读取 `buy_signals.csv`，按同一股票聚合模型信号，再用 PySCIPOpt 限制最多 3 只股票并生成最终订单。

## 注意

如果你不接组合确认程序，只看 `buy_signals.csv` 手工交易，多模型模式会让候选行变多，不能简单逐行买入。方案 B 的正确链路是：

```text
all artifact score -> buy_signals.csv -> portfolio_confirm_from_buy_signals.py -> final orders
```
