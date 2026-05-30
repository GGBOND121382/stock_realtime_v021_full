# stock_realtime 项目说明

这个目录是 A 股短线/次日交易模型的本地研究工作区。核心任务是从 BaoStock、AKShare、同花顺行业指数、外部行业/港股数据和本地 Level2 数据中构造特征，训练次日交易信号模型，并保存可复用的单股模型用于盘后/临近收盘打分。

本文档刻意保持简洁，方便后续 Codex 快速理解目录，不需要反复扫描大量输出文件。

## 警告

除了项目创建者本人，任何人/AI不得以任何形式删除该工程下的数据！！！！！！

## 核心数据流

```text
原始行情/外部数据
  -> data_collection/update_baostock_raw_cache.py / data_collection/collect_akshare_l1_cache.py / data88 提取脚本
  -> *_base_out, akshare_*_cache, PurchasedData
  -> feature_building/build_nextday_samples_from_baostock.py
  -> *_samples*_out/training_samples.csv
  -> feature_building/build_*_features.py
  -> *_fundamental*_out / *_sector*_out / *_external*_out / *_hog*_out
  -> model_training/optimize_nextday_vwap_model.py / model_training/search_walk_forward_model_complexity.py
  -> *_search_out / walk_forward_*_out
  -> model_saving/save_nextday_model.py / prediction/predict_saved_nextday_model.py
  -> saved_models / saved_data/trade_plots
```

## 主要脚本

### 目录结构

- `data_collection/`：联网或本地原始数据采集、data88 解包、BaoStock 原始缓存维护。
- `feature_building/`：样本构造、基本面/行业/外部特征构建。
- `model_training/`：模型实验、walk-forward 搜索、结果汇总。
- `model_saving/`：把选定模型训练并落盘为 `saved_models/<stock>/<artifact>/`。
- `prediction/`：加载已保存模型，对准备好的样本或最新未标注行打分。
- `visualization/`：交易信号、分数、收益曲线绘图。
- `pipelines/`：端到端流水线和交易时段调度入口。
- `bootstrap/`：旧版 BaoStock 双机会脚本、状态缓存 helper 和其依赖的 T 策略特征/回测模块；当前流水线仍用它的 `update_data` 模式生成基础日线/5 分钟线特征。
- `experiments/legacy/`：早期 VWAP、过滤器和退出方式实验，作为参考保留，不再放在主工作流根目录。
- `docs/`：Codex 任务记录、搜索计划、data88 工具说明等辅助文档。
- `requirements/`：按工具拆分的补充依赖文件。
- `saved_configs/`：保存可复现命令配置；每个 `.toml` 文件是一组带注释的命令，可用 `run_saved_config.py` 读取、打印或执行。

### 数据收集

- `data_collection/update_baostock_raw_cache.py`：增量拉取单只股票 BaoStock 日线和 5 分钟线，写入 `raw_cache`。
- `data_collection/collect_akshare_l1_cache.py`：采集 AKShare 实时/五档数据，主要用于 data88 尚未覆盖日期的临时推理数据。
- `data_collection/extract_data88_selected.py`、`data_collection/batch_extract_data88_selected.py`：解包/整理本地购买的 data88 Level2 数据。

### 特征构建

- `feature_building/build_nextday_samples_from_baostock.py`：用日线特征和 5 分钟线生成次日 VWAP/收盘/最高价标签，输出 `training_samples.csv`。
- `feature_building/build_fundamental_features.py`：构造点时间可用的基本面特征，来源包括 BaoStock 估值、季度财务和可选 AKShare 资金流。
- `feature_building/build_sector_features.py`：抓取并合并同花顺行业指数特征。
- `feature_building/build_hog_industry_features.py`：生猪/养殖行业外部特征。
- `feature_building/build_haida_feed_external_features.py`：海大/饲料相关外部特征。
- `feature_building/build_muyuan_hk_external_features.py`：牧原相关港股/行业外部特征。
- `feature_building/build_zijin_external_features.py`：紫金矿业相关金属、港股、行业外部特征。

### 模型训练

- `model_training/optimize_nextday_vwap_model.py`：主实验脚本，比较次日 VWAP/反弹过滤/收盘卖出等模型设计，输出预测、指标和特征重要性。
- `model_training/search_walk_forward_model_complexity.py`：固定 walk-forward 评估协议，搜索 XGBoost、LightGBM、ExtraTrees、RandomForest 等复杂度。
- `model_training/summarize_nextday_search_results.py`：汇总 pipeline/search 输出，生成 leaderboard。
- `experiments/legacy/ashare_xgb_nextday_vwap_return.py`：较早的次日 VWAP 收益预测实验。
- `experiments/legacy/vwap_nextday_rebound_backtest.py`、`experiments/legacy/vwap_nextday_model_filter_eval.py`、`experiments/legacy/vwap_nextday_exit_variants_eval.py`：次日 VWAP 交易过滤和退出方式评估。
- `bootstrap/t_strategy_backtest_cv5_split_eval.py`：较早的 T 策略/交叉验证回测脚本，现在作为 `bootstrap/` 数据构建链路的依赖保留。
- `docs/CODEX_TASK_PROMPT.md`、`docs/SEARCH_PLAN.md`、`docs/CURRENT_RESULTS_SUMMARY.md`：历史 H6 预测实验说明。注意：当前压缩包未包含旧版 `src/future_h6_codex_search.py`，需要恢复该脚本后才能复现实验。

### 保存模型

- `model_saving/save_nextday_model.py`：把当前较优的次日模型训练并保存为 `saved_models/<stock>/<artifact>/`。

### 预测

- `prediction/predict_saved_nextday_model.py`：加载已保存模型，对准备好的样本行打分，支持 `--allow-unlabeled` 给最新未标注日期预测。

### 绘图

- `visualization/plot_nextday_trade_signals.py`、`visualization/plot_saved_nextday_trade_signals.py`：把分数、买卖点、收益曲线画到 `saved_data/trade_plots/`。

### 全流水线

- `pipelines/run_nextday_pipeline.py`：单股端到端流水线，串联基础数据、样本、特征、搜索和汇总。
- `pipelines/run_intraday_nextday_signals.py`：交易时段调度脚本。读取 `selected_watchlist.txt`，启动时先增量补齐 BaoStock 5 分钟线，默认不用五档盘口，预测时复用缓存特征，并在收盘前五分钟对 `saved_models/` 中有模型的股票输出买入信号。

## 重要目录

- `*_base_out/`：单股基础行情与特征目录，常见文件有：
  - `<code>_daily.csv`
  - `<code>_5m.csv`
  - `daily_features.csv`
  - `intraday_features.csv`
  - `raw_cache/`
  - `feature_cache/`
- `*_samples*_out/`：训练样本，核心文件是 `training_samples.csv` 和 `validation_report.json`。
- `*_fundamental*_out/`：基本面增强样本和中间表。
- `*_sector*_out/`：行业指数增强样本和中间表。
- `*_external*_out/`、`*_hog*_out/`、`*_feed*_out/`：行业、港股、商品或业务相关外部特征。
- `*_search_out/`、`walk_forward_*_out/`：模型搜索/走步验证结果，通常包含：
  - `summary_*.json`
  - `summary_*.csv`
  - `metrics_*.csv`
  - `predictions_*.csv`
  - `feature_importance_*.csv`
- `saved_data/`：按标的或任务保存的数据产物根目录，例如 `<code>_pipeline_out/`、`akshare_realtime_cache/`、`intraday_nextday_signals/`、`trade_plots/`。
- `saved_models/`：可复用模型资产根目录，只放模型 artifact。典型结构：
  - `model.joblib`
  - `metadata.json`
  - `feature_columns.txt`
  - `feature_median.csv`
  - `validation_tail_predictions.csv`
  - `scores_*.csv`
- `saved_data/trade_plots/`：交易信号图和对应交易明细 CSV。
- `PurchasedData/`：本地购买/解包的 Level2 数据和说明。
- `saved_data/akshare_l1_cache/`、`saved_data/akshare_realtime_cache*/`：AKShare 采集的实时/五档缓存。
- `logs/`：早期批量搜索日志和 leaderboard。
- `data_collection/`：原始数据采集和 data88 解包脚本。
- `feature_building/`：样本与特征构建脚本。
- `model_training/`：训练、搜索和搜索结果汇总脚本。
- `model_saving/`：保存可复用模型资产的脚本。
- `prediction/`：加载保存模型并打分的脚本。
- `visualization/`：交易信号绘图脚本。
- `pipelines/`：端到端和交易时段调度入口。
- `bootstrap/`：流水线基础数据构建依赖的旧版 BaoStock/特征脚本。
- `experiments/legacy/`：早期或非主线实验脚本及小型结果表。
- `docs/`：辅助说明、历史任务记录和 data88 工具文档。
- `requirements/`：工具级依赖清单。
- `saved_configs/`：命令配置目录，当前包含 `600312_pipeline.toml`、`600312_data_collection.toml`、`600312_train_save.toml`、`600312_predict.toml`。
- `.venv/`、`.matplotlib/`、`.idea/`、`__pycache__/`：环境/IDE/缓存目录，理解业务时通常不用看。

## 当前重点股票与命名

目录名通常以股票代码或别名开头：

- `002311_*`：海大集团相关实验。
- `002714_*`：牧原/猪肉链相关实验，已有多个保存模型。
- `600176_*`：中国巨石相关实验，当前目录里有增强搜索输出。
- `600276_*`：恒瑞医药相关实验。
- `zijin_601899_*`：紫金矿业相关实验。

常见后缀含义：

- `_current_out`：当前一轮样本或特征输出。
- `_latest_unlabeled_*`：包含最新未标注日期，用于推理。
- `_50bps`、`_80bps`、`_100bps`：目标收益阈值，单位 bps。
- `_close_profit`：以次日收盘卖出盈利为标签/评估目标。
- `_noleak`、`strict`：强调避免未来信息泄露或更严格评估。

## 常用命令模板

以下只作路径结构参考，实际股票和目录按当前实验替换：

```powershell
# 1. 更新 BaoStock 原始缓存
.\.venv\Scripts\python.exe data_collection\update_baostock_raw_cache.py --symbol 600176 --start-date 2026-04-01 --end-date 2026-05-07 --raw-cache-dir saved_data/600176_base_out/raw_cache

# 2. 从基础特征生成次日样本
.\.venv\Scripts\python.exe feature_building\build_nextday_samples_from_baostock.py --daily-features saved_data/600176_base_out/daily_features.csv --intraday-bars saved_data/600176_base_out/600176_5m.csv --out-dir saved_data/600176_samples_current_out --keep-unlabeled-tail

# 3. 加基本面或行业特征
.\.venv\Scripts\python.exe feature_building\build_fundamental_features.py --symbol 600176 --daily-samples saved_data/600176_samples_current_out/training_samples.csv --out-dir saved_data/600176_fundamental_current_out
.\.venv\Scripts\python.exe feature_building\build_sector_features.py --samples saved_data/600176_fundamental_current_out/training_samples_with_fundamentals.csv --out-dir saved_data/600176_sector_current_out --sector-symbol 建筑材料

# 4. 搜索/评估模型
.\.venv\Scripts\python.exe model_training\search_walk_forward_model_complexity.py --samples saved_data/600176_sector_current_out/training_samples_with_sector.csv --intraday-bars saved_data/600176_base_out/600176_5m.csv --out-dir saved_data/600176_close_profit_current_search_out

# 5. 用已保存模型打分
.\.venv\Scripts\python.exe prediction\predict_saved_nextday_model.py --artifact-dir saved_models/002714.SZ/nextday_close_profit_xgb_d4_hk02714_force_v3 --stock-code 002714.SZ --samples saved_data/002714_hk_external_current_out/training_samples_with_hk_external.csv --allow-unlabeled

# 6. 交易时段采集 selected_watchlist，并在 14:55 输出隔日模型买入信号
.\.venv\Scripts\python.exe pipelines\run_intraday_nextday_signals.py collect-and-score --with-trades --signal-time 14:55

# 只测试当前已有数据的打分汇总，不联网采集
.\.venv\Scripts\python.exe pipelines\run_intraday_nextday_signals.py score-now --skip-final-collect --skip-build-bars

# 查看/复现保存的命令配置
.\.venv\Scripts\python.exe run_saved_config.py list
.\.venv\Scripts\python.exe run_saved_config.py show 600312_pipeline
.\.venv\Scripts\python.exe run_saved_config.py run 600312_train_save --dry-run
```

## 给后续 Codex 的低上下文读法

1. 先看这个 README，不要直接 `rg --files` 展开全部输出。
2. 明确目标股票后，只看对应前缀目录，例如 `600176_*`。
3. 找结果优先看 `summary_*.json`、`metrics_*.csv`、`metadata.json`，不要先打开大 CSV。
4. 查训练逻辑优先看 `model_training/optimize_nextday_vwap_model.py` 和 `model_training/search_walk_forward_model_complexity.py` 的函数/参数，不需要读完整输出目录。
5. 查线上/最新打分优先看 `saved_models/<stock>/<artifact>/metadata.json` 和 `prediction/predict_saved_nextday_model.py`。

## 注意事项

- 这是研究型工作区，输出目录很多，文件名比包结构更能说明实验 lineage。
- 多个脚本会联网拉 BaoStock/AKShare/同花顺数据；在受限环境里可能需要单独确认网络权限。
- 中文列名来源数据可能存在编码差异，修改脚本时优先保持已有输出格式和 `utf-8-sig` 保存习惯。
- 评估次日模型时要特别检查是否包含最新未标注行，以及是否有未来信息泄露。


## 命令

D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe model_training\summarize_nextday_search_results.py   --pipeline-out saved_data\600312_pipeline_out   --out-dir saved_data\600312_pipeline_out\99_summary   --excel


D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe  model_saving/save_nextday_model.py   --stock-code 600312.SH   --artifact-name nextday_all_days_close_profit_xgb_d3_reversal_fundamental_regime_v1   --samples saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv   --intraday-bars saved_data/600312_pipeline_out/00_base/600312_5m.csv   --out-dir saved_models   --feature-group reversal_fundamental_regime   --model-name xgb_d3_400_lr003_mcw3   --label-mode close_profit   --entry-policy all_days   --target-hit-bps 50   --round-trip-cost-bps 1.7   --valid-rows 252   --min-train-entries 80   --min-valid-trades 8   --quantiles 0.5,0.6,0.7,0.8



D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe model_saving/save_nextday_model.py   --stock-code 600312.SH   --artifact-name nextday_vwap_low_close_profit_xgb_d3_reversal_fundamental_regime_v1   --samples saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv   --intraday-bars saved_data/600312_pipeline_out/00_base/600312_5m.csv   --out-dir saved_models   --feature-group reversal_fundamental_regime   --model-name xgb_d3_400_lr003_mcw3   --label-mode close_profit   --entry-policy vwap_low   --target-hit-bps 50   --round-trip-cost-bps 1.7   --valid-rows 252   --min-train-entries 80   --min-valid-trades 8   --quantiles 0.5,0.6,0.7,0.8


python3 run_saved_config.py list
python3 pipelines/run_nextday_pipeline.py --help
python3 model_saving/save_nextday_model.py --help
python3 prediction/predict_saved_nextday_model.py --help




---

## 交易日 14:55 实盘信号流水线

本节说明每天盘前、盘中如何运行，以及在哪里查看买入信号。核心原则：

```text
盘前：只更新历史数据、样本、基本面/板块/外部历史特征；不做实时信号。
盘中：提前采集实时快照和实时上下文；14:55 前只做快速 score。
14:55：不再联网做慢采集，不再 build-bars，只读取 cutoff 前缓存输出 buy_signals.csv。
```

### 0. 推荐一键入口：信号 + 组合确认

每天实盘优先使用这个入口，它会串联：

```text
实时股票快照采集
实时板块/外部上下文采集
5min bar 构建与 OHLCV 修正
score-now 生成 all_scores / buy_signals / rejected_scores
portfolio optimizer 生成最终组合订单
```

推荐命令：

```bash
PYTHON=python3 bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

端到端对比 pipeline、盯盘采集、BaoStock、预测特征、信号和 portfolio：

```bash
PYTHON=python3 bash scripts/compare_today_collected_vs_baostock.sh
```

默认会先调用 `pipelines/run_premarket_history_update.py` 补齐 pipeline 历史/样本/fundamental/sector/external 数据，再做对账。只对已有文件做离线对比时：

```bash
RUN_PREMARKET_UPDATE=0 PYTHON=python3 bash scripts/compare_today_collected_vs_baostock.sh
```

指定交易日、cutoff 和标的：

```bash
DATE=20260520 CUTOFF_TIME=14:55 SYMBOLS=600487.SH,300308.SZ PYTHON=python3 \
bash scripts/compare_today_collected_vs_baostock.sh
```

跳过 BaoStock 联网查询，只比较盯盘缓存、pipeline 文件、预测特征、信号和 portfolio：

```bash
RUN_PREMARKET_UPDATE=0 SKIP_BAOSTOCK_QUERY=1 DATE=20260520 PYTHON=python3 \
bash scripts/compare_today_collected_vs_baostock.sh
```

默认读取 `saved_data/akshare_realtime_cache`，输出到 `saved_data/baostock_compare/${DATE}/`：

```text
comparison_summary.csv          # 盯盘 5m vs BaoStock 5m
pipeline_file_inventory.csv     # pipeline 文件清单、行数、是否含当日 date
pipeline_vs_collected_daily.csv # pipeline 日线/汇总 vs 盯盘 daily_features
pipeline_vs_collected_5m.csv    # pipeline 5m raw_cache vs 盯盘 minute_bars_5min
prediction_feature_diff.csv     # pipeline 样本行 vs 盯盘 overlay 后预测特征逐字段差异
prediction_signal_diff.csv      # 两套预测特征对应的模型分数/阈值通过情况
portfolio_file_inventory.csv    # 信号和 portfolio 产物文件清单
portfolio_signal_diff.csv       # buy_signals 与 portfolio selected/orders 标的差异
comparison_summary.json
```

指定交易日回放/补跑时同时指定紧凑日期和横线日期：

```bash
DATE_COMPACT=20260515 DATE_DASH=2026-05-15 PYTHON=python3 \
bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

该入口内部会把 `DATE_COMPACT` 传给信号流水线的 `--date`，并把 `DATE_DASH` 传给组合确认模块，避免信号目录日期和组合报告日期错位。

默认实时股票源为：

```text
sina_batch,ths_etf,xq
```

含义：

```text
sina_batch：A 股/ETF 小批量目标代码实时源，默认主源
ths_etf：THS ETF 表，只用于 ETF 补充
xq：雪球慢速补洞源，只补仍缺核心字段的少量标的
```

不建议在 14:55 主流程中使用 `em` 或全市场接口。`em` 在实时路径中被显式禁用，`em_full` 只适合人工诊断，不适合作为临近收盘主流程数据源。


### 1. 盘前：更新 BaoStock / 样本 / 历史特征

建议在前一晚收盘后或交易日 09:00~09:15 执行：

```bash
cd /opt/stock_realtime/stock_realtime
source .venv/bin/activate

python pipelines/run_premarket_history_update.py   --models-dir saved_models   --saved-data-dir saved_data   --context-config configs/realtime_context_sources.toml   --end-date today   --cache-mode incremental   --feature-cache-mode incremental   --keep-going
```

该脚本会扫描 `saved_models/`，只更新已有模型的标的；不会重新搜索模型、不会保存模型、不会输出交易信号。输出报告在：

```text
saved_data/premarket_history_update/YYYYMMDD/
  premarket_update_plan.csv
  premarket_history_update_report.csv
  premarket_history_update_report.json
  premarket_history_update.log
```

盘前更新后，建议确认 saved model 指向的样本文件存在：

```bash
python - <<'PY'
import json
from pathlib import Path

for p in Path("saved_models").glob("*/*/metadata.json"):
    m = json.loads(p.read_text(encoding="utf-8"))
    s = Path(m["samples"])
    print(p)
    print("samples:", s)
    print("exists:", s.exists())
    print()
PY
```

所有模型都应显示 `exists: True`。如果路径指向旧工程目录，应重新保存模型或修正 `metadata.json`。

### 2. 盘中：一键采集并在 14:55 前输出信号

建议在 14:25~14:35 启动。默认流程会：

```text
plan：扫描 saved_models，只生成有模型标的的 effective_watchlist
股票实时采集：只采 effective_watchlist；多源补洞；核心字段完整后短路
板块/外部上下文采集：按 configs/realtime_context_sources.toml 和模型 feature_columns 动态决定
build-bars：提前构造股票盘中特征，严格使用 cutoff 前数据
build context features：提前构造板块/外部 as-of 特征
score-now：14:54/14:55 只打分并输出买入信号
```

推荐命令：

```bash
python3 pipelines/run_trading_day_signal_pipeline.py \
  --watchlist selected_watchlist.txt \
  --context-config configs/realtime_context_sources.toml \
  --cutoff-time 14:55 \
  --stock-collect-until 14:52 \
  --context-collect-until 14:52 \
  --build-time 14:52 \
  --score-time 14:54 \
  --spot-source-priority sina_batch,ths_etf,xq \
  --required-fields close,open,high,low,volume,amount \
  --xq-max-symbols-per-round 10 \
  --xq-per-symbol-timeout-seconds 2 \
  --stock-collect-wait-timeout-seconds 45 \
  --context-collect-wait-timeout-seconds 45 \
  --max-missing-features 5 \
  --min-amount-yuan 50000000
```

```bash
python3 pipelines/run_trading_day_signal_pipeline.py   --watchlist selected_watchlist.txt   --models-dir saved_models   --model-policy all   --context-config configs/realtime_context_sources.toml   --cutoff-time 14:55   --stock-collect-until 14:52   --context-collect-until 14:52   --build-time 14:52   --score-time 14:54   --spot-source-priority sina_batch,ths_etf,xq   --required-fields close,open,high,low,volume,amount   --xq-max-symbols-per-round 10   --xq-per-symbol-timeout-seconds 2   --stock-collect-wait-timeout-seconds 45   --context-collect-wait-timeout-seconds 45   --max-missing-features 5   --min-amount-yuan 50000000
```

先检查命令链可用性：

```bash
python pipelines/run_trading_day_signal_pipeline.py \
  --watchlist selected_watchlist.txt \
  --context-config configs/realtime_context_sources.toml \
  --cutoff-time 14:55 \
  --stock-collect-until 14:52 \
  --context-collect-until 14:52 \
  --build-time 00:00 \
  --score-time 00:00 \
  --dry-run
```

### 3. 何时、何处查看买入信号

交易日当天的信号输出目录：

```text
saved_data/intraday_nextday_signals/YYYYMMDD/
```

核心文件：

```text
buy_signals.csv       # 只包含真正通过过滤的买入候选
all_scores.csv        # 所有模型打分和诊断信息
rejected_scores.csv   # 未通过的模型结果及拒绝原因
run_summary.json      # 本次打分摘要
trading_day_pipeline.log
trading_day_pipeline_summary.json
```

查看买入候选：

```bash
cat saved_data/intraday_nextday_signals/$(date +%Y%m%d)/buy_signals.csv
```

更清晰地打印：

```bash
python - <<'PY'
import pandas as pd
from datetime import datetime
from pathlib import Path

p = Path(f"saved_data/intraday_nextday_signals/{datetime.now():%Y%m%d}/buy_signals.csv")
if not p.exists():
    print("buy_signals.csv not found:", p)
else:
    df = pd.read_csv(p)
    cols = [c for c in [
        "rank", "stock_code", "artifact_name", "entry_policy",
        "close", "daily_vwap", "hit_score", "threshold", "score_margin",
        "snapshot_time", "cutoff_time", "source_used", "context_status",
        "missing_feature_count", "amount", "reject_reason"
    ] if c in df.columns]
    print(df[cols].to_string(index=False))
PY
```

### 4. `buy_signals.csv` 字段说明

`buy_signals.csv` 是干净候选列表，只保留：

```text
signal=True
error 为空
日期为当天
股票核心实时字段完整
上下文特征满足模型要求
缺失特征数量不超过 --max-missing-features
成交额不低于 --min-amount-yuan
```

常见字段：

| 字段 | 含义 |
|---|---|
| `rank` | 按 `score_margin` 降序排序后的候选排名 |
| `trade_date` | 信号所属交易日 |
| `stock_code` | 股票代码 |
| `artifact_name` | 触发信号的模型 artifact |
| `entry_policy` | `all_days` 或 `vwap_low` |
| `close` | cutoff 前最新价格/近似收盘价 |
| `daily_vwap` | cutoff 前累计成交额/成交量估算 VWAP |
| `hit_score` | 模型输出概率/分数 |
| `threshold` | 保存模型时确定的交易阈值 |
| `score_margin` | `hit_score - threshold`，越大越强 |
| `entry_signal` | 是否满足入场规则，如 `vwap_low` 的 VWAP 低吸条件 |
| `signal_raw_score_pass` | 分数是否超过阈值 |
| `signal` | 最终是否给出买入信号 |
| `snapshot_time` | 用于评分的股票快照时间，必须不晚于 `cutoff_time` |
| `cutoff_time` | 评分数据硬截止时间 |
| `source_used` | 股票实时数据实际使用的数据源，例如 `sina,ths,xq` |
| `core_complete` | 个股实时核心字段是否完整 |
| `missing_core_fields` | 缺失的核心字段；买入候选中应为空 |
| `context_status` | 板块/外部上下文状态：`ok` / `not_required` / `partial` / `missing` |
| `missing_context_features` | 缺失的上下文特征；买入候选中应为空或非阻断 |
| `missing_feature_count` | 模型输入中被中位数填充的特征数量 |
| `amount` | cutoff 前成交额，用于流动性过滤 |
| `reject_reason` | 买入候选中通常为空；拒绝原因见 `rejected_scores.csv` |

### 5. `all_scores.csv` 和 `rejected_scores.csv` 怎么看

如果 `buy_signals.csv` 为空，先看：

```bash
python - <<'PY'
import pandas as pd
from datetime import datetime
from pathlib import Path

p = Path(f"saved_data/intraday_nextday_signals/{datetime.now():%Y%m%d}/all_scores.csv")
df = pd.read_csv(p)
cols = [c for c in [
    "stock_code", "artifact_name", "hit_score", "threshold", "score_margin",
    "signal", "reject_reason", "date_status", "core_complete",
    "missing_core_fields", "context_status", "missing_context_features",
    "missing_feature_count", "source_used", "snapshot_time"
] if c in df.columns]
print(df[cols].to_string(index=False))
PY
```

常见拒绝原因：

| 拒绝原因 | 含义 |
|---|---|
| `score_below_threshold` | 模型分数未过阈值 |
| `entry_signal_false` | 不满足入场规则 |
| `missing_core_fields` | 个股实时核心字段不完整 |
| `missing_required_realtime_context` | 模型需要的板块/外部上下文没有估算成功 |
| `filled_features_gt_5` | 缺失特征填充过多，超过 `--max-missing-features` |
| `amount_lt_50000000` | 成交额低于过滤阈值 |
| `not_exact_trade_date` | 没有用到当天样本，不能实盘采用 |
| `unsupported_realtime_features` | 模型依赖当前实时系统不支持的盘口/资金流字段 |

### 6. 时间要求

不要在 14:55 后再采集慢数据。默认推荐：

```text
14:25~14:35  启动 run_trading_day_signal_pipeline.py
14:52        股票采集和上下文采集停止
14:52        build-bars / build context features
14:54        score-now
14:55 前后    查看 buy_signals.csv
```

如果采集进程在 build-time 后仍未退出，主流水线会按超时参数终止采集并继续后续步骤，避免再次出现 15:00 后才出信号。



## 补丁260516
cd /root/stock_realtime_v021_full

# 1. 应用大补丁
unzip -o /path/to/big_safe_model_retrain_patch.zip -d .
bash scripts/apply_big_safe_patch.sh

# 2. 覆盖 603308 脚本为 no-search 版本
unzip -o /path/to/big_safe_model_retrain_patch_v2_no_search.zip -d .
bash scripts/apply_603308_no_search_patch.sh

# 3. 重构 603308 训练数据，不跑 search，并保存 603308 模型
PYTHON=python3 END_DATE=2026-05-15 bash scripts/rebuild_603308_pipeline_safe.sh

# 4. 先 dry-run 检查其他好模型
DRY_RUN=1 SKIP_PIPELINE=1 PYTHON=python3 END_DATE=2026-05-15 \
ONLY=600522.SH,600487.SH,600312.SH,601899.SH,603308.SH \
bash scripts/update_ranked_models_latest.sh

# 5. 正式保存其他好模型
SKIP_PIPELINE=1 PYTHON=python3 END_DATE=2026-05-15 \
ONLY=600522.SH,600487.SH,600312.SH,601899.SH,603308.SH \
bash scripts/update_ranked_models_latest.sh

# 6. 验证采集配置
python3 - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
tomllib.load(open("configs/realtime_context_sources.toml", "rb"))
print("TOML OK")
PY

# 7. 修当天 5min bar
python3 tools/fix_5m_ohlcv_from_snapshots.py \
  --date 20260515 \
  --cache-dir saved_data/akshare_realtime_cache \
  --symbols-file saved_data/intraday_nextday_signals/20260515/effective_watchlist.txt \
  --cutoff-time 14:55

# 8. 检查实盘 context plan
python3 data_collection/collect_realtime_context.py plan \
  --watchlist selected_watchlist.txt \
  --models-dir saved_models \
  --model-policy all \
  --config configs/realtime_context_sources.toml \
  --out-dir saved_data/realtime_context \
  --date 20260515 \
  --cutoff-time 14:55 \
  --refresh-plan

### 每日流程关键检查点

1. `scripts/run_trading_day_signal_and_portfolio_all_models.sh` 是推荐的一键入口。
2. `pipelines/run_intraday_nextday_signals.py` 的默认 `--spot-source-priority` 已对齐为 `sina_batch,ths_etf,xq`。
3. 历史补跑时必须同时设置：
   - `DATE_COMPACT=YYYYMMDD`
   - `DATE_DASH=YYYY-MM-DD`
4. 信号输出目录：
   - `saved_data/intraday_nextday_signals/YYYYMMDD/all_scores.csv`
   - `saved_data/intraday_nextday_signals/YYYYMMDD/buy_signals.csv`
   - `saved_data/intraday_nextday_signals/YYYYMMDD/rejected_scores.csv`
5. 组合输出目录：
   - `portfolio_reports/daily_portfolio_orders_YYYY-MM-DD.csv`
   - `portfolio_reports/daily_portfolio_selected_YYYY-MM-DD.csv`
   - `portfolio_reports/daily_portfolio_rejected_YYYY-MM-DD.csv`
   - `portfolio_reports/daily_portfolio_report_YYYY-MM-DD.json`
6. 清理模型库不是每日交易流程的一部分。`cleanup-apply` 只应在检查 `cleanup-preview` 报告后单独执行。




## 新增标的
601100
002297
000657
002601
600438
002460
603259
002261
002895
600919
600361
002028
600885
600030
601818
601336
605499
601390
601186
600016
000786
002128
002364
003816
601991
002518
600584

<<<<<<< HEAD

## 近一年回测（A; A+B）
python3 scripts/rolling_retrain_a_active_asof1455_backtest.py \
  --start-date 2025-05-27 \
  --end-date 2026-05-27 \
  --out-dir portfolio_reports/backtests/a_active_vs_backup_asof1455_1y \
  --keep-going
=======
# ML4T脚本
python3 scripts/backtest_ml4t_asof1455_lgbm.py \
  --sample-glob "saved_data/*_pipeline_out/01_samples_asof1455/training_samples_asof1455.csv" \
  --bars-glob "saved_data/*_pipeline_out/00_base/*_5m.csv" \
  --out-root saved_data/ml4t_asof1455_lgbm_pipeline_out \
  --entry-price-col close_asof1455 \
  --exit-price-col next_day_close \
  --entry-policy all_days \
  --round-trip-cost-bps 1.7 \
  --train-days 756 \
  --test-days 21 \
  --embargo-days 1 \
  --selection-rule strict_top_decile_positive \
  --min-pred-return-bps 0.0 \
  --max-positions 3 \
  --n-estimators 500 \
  --learning-rate 0.03 \
  --num-leaves 15 \
  --min-data-in-leaf 250 \
  --bagging-fraction 0.75 \
  --feature-fraction 0.75
>>>>>>> fd7c352016465c57b8366180a06f745a1066e355
