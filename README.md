# stock_realtime 项目说明

这个目录是 A 股短线/次日交易模型的本地研究工作区。核心任务是从 BaoStock、AKShare、同花顺行业指数、外部行业/港股数据和本地 Level2 数据中构造特征，训练次日交易信号模型，并保存可复用的单股模型用于盘后/临近收盘打分。

本文档刻意保持简洁，方便后续 Codex 快速理解目录，不需要反复扫描大量输出文件。

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




D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe model_training\summarize_nextday_search_results.py   --pipeline-out saved_data\600312_pipeline_out   --out-dir saved_data\600312_pipeline_out\99_summary   --excel


D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe  model_saving/save_nextday_model.py   --stock-code 600312.SH   --artifact-name nextday_all_days_close_profit_xgb_d3_reversal_fundamental_regime_v1   --samples saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv   --intraday-bars saved_data/600312_pipeline_out/00_base/600312_5m.csv   --out-dir saved_models   --feature-group reversal_fundamental_regime   --model-name xgb_d3_400_lr003_mcw3   --label-mode close_profit   --entry-policy all_days   --target-hit-bps 50   --round-trip-cost-bps 1.7   --valid-rows 252   --min-train-entries 80   --min-valid-trades 8   --quantiles 0.5,0.6,0.7,0.8



D:\VSCodeWorkspace\stockAnalysis\.venv\Scripts\python.exe model_saving/save_nextday_model.py   --stock-code 600312.SH   --artifact-name nextday_vwap_low_close_profit_xgb_d3_reversal_fundamental_regime_v1   --samples saved_data/600312_pipeline_out/03_sector/training_samples_with_sector.csv   --intraday-bars saved_data/600312_pipeline_out/00_base/600312_5m.csv   --out-dir saved_models   --feature-group reversal_fundamental_regime   --model-name xgb_d3_400_lr003_mcw3   --label-mode close_profit   --entry-policy vwap_low   --target-hit-bps 50   --round-trip-cost-bps 1.7   --valid-rows 252   --min-train-entries 80   --min-valid-trades 8   --quantiles 0.5,0.6,0.7,0.8