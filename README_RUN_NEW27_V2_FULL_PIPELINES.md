# New 27 Full V2 Pipeline

新增：

```text
scripts/run_new27_v2_full_pipelines.sh
```

这版脚本保持和现有 v2 pipeline 一致：

```text
pipelines/run_nextday_pipeline.py
--feature-pipeline fundamental,sector
--search-targets hit50,hit80,close_profit
--entry-policies vwap_low,all_days
--groups reversal_fundamental_regime,reversal_fundamental_regime_sector,reversal_fundamental_regime_sector_external,all_no_ak
--models xgb_d2_200_lr003_mcw5,xgb_d3_400_lr003_mcw3,xgb_d3_600_lr002_mcw3,xgb_d4_500_lr002_mcw5,lgbm_leaves7_400,lgbm_leaves15_700,extra_trees_600_d3,random_forest_600_d4
--quantiles 0.5,0.6,0.7,0.8
--train-rows 756
--valid-rows 126
--test-rows 63
--min-valid-trades 8
--min-train-entries 80
```

## 数据策略

不使用 `--samples-override`，不拿已有样本糊弄。

缺数据时由现有 pipeline 阶段通过 BaoStock / AKShare 获取：

```text
A 股日线 / ETF proxy：BaoStock / AKShare cache
THS 板块：AKShare stock_board_industry_index_ths / stock_board_concept_hist_ths
国内期货：AKShare futures_zh_daily_sina
港股/旧 external：沿用现有 legacy external builder
美股：仅 ai_compute profile 启用 yfinance，并在 builder 中强制 T-1 对齐
```

## 使用的 THS 板块名称

脚本中的 `--sector-symbol` 使用你提供的同花顺板块名称：

```text
银行
证券
保险
建筑装饰
电网设备
其他电源设备
电力
煤炭开采加工
光伏设备
能源金属
小金属
化学原料
农化制品
工业金属
工程机械
军工装备
医疗服务
软件开发
饮料制造
建筑材料
半导体
```

## 运行

先 dry-run：

```bash
cd /root/stock_realtime_v021_full

DRY_RUN=1 PYTHON=python3 END_DATE=2026-05-15 JOB_TIMEOUT=8h \
bash scripts/run_new27_v2_full_pipelines.sh
```

正式跑：

```bash
PYTHON=python3 END_DATE=2026-05-15 JOB_TIMEOUT=8h RUN_TAG=v2_new27_full \
bash scripts/run_new27_v2_full_pipelines.sh
```

如果怀疑旧缓存或旧输出不可信，可以强制刷新：

```bash
FORCE_REFRESH=1 RESUME=0 PYTHON=python3 END_DATE=2026-05-15 JOB_TIMEOUT=8h RUN_TAG=v2_new27_full_fresh \
bash scripts/run_new27_v2_full_pipelines.sh
```

## 查看失败

```bash
cat saved_data/model_search_queue_logs/new27_v2_new27_full_*/queue_summary.csv | grep -v ',ok,'
```

## 输出

每只标的输出到：

```text
saved_data/<code>_pipeline_out_<RUN_TAG>/
```

例如：

```text
saved_data/600584_pipeline_out_v2_new27_full/
```
