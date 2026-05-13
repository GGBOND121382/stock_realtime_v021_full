# 实时上下文补丁：最终模型池 + all_days 候选 + 应流 external

## 目的

这版补丁让 README 里的交易日流水线在最终模型池下更接近“实盘可用”：

- 补齐 `600096.SH` 云天化、`601985.SH` 中国核电、`603308.SH` 应流股份的实时板块配置；
- 补齐应流 `ane_*` external 模型所需的 A 股同业、ETF、期货、同花顺板块实时上下文；
- 同时预留云天化 `fert_*`、中国核电 `pur_*` external 的实时上下文配置；
- 避开 EM/东方财富作为主实时源，A 股/ETF 相关上下文用 Sina targeted 小批量接口；
- THS 板块/概念上下文仍然只抓一张行业 summary，再筛选需要的板块；
- 支持 basket 类特征，例如 `ane_stock_basket_close_ret5`、`ane_stock_vs_stock_basket_ret20`。

## 覆盖文件

```text
configs/realtime_context_sources.toml
data_collection/collect_realtime_context.py
pipelines/run_intraday_nextday_signals.py
scripts/check_realtime_context_coverage.py
```

## 使用方式

在项目根目录解压：

```bash
unzip -o realtime_context_official_models_patch.zip -d .
```

语法检查：

```bash
python3 -m compileall -q data_collection pipelines scripts
```

检查 saved_models 的实时上下文覆盖：

```bash
python3 scripts/check_realtime_context_coverage.py \
  --models-dir saved_models \
  --watchlist selected_watchlist.txt \
  --config configs/realtime_context_sources.toml \
  --model-policy all
```

如果输出 `OK: no missing realtime context config features.`，说明配置层面已覆盖模型需要的实时上下文。

## 交易日命令仍然使用 README 原命令

原来的交易日流水线命令不用换，只要 `--context-config configs/realtime_context_sources.toml` 指向这版配置即可。

```bash
python3 pipelines/run_trading_day_signal_pipeline.py \
  --watchlist selected_watchlist.txt \
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

## 关键采集范围

### 投资标的自身行情

最终模型池通常包括：

```text
600312.SH 平高电气
601899.SH 紫金矿业
603308.SH 应流股份
600096.SH 云天化
002311.SZ 海大集团
601985.SH 中国核电
600276.SH 恒瑞医药
002714.SZ 牧原股份
```

### 必要板块

```text
电网设备
贵金属 / 工业金属 / 小金属
通用设备
农化制品
农产品加工
化学制药
养殖业
电力
```

### 应流 ane_* external

```text
A股同业：600893.SH, 600765.SH, 000768.SZ, 003816.SZ, 601985.SH, 601611.SH, 300034.SZ
ETF：512660.SH, 512670.SH, 512710.SH
期货：NI0, SS0, AL0
板块：国防军工, 航空装备, 通用设备, 专用设备
```

## 注意

1. 是否真正采集某个 context group，由模型 `feature_columns.txt` 决定；没有对应前缀就不会采。
2. 美股不在这版实时采集补丁里。美股如果训练时使用 `us_lag_days >= 1`，交易日应按 T-1 缓存口径处理，不能用 A 股当天收盘后的美股数据。
3. 如果应流 external 模型仍然缺 `ane_*` 特征，先运行 `scripts/check_realtime_context_coverage.py` 看 `missing_context_config_features`。
