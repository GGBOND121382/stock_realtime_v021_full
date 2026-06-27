# AS1455 live fast auto 一键入口

生产入口：

```bash
TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_v1.sh
```

默认 `auto` 模式：

1. 14:50 前启动：执行 `pre` + `prefast`，等待到 `COLLECT_START_TIME`，再执行 `postfast`。
2. 14:50 后启动：要求 `06_live_feature_state_fast.npz` 已存在，然后只执行 `postfast`。

分段命令：

```bash
TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_v1.sh pre
TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_v1.sh post
TRADE_DATE=today bash scripts/run_as1455_live_fast_auto_v1.sh status
```

关键产物：

- `06_live_feature_state_fast.npz`：14:55 前预计算 state。
- `11_live_model_features_for_prediction.csv`：14:55 后模型预测专用特征文件。
- `12_feature_build_report.json`、`13_live_feature_strict_validation_report.json`：fast finalize 校验报告。

不要把慢路径 `features` / `rebuild_features_strict_v2` 放到 14:55 生产流程里。
