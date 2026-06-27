# AS1455 live fast path

目标：14:55 后不再重算 252 天历史特征面板，只做今日行情注入、末端特征更新、横截面分箱和预测特征输出。

## 新入口

盘前/盘中，先跑正常 `pre` 并生成 fast state：

```bash
TRADE_DATE=20260625 bash scripts/run_as1455_live_prefast_v1.sh
```

14:55 后，采集并快速生成模型输入：

```bash
TRADE_DATE=20260625 bash scripts/run_as1455_live_postfast_v1.sh
```

若已经采集过 08 文件，只想回放 fast finalize：

```bash
SKIP_COLLECT=1 TRADE_DATE=20260625 bash scripts/run_as1455_live_postfast_v1.sh
```

## 关键产物

- `06_live_feature_state_fast.npz`：14:55 前生成的历史价格/sector/feature columns 状态。
- `06_live_feature_state_fast_report.json`：prefast 报告。
- `09_live_qfq_row_as1455.csv`：今日 live qfq 行。
- `11_live_model_features.csv`：全量审计文件。
- `11_live_model_features_for_prediction.csv`：模型预测专用文件，必须无缺失。
- `11_live_model_features_dropped_rows.csv`：因为 sector 内 qcut 样本不足等原因剔除的行。
- `12_feature_build_report.json`：fast finalize 报告。
- `13_live_feature_strict_validation_report.json`：严格校验报告。

## 时间约束

默认 `MAX_FINALIZE_SECONDS=40`。超过会失败；临时回放可用：

```bash
WARN_ONLY_TIME=1 SKIP_COLLECT=1 TRADE_DATE=20260625 bash scripts/run_as1455_live_postfast_v1.sh
```

## 注意

旧 `features` / `run_as1455_live_rebuild_features_strict_v2.sh` 是慢速审计/回填路径，不是 14:55 生产路径。
