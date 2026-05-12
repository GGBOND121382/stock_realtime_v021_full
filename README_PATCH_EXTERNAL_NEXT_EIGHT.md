# Next-day external features v2 patch

本补丁面向两件事：

1. 为后续 8 个标的新增现实可落地的 external builder；
2. 为现有/已计划保留模型提供 v2 pipeline 与 v2 artifact 保存脚本，并与旧输出隔离。

## 0. 安装依赖

基础链路不新增强依赖，继续使用项目已有依赖：

```bash
python3 -m pip install -U akshare baostock pandas numpy scikit-learn xgboost lightgbm openpyxl joblib
```

如果你要给工业富联 `ai_compute` 加美股映射，安装可选依赖：

```bash
python3 -m pip install -U yfinance
```

没有安装 `yfinance` 时，只要不传 `--enable-us-yf`，pipeline 不会使用美股数据。

## 1. 解压补丁

在项目根目录执行：

```bash
unzip -o stock_external_nextday_v2_patch.zip -d .
```

语法检查：

```bash
python3 -m compileall -q feature_building model_training pipelines scripts
```

## 2. 输出隔离规则

`pipelines/run_nextday_pipeline.py` 新增：

```bash
--run-tag v2_models
```

如果没有显式传 `--out-root`，输出目录会从：

```text
saved_data/<code>_pipeline_out
```

变成：

```text
saved_data/<code>_pipeline_out_<run_tag>
```

例如：

```bash
python3 pipelines/run_nextday_pipeline.py \
  --symbol 600312.SH \
  --sector-symbol 电网设备 \
  --run-tag v2_models \
  --resume --excel
```

输出到：

```text
saved_data/600312_pipeline_out_v2_models/
```

旧目录 `saved_data/600312_pipeline_out/` 不会被覆盖。

## 3. 外部特征防泄露规则

新增 `feature_building/build_stock_external_features.py` 支持分来源 as-of lag：

```bash
--domestic-lag-days 0
--future-lag-days 1
--us-lag-days 1
```

默认含义：

| 来源 | 默认 lag | 说明 |
|---|---:|---|
| A股同业 / ETF / 同花顺板块 | 0 | 假设在 A 股收盘后做隔日决策，D 日收盘数据可用 |
| 国内期货 | 1 | 保守避开夜盘/结算口径不清晰 |
| 美股 yfinance | >=1 | 强制至少 T-1，避免使用 A 股 D 日收盘后的美股 D 日数据 |

即使误传：

```bash
--us-lag-days 0
```

代码内部也会强制变成 `>=1`。

`validation_report.json` 会写入：

```json
"lag_policy": {
  "domestic": 0,
  "futures": 1,
  "us": 1
}
```

## 4. 后续 8 个标的 pipeline

运行：

```bash
chmod +x scripts/run_next_eight_pipeline_with_external.sh
PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/run_next_eight_pipeline_with_external.sh
```

默认输出：

```text
saved_data/<code>_pipeline_out_v2_external/
```

默认 `601138.SH` 工业富联会启用 yfinance 美股映射。如果你不想用美股：

```bash
ENABLE_YF_FOR_AI=0 PYTHON=python3 END_DATE=2026-05-12 ./scripts/run_next_eight_pipeline_with_external.sh
```

映射关系：

| 标的 | external |
|---|---|
| `601138.SH` 工业富联 | `ai_compute` |
| `002080.SZ` 中材科技 | `material_wind_battery` |
| `601985.SH` 中国核电 | `power_utility_rate` |
| `600096.SH` 云天化 | `fertilizer` |
| `002518.SZ` 科士达 | `storage_power` |
| `603308.SH` 应流股份 | `aero_nuclear_equipment` |
| `600522.SH` 中天科技 | `optical_cable_grid` |
| `600487.SH` 亨通光电 | `optical_cable_grid` |

## 5. 现有/已计划保留模型的 v2 pipeline

运行：

```bash
chmod +x scripts/run_existing_models_v2_pipelines.sh
PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/run_existing_models_v2_pipelines.sh
```

默认输出：

```text
saved_data/<code>_pipeline_out_v2_models/
```

包含：

```text
002270.SZ 华明装备
002311.SZ 海大集团
002714.SZ 牧原股份
600276.SH 恒瑞医药
600312.SH 平高电气
601899.SH 紫金矿业
```

明确不包含：

```text
600176.SH 中国巨石
600309.SH 万华化学
```

## 6. 保存现有模型对应的 v2 artifact

先跑完第 5 步 pipeline，再执行：

```bash
python3 scripts/batch_save_existing_models_v2.py \
  --pipeline-run-tag v2_models \
  --out-dir saved_models_v2
```

默认保存到：

```text
saved_models_v2/<stock_code>/<artifact_name>_v2/
```

这样不会影响 `saved_models/` 中的 v1 模型，也不会被现有盘中脚本默认扫描到。

如果你确认要把 v2 放进现有 `saved_models/`，执行：

```bash
python3 scripts/batch_save_existing_models_v2.py \
  --pipeline-run-tag v2_models \
  --out-dir saved_models
```

artifact 名称仍然以 `_v2` 结尾，因此不会覆盖 `_v1`。

只保存某几只：

```bash
python3 scripts/batch_save_existing_models_v2.py --only 600312.SH,601899.SH
```

只打印命令不执行：

```bash
python3 scripts/batch_save_existing_models_v2.py --dry-run
```

## 7. 注意

- 本补丁不强行修改旧模型；v1 继续按旧特征口径使用。
- v2 的 pipeline 输出和模型 artifact 都可独立保存，便于 A/B 对比。
- 若 v2 存入 `saved_models/`，现有 `--model-policy preferred` 可能会因为 v2 更新时间更晚而优先选择 v2；如果只是观察，建议先存 `saved_models_v2/`。
