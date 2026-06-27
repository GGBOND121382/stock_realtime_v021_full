# AS1455 live feature alignment V3 fixed

本版修复 V2 中读取训练 HDF sector 映射时，`symbol` 同时是 index level 和 column 导致 pandas groupby 歧义的问题。

# AS1455 live feature alignment v4

这个补丁把训练特征和盯盘特征的两个已确认不一致点修掉，并加严格自检。

## 已确认问题

1. `dollar_vol_rank` 全 NaN  
   `compute_ch12_features()` 中 `stack().swaplevel()` 把索引从 `(date, symbol)` 改成 `(symbol, date)`，赋值回 `prices` 时完全对不上，导致 live 的 `dollar_vol_rank` 全空。

2. `sector = -1`  
   `01_universe.csv` 里的 `code` 从 CSV 读成了整数，例如 `1`，而价格索引用的是 `"000001"`。旧 `sector_map` 以整数建索引，映射字符串代码时全失败。  
   v4 修复后会优先从训练 `model_data_as1455.h5` 读取 `symbol -> sector`，保证 live 的 sector 与训练数据一致。

## 安装

```bash
cd ~/stock_realtime_v021_full
unzip -oq as1455_live_feature_align_onekey.zip
bash as1455_live_feature_align_onekey/install.sh --repo .
```

## 用今天已经采到的数据重建特征，不重新采集

```bash
TRADE_DATE=20260625 bash scripts/run_as1455_live_rebuild_features_strict_v4.sh
```

成功标准：

- `12_feature_build_report.json` 中 `feature_passed = true`
- `13_live_feature_strict_validation_report.json` 中 `passed = true`
- `11_live_model_features.csv` 含 31 个训练特征列
- 训练特征列无 NaN
- `dollar_vol_rank` 与 `10_live_feature_panel_tail` 重算结果完全一致
- `sector` 与训练 `model_data_as1455.h5` 中的 sector 映射一致


## v4 说明

`11_live_model_features.csv` 保留全量 1000 行用于审计；如果少数股票因 sector 内 qcut 样本不足产生 `rXXq_sector` NaN，不再强行填充。

生产预测应使用：

```text
11_live_model_features_for_prediction.csv
```

该文件等同于 `11_live_model_features_usable.csv`，只包含 31 个训练特征完整非空的行。`12/13` 报告按 usable 文件判定是否可用于模型预测。
