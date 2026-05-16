# Strict THS Board Names Patch

这版只做一件事：把 active pipeline / external config 中不在你提供的 `ths_name` 列表里的板块名替换掉，避免 AKShare/THS 获取失败。

## 替换规则

```text
玻璃玻纤 -> 非金属材料
公用事业 -> 燃气
基础化工 -> 化学原料
工程建设 -> 建筑装饰
煤炭行业 -> 煤炭开采加工
饮料乳品 -> 饮料制造
```

其中前三个是 `feature_building/build_stock_external_features.py` 和 `configs/realtime_context_sources.toml` 内部 external profile 的板块名；后三个是早期 new27 wrapper/readme 中可能出现的旧名称。

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/ths_board_names_strict_patch.zip -d .
PYTHON=python3 bash scripts/apply_ths_board_names_strict_patch.sh
```

脚本会备份到：

```text
saved_data/patch_backups/ths_board_names_strict_YYYYMMDD_HHMMSS/
```

## 验证

```bash
grep -R -n -E '公用事业|玻璃玻纤|基础化工|工程建设|煤炭行业|饮料乳品' \
  feature_building/build_stock_external_features.py \
  configs/realtime_context_sources.toml \
  scripts/run_new27_v2_full_pipelines.sh \
  README_RUN_NEW27_V2_FULL_PIPELINES.md
```

没有输出才是正确结果。
