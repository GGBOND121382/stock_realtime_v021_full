# Today scoring pipeline fix

根目录解压后一键执行：

```bash
unzip -o today_scoring_pipeline_fix_patch.zip
bash scripts/apply_today_scoring_pipeline_fixes.sh
```

修复内容：

1. `amount` / `volume` / `prev_close` / `pct_chg` 不再因为训练样本列不存在而在 score 阶段丢失。
2. score 阶段会合并 `saved_data/akshare_realtime_cache/pending/<DATE>/<STOCK>/minute_bars_1min.csv` / `minute_bars_5min.csv`，补当日 `morning/afternoon/last_30m` 等盘中特征。
3. `sector_range_z20` 不再作为 realtime context 的 hard-required 特征，因为仅靠当天 THS summary 不能正确计算 20 日 z-score。
4. 尝试在 `all_scores.csv` 中新增 `missing_feature_names`。如果你本地代码结构和补丁模式不匹配，脚本不会硬失败，但会保留现有输出。
5. `collect_akshare_l1_cache.py` 的 core field alias 兼容英文归一化字段，避免 `last_price`/`amount` 明明存在却被误判缺失。

注意：

- 该补丁不移动/删除任何旧模型。
- 该补丁不重训模型。
- `sector_range_z20` 的正确长期方案是从历史 sector range 序列计算；本补丁先解除它对 14:55 信号的硬拦截。
- 如果当天采集不是从 09:30/09:35 开始，`first_30m_ret` / `first_60m_ret` 仍无法准确补齐。
