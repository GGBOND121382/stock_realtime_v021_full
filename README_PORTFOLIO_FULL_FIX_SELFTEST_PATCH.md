# Full portfolio fix with self-test

这个包应用时会自测，失败自动回滚。

修复内容：
- backtest 加载 samples 后调用 `add_trade_returns(...)`
- adapter 整体替换 `build_inputs()`，输出 `entry_policy / entry_vwap_premium_bps / samples / expected_return_col / metadata_path`
- optimizer 传播这些字段
- 修复 dataclass 字段顺序
- 修复 rejected_scores 的索引 bug
- synthetic adapter unit test

应用：

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_full_fix_selftest_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_full_fix_selftest_patch.sh
```

成功后再重跑回测与一致性审计。
