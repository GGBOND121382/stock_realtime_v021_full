# Complete portfolio fix patch

完整补丁包含：

1. 旧回测目录移动到 `cleanup_trash`，不删除。
2. backtest 加载每个 artifact 的 samples 后，用 `add_trade_returns(...)` 重新生成 `entry_signal / trade_net_close_return / trade_target_or_close_return`。
3. adapter 传播 `entry_policy / entry_vwap_premium_bps / samples / expected_return_col / metadata_path`。
4. 修复 pandas `NaN -> "nan"` 字符串问题。
5. optimizer Candidate / orders 继续保留上述字段。
6. 修复 `TradeRecord` dataclass 默认字段顺序。
7. 修复 `rejected_scores` 的 `reset_index` 后索引错误。
8. 新增一致性审计脚本与深度验证脚本。

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_complete_fix_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_complete_fix_patch.sh
```

不清理旧目录：

```bash
CLEAN_OLD_BACKTEST=0 PYTHON=python3 bash scripts/apply_portfolio_complete_fix_patch.sh
```

多查模型样本：

```bash
VALIDATE_N=100 PYTHON=python3 bash scripts/apply_portfolio_complete_fix_patch.sh
```

## 重新跑回测

```bash
START_DATE=2026-01-05 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```

## 一致性审计

```bash
PYTHON=python3 bash scripts/run_portfolio_consistency_audit.sh
```

严格模式：

```bash
STRICT=1 PYTHON=python3 bash scripts/run_portfolio_consistency_audit.sh
```
