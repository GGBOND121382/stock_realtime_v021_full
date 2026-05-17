# Portfolio complete fix v3

完整修复包，应用时自测，失败自动回滚。

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_complete_fix_v3_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_complete_fix_v3.sh
```

默认会移动旧回测目录到 `cleanup_trash/`。不移动旧目录：

```bash
CLEAN_OLD_BACKTEST=0 PYTHON=python3 bash scripts/apply_portfolio_complete_fix_v3.sh
```

## 修复内容

- backtest 加载 samples 后调用 `add_trade_returns(...)`
- backtest 的 OpenLot/TradeRecord 传播 `entry_policy/label_mode/expected_return_col/samples/metadata_path`
- 修复 TradeRecord dataclass 字段顺序
- 修复 rejected_scores reset_index 索引 bug
- adapter 整体替换 `build_inputs()`，输出 `entry_policy/entry_vwap_premium_bps/samples/expected_return_col/metadata_path`
- optimizer Candidate 传播上述字段
- py_compile + import + dataclass order + synthetic adapter test

## 重跑

```bash
START_DATE=2026-01-05 END_DATE=2026-05-15 INITIAL_CASH=200000 \
PYTHON=python3 bash scripts/backtest_historical_score_portfolio.sh
```
