# Portfolio complete fix v4

完整修复包，单一入口，应用时自测，失败回滚。

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_complete_fix_v4_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_complete_fix_v4.sh
```

不移动旧回测目录：

```bash
CLEAN_OLD_BACKTEST=0 PYTHON=python3 bash scripts/apply_portfolio_complete_fix_v4.sh
```

自测内容：
- py_compile backtest/adapter/optimizer
- import backtest
- dataclass 字段顺序
- backtest add_trade_returns marker
- adapter `build_inputs()` synthetic unit test
- optimizer 字段传播 marker
