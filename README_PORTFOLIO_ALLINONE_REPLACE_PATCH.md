# Portfolio All-in-One Replacement Patch

这版用于修复前一版 all-in-one 补丁因“部分已经修改、字符串块匹配失败”导致的报错：

```text
[ERROR] cannot find block for candidate enabled and multiplier
```

这版不再做脆弱的字符串逐块 patch，而是：

```text
1. 备份现有 portfolio 文件；
2. 用已整理好的完整版本替换：
   - portfolio_decision/portfolio_confirm_from_buy_signals.py
   - portfolio_decision/daily_portfolio_confirm_pyscipopt.py
   - scripts/run_portfolio_confirm_from_signals.sh
3. 保留 configs/portfolio_confirm_config.json 的现有字段，只写入：
   - max_policy_weight = 0.15
   - max_positions = 7
4. 若 configs/portfolio_model_overrides.csv 不存在，则创建模板；已存在则不覆盖。
```

## 包含功能

```text
- 从 configs/realtime_context_sources.toml 补 sector，减少 UNKNOWN。
- 支持 configs/portfolio_model_overrides.csv。
- 支持 RECENT_PERF=/path/to/recent_perf.csv。
- optimizer 支持 enabled / weight_multiplier。
- optimizer 支持 max_weight_override / max_add_weight_override。
- 账户级单票上限 max_policy_weight=15%。
- 最大持仓数 max_positions=7。
```

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_allinone_replace_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_allinone_replace_patch.sh
```

## 验证

```bash
grep -n 'max_policy_weight' portfolio_decision/daily_portfolio_confirm_pyscipopt.py configs/portfolio_confirm_config.json
grep -n 'model-overrides' portfolio_decision/portfolio_confirm_from_buy_signals.py
grep -n 'CONTEXT_CONFIG' scripts/run_portfolio_confirm_from_signals.sh
python3 -m py_compile portfolio_decision/portfolio_confirm_from_buy_signals.py portfolio_decision/daily_portfolio_confirm_pyscipopt.py
bash -n scripts/run_portfolio_confirm_from_signals.sh
```

## 回滚

应用脚本会输出备份目录，例如：

```text
saved_data/patch_backups/portfolio_allinone_replace_YYYYMMDD_HHMMSS/
```

要回滚，把该目录下对应文件复制回工程根目录即可。
