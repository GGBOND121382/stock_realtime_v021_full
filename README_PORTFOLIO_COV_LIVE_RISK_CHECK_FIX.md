# portfolio_cov_live_risk false-positive check fix

你遇到的报错不是代码里还有 nonlinear objective，而是 apply 脚本的 grep 检查把注释里的：

```text
amount[i] * amount[j]
```

也当成了真实表达式。

本补丁做两件事：

```text
1. 把注释中的 amount[i] * amount[j] 改成 amount_i times amount_j；
2. 把检查改成只检查非注释行里的 amount[i] / amount[j]。
```

## 应用

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_cov_live_risk_check_fix_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_cov_live_risk_check_fix_patch.sh
```

如果你刚才已经运行过 `apply_portfolio_cov_live_risk_patch.sh`，虽然最后 check 失败，但核心修改已经写入了。本补丁只修正这个误报检查。
