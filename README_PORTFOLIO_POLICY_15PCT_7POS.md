# Portfolio Policy Patch: max_policy_weight=15%, max_positions=7

This patch changes the portfolio policy to:

```text
max_policy_weight = 0.15
max_positions = 7
```

## Rationale

With total assets of 200,000:

```text
15% single-name cap = 30,000 per stock
ceil(100% / 15%) = 7 names minimum if the equity sleeve may become fully invested
```

So `max_positions=7` is the smallest internally consistent cap under a 15% single-name limit. It is also easier to operate than 10 positions.

## Apply

```bash
cd /root/stock_realtime_v021_full
unzip -o /path/to/portfolio_policy_15pct_7pos_patch.zip -d .
PYTHON=python3 bash scripts/apply_portfolio_policy_15pct_7pos_patch.sh
```

Backups are written under:

```text
saved_data/patch_backups/portfolio_policy_15pct_7pos_YYYYMMDD_HHMMSS/
```

## Verify

```bash
grep -n '"max_policy_weight"' configs/portfolio_confirm_config.json
grep -n '"max_positions"' configs/portfolio_confirm_config.json
grep -n 'max_policy_weight' portfolio_decision/daily_portfolio_confirm_pyscipopt.py
```
