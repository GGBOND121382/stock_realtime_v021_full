# AS1455 checkpoint signal backtest-like v5

This patch keeps the existing checkpoint inference and rank stages, and replaces the live trade-signal layer with a more backtest-like single-day planner.

It mirrors the backtest where possible:

- rebalance-day gate (`rebalance_every`, `rebalance_offset`);
- sell positions only when `rank > sell_rank`;
- fill empty slots only from `rank <= buy_candidate_rank`;
- do not replace existing holdings just because a new name has a better rank;
- T+1 sell restriction when `buy_date` / `entry_date` exists in `current_positions.csv`;
- `can_buy` / `can_sell` checks for mainboard, ST, tradable, bad price, limit up/down;
- backtest-style sizing: `base_target = min(nav_after_sells / max_positions, cash / slots)`;
- A-share lot rounding, fees, slippage, and optional last-5min capacity mode.

## Install

```bash
cd ~/stock_realtime_v021_full
unzip -oq as1455_checkpoint_signal_backtestlike_v5.zip
bash as1455_checkpoint_signal_backtestlike_v5/install.sh --repo .
```

## Basic run

```bash
SIGNAL_CASH=200000 TRADE_DATE=20260626 \
bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

## Rebalance parity

Exact rebalance parity requires one of:

```bash
# explicit day index, same semantics as backtest day_index
DAY_INDEX=123 SIGNAL_CASH=200000 TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh

# or a calendar CSV with date/trade_date column
REBALANCE_CALENDAR=saved_data/ashare_ml4t/trade_calendar.csv SIGNAL_CASH=200000 TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh

# or intentional override
FORCE_REBALANCE=1 SIGNAL_CASH=200000 TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

Default `CALENDAR_UNKNOWN_POLICY=force` preserves the old behavior of generating a tradable plan when no calendar is provided, but the report marks `calendar_exact=false`. For stricter behavior:

```bash
CALENDAR_UNKNOWN_POLICY=fail TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
CALENDAR_UNKNOWN_POLICY=skip TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

## Position file

For T+1 parity, include `buy_date` or `entry_date`:

```csv
symbol,shares,buy_date
600000.SH,1000,2026-06-25
000001.SZ,1200,2026-06-20
```

If buy date is missing, default `UNKNOWN_BUY_DATE_POLICY=allow` keeps the plan usable and writes `t_plus_1_unknown_buy_date_allowed` in the reason. For strict blocking:

```bash
UNKNOWN_BUY_DATE_POLICY=block SIGNAL_CASH=200000 TRADE_DATE=20260626 bash scripts/run_as1455_live_checkpoint_signal_v1.sh
```

## Backtest-like execution parameters

Defaults match the close-auction v7 grid defaults as far as the single-day signal layer can know:

```bash
REBALANCE_EVERY=3
REBALANCE_OFFSET=0
PROFILE=close_auction_skip_limit
CAPACITY_MODE=none
LOT_SIZE=100
COMMISSION_RATE=0.000085
STAMP_TAX_RATE=0.0005
TRANSFER_FEE_RATE=0.00001
MIN_COMMISSION=5
SLIPPAGE_BPS=0
MAINBOARD_ONLY=1
EXCLUDE_ST=0
```

## Output

`16_live_trade_signal.csv` includes:

- `action`: BUY / SELL / HOLD / WATCH / BUY_BLOCKED / SELL_BLOCKED
- `order_side`: BUY / SELL / empty
- `order_shares`, `order_price`, `order_amount_est`
- fee columns
- `reason`, `is_rebalance_day`, `day_index`, `capacity_reason`

`16_live_trade_signal_report.json` includes parity notes and whether rebalance calendar parity was exact.
