#!/usr/bin/env python3
"""Run AS1455 close-auction v7 max-position grid.

This runner deliberately keeps the model/prediction file fixed and varies only:
- max_positions
- sell_rank
- rebalance_every / rebalance_offset

It invokes run_as1455_close_auction_backtest_v7_maxpos_grid.py once per
configuration, writes per-run logs, and aggregates summary.json files into
leaderboards.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MAX_POSITIONS = [5, 10, 15, 20, 25]
DEFAULT_SELL_RANKS = [75, 100, 150, 200, 250, 300]
DEFAULT_REBALANCE_EVERY = [1, 2, 3, 4, 5]
SMOKE_CONFIGS = [
    (25, 75, 1, 0),
    (10, 150, 3, 0),
    (5, 300, 5, 0),
    (25, 300, 5, 0),
]


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(','):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError(f"empty integer list: {value!r}")
    return out


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def run_name(max_positions: int, sell_rank: int, rebalance_every: int, rebalance_offset: int) -> str:
    return f"max{max_positions:02d}_sell{sell_rank:03d}_reb{rebalance_every}_off{rebalance_offset}"


def build_configs(args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    if args.smoke:
        return SMOKE_CONFIGS
    configs: list[tuple[int, int, int, int]] = []
    for max_pos in args.max_positions_list:
        for sell_rank in args.sell_rank_list:
            for reb_every in args.rebalance_every_list:
                offsets = range(reb_every) if args.offset_mode == 'full' else [0]
                for off in offsets:
                    configs.append((max_pos, sell_rank, reb_every, off))
    return configs


def write_grid_config(path: Path, configs: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['run_name', 'max_positions', 'sell_rank', 'buy_candidate_rank', 'rebalance_every', 'rebalance_offset'])
        w.writeheader()
        for max_pos, sell_rank, reb_every, off in configs:
            w.writerow({
                'run_name': run_name(max_pos, sell_rank, reb_every, off),
                'max_positions': max_pos,
                'sell_rank': sell_rank,
                'buy_candidate_rank': sell_rank,
                'rebalance_every': reb_every,
                'rebalance_offset': off,
            })


def flatten_summary(run_dir: Path, cfg_tuple: tuple[int, int, int, int], status: str, returncode: int | None = None) -> dict[str, Any]:
    max_pos, sell_rank, reb_every, off = cfg_tuple
    row: dict[str, Any] = {
        'run_name': run_name(max_pos, sell_rank, reb_every, off),
        'run_dir': str(run_dir),
        'status': status,
        'returncode': returncode,
        'max_positions': max_pos,
        'sell_rank': sell_rank,
        'buy_candidate_rank': sell_rank,
        'rebalance_every': reb_every,
        'rebalance_offset': off,
    }
    summary_path = run_dir / 'summary.json'
    if not summary_path.exists():
        return row
    try:
        data = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception as exc:
        row['status'] = f'summary_read_error:{exc}'
        return row
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[k] = v
        elif k == 'rejection_reason_counts' and isinstance(v, dict):
            for rk, rv in v.items():
                row[f'reject_{rk}'] = rv
        # Skip nested notes/top_5_drawdowns in the flat CSV; they remain in summary.json.
    return row


def write_leaderboards(summary_csv: Path, out_root: Path) -> None:
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    summary_dir = out_root / '02_summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    ok = df[df['status'].eq('ok')].copy() if 'status' in df.columns else df.copy()
    if ok.empty:
        return
    leaderboard_specs = [
        ('leaderboard_by_total_return.csv', 'total_return', False),
        ('leaderboard_by_annual_return.csv', 'annual_return', False),
        ('leaderboard_by_sharpe.csv', 'sharpe', False),
        ('leaderboard_by_calmar.csv', 'calmar', False),
        ('leaderboard_by_max_drawdown.csv', 'max_drawdown', True),
        ('leaderboard_by_trade_win_rate.csv', 'trade_win_rate', False),
        ('leaderboard_by_low_turnover.csv', 'avg_turnover', True),
        ('leaderboard_by_fee_efficiency.csv', 'fee_to_initial_cash', True),
    ]
    for filename, col, ascending in leaderboard_specs:
        if col not in ok.columns:
            continue
        tmp = ok.copy()
        tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
        tmp = tmp.dropna(subset=[col]).sort_values(col, ascending=ascending)
        tmp.to_csv(summary_dir / filename, index=False, encoding='utf-8-sig')

    # Helpful compact comparison table.
    key_cols = [
        'run_name', 'max_positions', 'sell_rank', 'rebalance_every', 'rebalance_offset',
        'total_return', 'annual_return', 'sharpe', 'calmar', 'max_drawdown',
        'daily_win_rate', 'monthly_win_rate', 'trade_win_rate', 'round_trip_win_rate',
        'avg_turnover', 'annualized_turnover', 'gross_trade_amount', 'total_fee',
        'fee_to_initial_cash', 'avg_positions', 'n_orders', 'n_rejections', 'run_dir'
    ]
    compact_cols = [c for c in key_cols if c in ok.columns]
    ok[compact_cols].to_csv(summary_dir / 'grid_summary_compact.csv', index=False, encoding='utf-8-sig')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run AS1455 v7 max-position/sell-rank/rebalance grid. Unknown args after known options are forwarded to each backtest run.'
    )
    parser.add_argument('--script', default=None, help='Path to run_as1455_close_auction_backtest_v7_maxpos_grid.py')
    parser.add_argument('--python-bin', default=sys.executable or 'python3')
    parser.add_argument('--out-root', required=True, help='Grid output root')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--raw-daily-cache-dir', required=True)
    parser.add_argument('--raw-5m-cache-dir', default=None)
    parser.add_argument('--last5-panel', default=None)
    parser.add_argument('--universe', default=None)
    parser.add_argument('--st-symbols', default=None)
    parser.add_argument('--st-status', default=None)
    parser.add_argument('--corporate-actions', default=None)
    parser.add_argument('--start-date', default=None)
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--profile', default='close_auction_skip_limit')
    parser.add_argument('--capacity-mode', default='none', choices=['none', 'last5_amount', 'last5_volume', 'last5_both'])
    parser.add_argument('--capacity-missing-policy', default='fail', choices=['fail', 'reject', 'disable'])
    parser.add_argument('--participation-rate', default='0.05')
    parser.add_argument('--initial-cash', default='200000')
    parser.add_argument('--commission-rate', default='0.000085')
    parser.add_argument('--min-commission', default='5')
    parser.add_argument('--stamp-tax-rate', default='0.0005')
    parser.add_argument('--transfer-fee-rate', default='0.00001')
    parser.add_argument('--slippage-bps', default='0')
    parser.add_argument('--max-positions-list', type=parse_int_list, default=DEFAULT_MAX_POSITIONS)
    parser.add_argument('--sell-rank-list', type=parse_int_list, default=DEFAULT_SELL_RANKS)
    parser.add_argument('--rebalance-every-list', type=parse_int_list, default=DEFAULT_REBALANCE_EVERY)
    parser.add_argument('--offset-mode', choices=['zero', 'full'], default='zero')
    parser.add_argument('--smoke', action='store_true', help='Run only 4 smoke configurations')
    parser.add_argument('--force', action='store_true', help='Rerun even if summary.json already exists')
    parser.add_argument('--dry-run', action='store_true', help='Write grid config and print commands without running')
    args, passthrough = parser.parse_known_args()

    script = Path(args.script) if args.script else Path(__file__).with_name('run_as1455_close_auction_backtest_v7_maxpos_grid.py')
    if not script.exists():
        raise SystemExit(f'backtest script not found: {script}')

    out_root = Path(args.out_root)
    runs_root = out_root / '01_runs'
    logs_root = out_root / '04_logs'
    summary_root = out_root / '02_summary'
    for p in [runs_root, logs_root, summary_root]:
        p.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)
    write_grid_config(out_root / '00_grid_config.csv', configs)

    common_args: list[str] = [
        '--predictions', args.predictions,
        '--raw-daily-cache-dir', args.raw_daily_cache_dir,
        '--profile', args.profile,
        '--capacity-mode', args.capacity_mode,
        '--capacity-missing-policy', args.capacity_missing_policy,
        '--participation-rate', args.participation_rate,
        '--initial-cash', args.initial_cash,
        '--commission-rate', args.commission_rate,
        '--min-commission', args.min_commission,
        '--stamp-tax-rate', args.stamp_tax_rate,
        '--transfer-fee-rate', args.transfer_fee_rate,
        '--slippage-bps', args.slippage_bps,
    ]
    optional_pairs = [
        ('--raw-5m-cache-dir', args.raw_5m_cache_dir),
        ('--last5-panel', args.last5_panel),
        ('--universe', args.universe),
        ('--st-symbols', args.st_symbols),
        ('--st-status', args.st_status),
        ('--corporate-actions', args.corporate_actions),
        ('--start-date', args.start_date),
        ('--end-date', args.end_date),
    ]
    for flag, value in optional_pairs:
        if value:
            common_args.extend([flag, str(value)])
    common_args.extend(passthrough)

    rows: list[dict[str, Any]] = []
    for i, cfg_tuple in enumerate(configs, start=1):
        max_pos, sell_rank, reb_every, off = cfg_tuple
        name = run_name(max_pos, sell_rank, reb_every, off)
        run_dir = runs_root / name
        log_path = logs_root / f'{name}.log'
        if (run_dir / 'summary.json').exists() and not args.force:
            print(f'[{i}/{len(configs)}] SKIP existing {name}')
            rows.append(flatten_summary(run_dir, cfg_tuple, 'ok', returncode=0))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin,
            str(script),
            '--out-dir', str(run_dir),
            '--max-positions', str(max_pos),
            '--sell-rank', str(sell_rank),
            '--buy-candidate-rank', str(sell_rank),
            '--rebalance-every', str(reb_every),
            '--rebalance-offset', str(off),
            *common_args,
        ]
        print(f'[{i}/{len(configs)}] RUN {name}')
        if args.dry_run:
            print(' '.join(cmd))
            rows.append(flatten_summary(run_dir, cfg_tuple, 'dry_run', returncode=None))
            continue
        with log_path.open('w', encoding='utf-8') as logf:
            logf.write('[CMD] ' + ' '.join(cmd) + '\n\n')
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
        status = 'ok' if proc.returncode == 0 and (run_dir / 'summary.json').exists() else 'failed'
        rows.append(flatten_summary(run_dir, cfg_tuple, status, returncode=proc.returncode))
        if status != 'ok':
            print(f'    FAILED returncode={proc.returncode}; log={log_path}')
        else:
            print(f'    OK log={log_path}')

    summary_df = pd.DataFrame(rows)
    summary_csv = summary_root / 'grid_summary.csv'
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    # Also place a copy at the root for quick discovery.
    summary_df.to_csv(out_root / 'grid_summary.csv', index=False, encoding='utf-8-sig')
    write_leaderboards(summary_csv, out_root)
    print(f'[OK] grid configs={len(configs)} summary={summary_csv}')


if __name__ == '__main__':
    main()
