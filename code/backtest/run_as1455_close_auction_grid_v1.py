#!/usr/bin/env python3
"""Run AS1455 close-auction v7 max-position grid with model/signal dimension.

Default full grid:
- signals: model_0..model_4, ensemble_first3_mean, ensemble_all5_mean
- max_positions: 5, 10, 15, 20, 25
- sell_rank: 75, 100, 150, 200, 250, 300
- rebalance_every: 1, 2, 3, 4, 5

That is 7 * 5 * 6 * 5 = 1050 runs with offset_mode=zero.
With offset_mode=full, it is 7 * 5 * 6 * (1+2+3+4+5) = 3150 runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MAX_POSITIONS = [5, 10, 15, 20, 25]
DEFAULT_SELL_RANKS = [75, 100, 150, 200, 250, 300]
DEFAULT_REBALANCE_EVERY = [1, 2, 3, 4, 5]
DEFAULT_SIGNAL_SPECS = [
    "model_0:0:single",
    "model_1:1:single",
    "model_2:2:single",
    "model_3:3:single",
    "model_4:4:single",
    "ensemble_first3_mean:0,1,2:mean",
    "ensemble_all5_mean:0,1,2,3,4:mean",
]
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


def parse_signal_spec(value: str) -> dict[str, str]:
    """Parse name:cols[:mode], e.g. model_0:0 or ensemble_all5_mean:0,1,2,3,4:mean."""
    parts = str(value).strip().split(':')
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(f"signal spec must be name:cols[:mode], got {value!r}")
    name = parts[0].strip()
    cols = parts[1].strip()
    mode = parts[2].strip().lower() if len(parts) == 3 else None
    if not name:
        raise argparse.ArgumentTypeError(f"signal spec has empty name: {value!r}")
    if not cols:
        raise argparse.ArgumentTypeError(f"signal spec has empty cols: {value!r}")
    n_cols = len([x for x in cols.split(',') if x.strip()])
    if mode is None or not mode:
        mode = 'single' if n_cols == 1 else 'mean'
    if mode not in {'single', 'mean', 'median'}:
        raise argparse.ArgumentTypeError(f"signal mode must be single/mean/median, got {mode!r}")
    if mode == 'single' and n_cols != 1:
        raise argparse.ArgumentTypeError(f"signal mode single requires one column: {value!r}")
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_')
    if safe_name != name:
        raise argparse.ArgumentTypeError(f"signal name must be path-safe [A-Za-z0-9_.-], got {name!r}")
    return {'signal_name': name, 'signal_cols': cols, 'signal_mode': mode}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def run_name(signal_name: str, max_positions: int, sell_rank: int, rebalance_every: int, rebalance_offset: int) -> str:
    return f"{signal_name}_max{max_positions:02d}_sell{sell_rank:03d}_reb{rebalance_every}_off{rebalance_offset}"


def build_configs(args: argparse.Namespace) -> list[tuple[dict[str, str], int, int, int, int]]:
    signal_specs = args.signal_specs
    param_configs = SMOKE_CONFIGS if args.smoke else []
    if not args.smoke:
        for max_pos in args.max_positions_list:
            for sell_rank in args.sell_rank_list:
                for reb_every in args.rebalance_every_list:
                    offsets = range(reb_every) if args.offset_mode == 'full' else [0]
                    for off in offsets:
                        param_configs.append((max_pos, sell_rank, reb_every, off))
    configs: list[tuple[dict[str, str], int, int, int, int]] = []
    for spec in signal_specs:
        for max_pos, sell_rank, reb_every, off in param_configs:
            configs.append((spec, max_pos, sell_rank, reb_every, off))
    return configs


def write_grid_config(path: Path, configs: list[tuple[dict[str, str], int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'run_name', 'signal_name', 'signal_cols', 'signal_mode',
        'max_positions', 'sell_rank', 'buy_candidate_rank', 'rebalance_every', 'rebalance_offset'
    ]
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for spec, max_pos, sell_rank, reb_every, off in configs:
            w.writerow({
                'run_name': run_name(spec['signal_name'], max_pos, sell_rank, reb_every, off),
                'signal_name': spec['signal_name'],
                'signal_cols': spec['signal_cols'],
                'signal_mode': spec['signal_mode'],
                'max_positions': max_pos,
                'sell_rank': sell_rank,
                'buy_candidate_rank': sell_rank,
                'rebalance_every': reb_every,
                'rebalance_offset': off,
            })


def flatten_summary(run_dir: Path, cfg_tuple: tuple[dict[str, str], int, int, int, int], status: str, returncode: int | None = None) -> dict[str, Any]:
    spec, max_pos, sell_rank, reb_every, off = cfg_tuple
    row: dict[str, Any] = {
        'run_name': run_name(spec['signal_name'], max_pos, sell_rank, reb_every, off),
        'run_dir': str(run_dir),
        'status': status,
        'returncode': returncode,
        'signal_name': spec['signal_name'],
        'signal_cols': spec['signal_cols'],
        'signal_mode': spec['signal_mode'],
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

    # Per-signal best rows by Sharpe/return to quickly compare model columns and ensembles.
    if 'signal_name' in ok.columns:
        for metric, ascending in [('sharpe', False), ('total_return', False), ('calmar', False), ('max_drawdown', True)]:
            if metric not in ok.columns:
                continue
            tmp = ok.copy()
            tmp[metric] = pd.to_numeric(tmp[metric], errors='coerce')
            tmp = tmp.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
            per_signal = tmp.groupby('signal_name', as_index=False).head(1)
            per_signal.to_csv(summary_dir / f'best_by_signal_{metric}.csv', index=False, encoding='utf-8-sig')

    key_cols = [
        'run_name', 'status', 'signal_name', 'signal_cols', 'signal_mode',
        'max_positions', 'sell_rank', 'rebalance_every', 'rebalance_offset',
        'model_family', 'model_run', 'prediction_file_sha256',
        'total_return', 'annual_return', 'sharpe', 'calmar', 'max_drawdown',
        'daily_win_rate', 'monthly_win_rate', 'trade_win_rate', 'round_trip_win_rate',
        'avg_turnover', 'annualized_turnover', 'gross_trade_amount', 'total_fee',
        'fee_to_initial_cash', 'avg_positions', 'n_orders', 'n_rejections', 'run_dir'
    ]
    compact_cols = [c for c in key_cols if c in ok.columns]
    ok[compact_cols].to_csv(summary_dir / 'grid_summary_compact.csv', index=False, encoding='utf-8-sig')


def infer_model_run(predictions: str) -> str | None:
    p = Path(predictions)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.startswith('ch17_as1455_train'):
            return part
    return None


def infer_model_params_file(predictions: str) -> str | None:
    p = Path(predictions)
    candidates = [
        p.parent / 'best_params.csv',
        p.parent / 'cv_results.csv',
        p.parent.parent / 'best_params.csv',
        p.parent.parent / 'cv_results.csv',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run AS1455 v7 model/signal + max-position/sell-rank/rebalance grid. Unknown args are forwarded to each backtest run.'
    )
    parser.add_argument('--script', default=None, help='Path to run_as1455_close_auction_backtest_v7_maxpos_grid.py')
    parser.add_argument('--python-bin', default=sys.executable or 'python3')
    parser.add_argument('--out-root', required=True, help='Grid output root')
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--prediction-key', default=None)
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
    parser.add_argument('--signal-spec', dest='signal_specs', action='append', type=parse_signal_spec, default=None,
                        help='Signal spec name:cols[:mode]. Repeatable. Default runs model_0..model_4 plus first3/all5 mean ensembles.')
    parser.add_argument('--offset-mode', choices=['zero', 'full'], default='zero')
    parser.add_argument('--run-output-mode', choices=['summary', 'compact', 'full'], default='compact',
                        help='Per-run file retention passed to single backtest. compact avoids large orders/positions/round_trip CSVs.')
    parser.add_argument('--smoke', action='store_true', help='Run 4 parameter configs for every selected signal; default = 28 runs')
    parser.add_argument('--force', action='store_true', help='Rerun even if summary.json already exists')
    parser.add_argument('--dry-run', action='store_true', help='Write grid config and print commands without running')
    parser.add_argument('--model-family', default='ML4T Ch17 NN')
    parser.add_argument('--model-run', default=None)
    parser.add_argument('--model-params-file', default=None)
    parser.add_argument('--prediction-file-sha256', default=None)
    args, passthrough = parser.parse_known_args()

    if args.signal_specs is None:
        args.signal_specs = [parse_signal_spec(x) for x in DEFAULT_SIGNAL_SPECS]

    script = Path(args.script) if args.script else Path(__file__).with_name('run_as1455_close_auction_backtest_v7_maxpos_grid.py')
    if not script.exists():
        raise SystemExit(f'backtest script not found: {script}')
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f'prediction file not found: {pred_path}')

    prediction_sha = args.prediction_file_sha256 or sha256_file(pred_path)
    model_run = args.model_run or infer_model_run(args.predictions)
    model_params_file = args.model_params_file or infer_model_params_file(args.predictions)

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
        '--prediction-file-sha256', prediction_sha,
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
        '--model-family', args.model_family,
        '--output-mode', args.run_output_mode,
    ]
    if model_run:
        common_args.extend(['--model-run', model_run])
    if model_params_file:
        common_args.extend(['--model-params-file', model_params_file])
    optional_pairs = [
        ('--prediction-key', args.prediction_key),
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
        spec, max_pos, sell_rank, reb_every, off = cfg_tuple
        name = run_name(spec['signal_name'], max_pos, sell_rank, reb_every, off)
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
            '--signal-name', spec['signal_name'],
            '--signal-cols', spec['signal_cols'],
            '--signal-mode', spec['signal_mode'],
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
    summary_df.to_csv(out_root / 'grid_summary.csv', index=False, encoding='utf-8-sig')
    write_leaderboards(summary_csv, out_root)
    print(f'[OK] grid configs={len(configs)} summary={summary_csv}')
    print(f'[INFO] prediction_sha256={prediction_sha}')
    print(f'[INFO] model_run={model_run}')
    print(f'[INFO] model_params_file={model_params_file}')


if __name__ == '__main__':
    main()
