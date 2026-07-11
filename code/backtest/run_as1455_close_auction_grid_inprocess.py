#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AS1455 in-process close-auction grid with shared rankings.

Compared with run_as1455_close_auction_grid_v1.py, this runner:
- loads every signal once;
- builds the execution panel once;
- ranks every signal once per date;
- reuses those caches for all max_positions/sell_rank/offset combinations;
- skips non-rebalance-day rank-map access unless full position output is requested.

The actual portfolio logic is compiled from the existing v7 backtest function at
runtime, so execution/fee/limit/corporate-action semantics stay aligned.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


legacy = load_module("as1455_grid_legacy", HERE / "run_as1455_close_auction_grid_v1.py")
bt = load_module("as1455_bt_v7", HERE / "run_as1455_close_auction_backtest_v7_maxpos_grid.py")


def compile_prepared_backtest():
    """Patch only the data-preparation part of the existing backtest function."""
    src = inspect.getsource(bt.backtest)

    old_head = '''def backtest(
    preds: pd.DataFrame,
    exec_panel: pd.DataFrame,
    cfg: TradeConfig,
    corporate_actions: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | dict]:
    if preds.empty:
        raise ValueError("empty predictions")
    if exec_panel.empty:
        raise ValueError("empty execution panel")

    pred_dates = pd.DatetimeIndex(preds["date"].unique()).sort_values()
    exec_dates = pd.DatetimeIndex(exec_panel["date"].unique()).sort_values()
    exec_date_set = set(exec_dates)
    dates = [d for d in pred_dates if d in exec_date_set]
    if len(dates) < 2:
        raise ValueError(f"not enough overlapping dates: pred={len(pred_dates)} exec={len(exec_dates)} overlap={len(dates)}")

    exec_by_date = {d: g.set_index("symbol", drop=False) for d, g in exec_panel.groupby("date", sort=True)}
    preds_by_date = {d: g.copy() for d, g in preds.groupby("date", sort=True)}
    corporate_actions = corporate_actions if corporate_actions is not None else pd.DataFrame()
'''
    new_head = '''def backtest_prepared(
    prepared: dict,
    exec_by_date: dict,
    cfg: TradeConfig,
    corporate_actions: pd.DataFrame | None = None,
    collect_position_details: bool = False,
) -> dict[str, pd.DataFrame | dict]:
    dates = prepared["dates"]
    ranked_by_date = prepared["ranked_by_date"]
    rank_map_by_date = prepared["rank_map_by_date"]
    score_map_by_date = prepared["score_map_by_date"]
    corporate_actions = corporate_actions if corporate_actions is not None else pd.DataFrame()
'''
    if old_head not in src:
        raise RuntimeError("v7 backtest header changed; prepared-grid patch refused")
    src = src.replace(old_head, new_head, 1)

    old_loop = '''    for day_index, date in enumerate(dates):
        exec_t = exec_by_date[date]
        pred_t = preds_by_date[date].sort_values("score", ascending=False).copy()
        pred_t["rank"] = np.arange(1, len(pred_t) + 1)
        rank_map = pred_t.set_index("symbol")["rank"].to_dict()
        score_map = pred_t.set_index("symbol")["score"].to_dict()
        is_reb = is_rebalance_day_index(day_index, cfg)
'''
    new_loop = '''    for day_index, date in enumerate(dates):
        exec_t = exec_by_date[date]
        is_reb = is_rebalance_day_index(day_index, cfg)
        if is_reb or collect_position_details:
            pred_t = ranked_by_date[date]
            rank_map = rank_map_by_date[date]
            score_map = score_map_by_date[date]
        else:
            pred_t = None
            rank_map = {}
            score_map = {}
'''
    if old_loop not in src:
        raise RuntimeError("v7 daily loop changed; prepared-grid patch refused")
    src = src.replace(old_loop, new_loop, 1)

    pos_start = src.find('        for sym, value in sorted(holding_values.items()):')
    pos_end = src.find('        last_nav = nav', pos_start)
    if pos_start < 0 or pos_end < 0:
        raise RuntimeError("v7 position-detail block changed; prepared-grid patch refused")
    block = src[pos_start:pos_end]
    src = (
        src[:pos_start]
        + '        if collect_position_details:\n'
        + textwrap.indent(block, '    ')
        + src[pos_end:]
    )

    ns: dict[str, Any] = {}
    exec(src, bt.__dict__, ns)
    return ns["backtest_prepared"]


backtest_prepared = compile_prepared_backtest()


def prepare_signal(preds: pd.DataFrame, exec_panel: pd.DataFrame) -> dict[str, Any]:
    """Build rankings once, matching the original per-date sort operation."""
    pred_dates = pd.DatetimeIndex(preds["date"].unique()).sort_values()
    exec_dates = pd.DatetimeIndex(exec_panel["date"].unique()).sort_values()
    exec_date_set = set(exec_dates)
    dates = [d for d in pred_dates if d in exec_date_set]
    if len(dates) < 2:
        raise ValueError(
            f"not enough overlapping dates: pred={len(pred_dates)} "
            f"exec={len(exec_dates)} overlap={len(dates)}"
        )

    ranked_by_date = {}
    rank_map_by_date = {}
    score_map_by_date = {}
    for date, g in preds[preds["date"].isin(dates)].groupby("date", sort=True):
        ranked = g.sort_values("score", ascending=False).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked_by_date[date] = ranked
        rank_map_by_date[date] = ranked.set_index("symbol")["rank"].to_dict()
        score_map_by_date[date] = ranked.set_index("symbol")["score"].to_dict()

    return {
        "dates": dates,
        "ranked_by_date": ranked_by_date,
        "rank_map_by_date": rank_map_by_date,
        "score_map_by_date": score_map_by_date,
    }


def output_maps(result: dict[str, Any], mode: str):
    full = {
        "close_auction_nav.csv": result["nav"],
        "close_auction_orders.csv": result["orders"],
        "close_auction_trades.csv": result["trades"],
        "close_auction_rejections.csv": result["rejections"],
        "close_auction_positions.csv": result["positions"],
        "close_auction_corporate_actions.csv": result["corporate_actions"],
        "daily_drawdown.csv": result["daily_drawdown"],
        "round_trips.csv": result["round_trips"],
        "monthly_summary.csv": result["monthly_summary"],
        "yearly_summary.csv": result["yearly_summary"],
        "fee_summary.csv": result["fee_summary"],
        "turnover_summary.csv": result["turnover_summary"],
    }
    compact = {
        "close_auction_nav.csv": result["nav"],
        "daily_drawdown.csv": result["daily_drawdown"],
        "monthly_summary.csv": result["monthly_summary"],
        "yearly_summary.csv": result["yearly_summary"],
        "fee_summary.csv": result["fee_summary"],
        "turnover_summary.csv": result["turnover_summary"],
    }
    return full if mode == "full" else compact if mode == "compact" else {}, full


def write_run(
    run_dir: Path,
    result: dict[str, Any],
    cfg,
    signal_meta: dict[str, Any],
    args: argparse.Namespace,
    prediction_sha: str,
    exec_panel: pd.DataFrame,
    capacity_precheck: dict[str, Any],
    model_run: str | None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    selected, full = output_maps(result, args.run_output_mode)
    for name, df in selected.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(run_dir / name, index=False, encoding="utf-8-sig")

    model_meta = {
        "model_family": str(args.model_family),
        "model_run": str(model_run) if model_run else None,
    }
    summary = dict(result["summary"])
    summary.update(model_meta)
    summary.update({k: v for k, v in signal_meta.items() if not isinstance(v, (list, dict))})

    config = dict(cfg.__dict__)
    config.update(model_meta)
    config.update(signal_meta)
    config["output_mode"] = args.run_output_mode
    config["grid_engine"] = "inprocess_shared_rank_v1"

    run_meta = {
        "predictions": str(args.predictions),
        "prediction_file_sha256": prediction_sha,
        "model_meta": model_meta,
        "signal_meta": signal_meta,
        "capacity_precheck": capacity_precheck,
        "n_execution_rows": int(len(exec_panel)),
        "n_execution_symbols": int(exec_panel["symbol"].nunique()),
        "n_execution_dates": int(exec_panel["date"].nunique()),
        "config": cfg.__dict__,
        "summary": summary,
        "output_mode": args.run_output_mode,
        "output_files": sorted(selected),
        "suppressed_output_files": sorted(set(full) - set(selected)),
        "grid_engine": "inprocess_shared_rank_v1",
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )
    (run_dir / "close_auction_summary.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2, default=bt.json_default),
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser(description="AS1455 shared-ranking in-process grid")
    p.add_argument("--out-root", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--prediction-key", default=None)
    p.add_argument("--raw-daily-cache-dir", required=True)
    p.add_argument("--raw-5m-cache-dir", default=None)
    p.add_argument("--last5-panel", default=None)
    p.add_argument("--universe", default=None)
    p.add_argument("--st-symbols", default=None)
    p.add_argument("--st-status", default=None)
    p.add_argument("--corporate-actions", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--profile", default="close_auction_skip_limit",
                   choices=["close_auction_simple", "close_auction_skip_limit"])
    p.add_argument("--capacity-mode", default="none",
                   choices=["none", "last5_amount", "last5_volume", "last5_both"])
    p.add_argument("--capacity-missing-policy", default="fail",
                   choices=["fail", "reject", "disable"])
    p.add_argument("--min-last5-coverage", type=float, default=0.95)
    p.add_argument("--participation-rate", type=float, default=0.05)
    p.add_argument("--initial-cash", type=float, default=200000)
    p.add_argument("--commission-rate", type=float, default=0.000085)
    p.add_argument("--min-commission", type=float, default=5)
    p.add_argument("--stamp-tax-rate", type=float, default=0.0005)
    p.add_argument("--transfer-fee-rate", type=float, default=0.00001)
    p.add_argument("--slippage-bps", type=float, default=0)
    p.add_argument("--lot-size", type=int, default=100)
    p.add_argument("--allow-non-mainboard", action="store_true")
    p.add_argument("--allow-st", action="store_true")
    p.add_argument("--corporate-action-mode", default="synthetic_share_factor_from_preclose",
                   choices=["none", "synthetic_share_factor_from_preclose",
                            "synthetic_cash_from_preclose"])
    p.add_argument("--corporate-action-threshold", type=float, default=1e-3)
    p.add_argument("--min-price", type=float, default=0)
    p.add_argument("--limit-eps", type=float, default=1e-6)
    p.add_argument("--max-positions-list", type=legacy.parse_int_list,
                   default=legacy.DEFAULT_MAX_POSITIONS)
    p.add_argument("--sell-rank-list", type=legacy.parse_int_list,
                   default=legacy.DEFAULT_SELL_RANKS)
    p.add_argument("--rebalance-every-list", type=legacy.parse_int_list,
                   default=legacy.DEFAULT_REBALANCE_EVERY)
    p.add_argument("--signal-spec", dest="signal_specs", action="append",
                   type=legacy.parse_signal_spec, default=None)
    p.add_argument("--offset-mode", choices=["zero", "full"], default="zero")
    p.add_argument("--run-output-mode", choices=["summary", "compact", "full"],
                   default="compact")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--model-family", default="ML4T Ch17 NN")
    p.add_argument("--model-run", default=None)
    p.add_argument("--model-params-file", default=None)
    p.add_argument("--prediction-file-sha256", default=None)
    args = p.parse_args()
    if args.signal_specs is None:
        args.signal_specs = [legacy.parse_signal_spec(x) for x in legacy.DEFAULT_SIGNAL_SPECS]
    return args


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise SystemExit(f"prediction file not found: {pred_path}")

    out_root = Path(args.out_root)
    runs_root = out_root / "01_runs"
    logs_root = out_root / "04_logs"
    summary_root = out_root / "02_summary"
    for p in [runs_root, logs_root, summary_root]:
        p.mkdir(parents=True, exist_ok=True)

    configs = legacy.build_configs(args)
    legacy.write_grid_config(out_root / "00_grid_config.csv", configs)
    if args.dry_run:
        print(f"[DRY RUN] configs={len(configs)} engine=inprocess_shared_rank_v1")
        return

    prediction_sha = args.prediction_file_sha256 or legacy.sha256_file(pred_path)
    model_run = args.model_run or legacy.infer_model_run(args.predictions)
    model_params_file = args.model_params_file or legacy.infer_model_params_file(args.predictions)

    signals = {}
    all_symbols = set()
    for spec in args.signal_specs:
        preds, meta = bt.load_predictions(
            pred_path,
            args.prediction_key,
            None,
            signal_cols=bt.parse_csv_tokens(spec["signal_cols"]),
            signal_mode=spec["signal_mode"],
            signal_name=spec["signal_name"],
            prediction_file_sha256=prediction_sha,
            model_params_file=Path(model_params_file) if model_params_file else None,
        )
        preds = bt.apply_date_filters(preds, args.start_date, args.end_date)
        if preds.empty:
            raise SystemExit(f"empty predictions for {spec['signal_name']}")
        signals[spec["signal_name"]] = {"preds": preds, "meta": meta}
        all_symbols.update(preds["symbol"].unique())

    universe = bt.read_universe(Path(args.universe) if args.universe else None)
    st_symbols = bt.load_st_symbols(Path(args.st_symbols) if args.st_symbols else None)
    st_status = bt.load_st_status(Path(args.st_status) if args.st_status else None)
    last5_panel = bt.load_last5_panel(Path(args.last5_panel) if args.last5_panel else None)
    corporate_actions = bt.load_corporate_actions(
        Path(args.corporate_actions) if args.corporate_actions else None
    )
    exec_panel, exec_report = bt.build_execution_panel(
        all_symbols,
        Path(args.raw_daily_cache_dir),
        universe,
        st_symbols,
        st_status=st_status,
        last5_panel=last5_panel,
        raw_5m_cache_dir=Path(args.raw_5m_cache_dir) if args.raw_5m_cache_dir else None,
    )
    exec_panel = bt.apply_date_filters(exec_panel, args.start_date, args.end_date)
    if exec_panel.empty:
        raise SystemExit("empty execution panel")
    exec_by_date = {
        d: g.set_index("symbol", drop=False)
        for d, g in exec_panel.groupby("date", sort=True)
    }

    for name, payload in signals.items():
        payload["prepared"] = prepare_signal(payload["preds"], exec_panel)
        effective = args.capacity_mode
        precheck = bt.build_capacity_precheck(exec_panel, payload["preds"], effective)
        if effective != "none":
            coverage = float(precheck.get("coverage_rate", 0))
            positive = float(precheck.get("positive_rate", 0))
            if coverage < args.min_last5_coverage or positive <= 0:
                if args.capacity_missing_policy == "fail":
                    raise SystemExit(
                        f"capacity data insufficient for {name}: "
                        f"coverage={coverage:.6f} positive={positive:.6f}"
                    )
                if args.capacity_missing_policy == "disable":
                    effective = "none"
                    precheck["policy_action"] = "disabled_capacity_mode"
                else:
                    precheck["policy_action"] = "reject_on_missing_capacity"
        payload["capacity_mode"] = effective
        payload["capacity_precheck"] = precheck

    rows = []
    for i, cfg_tuple in enumerate(configs, 1):
        spec, max_pos, sell_rank, reb_every, off = cfg_tuple
        run_name = legacy.run_name(spec["signal_name"], max_pos, sell_rank, reb_every, off)
        run_dir = runs_root / run_name
        log_path = logs_root / f"{run_name}.log"

        if (run_dir / "summary.json").exists() and not args.force:
            print(f"[{i}/{len(configs)}] SKIP existing {run_name}")
            rows.append(legacy.flatten_summary(run_dir, cfg_tuple, "ok", returncode=0))
            continue

        payload = signals[spec["signal_name"]]
        cfg = bt.TradeConfig(
            max_positions=max_pos,
            buy_candidate_rank=sell_rank,
            sell_rank=sell_rank,
            rebalance_every=reb_every,
            rebalance_offset=off,
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            stamp_tax_rate=args.stamp_tax_rate,
            transfer_fee_rate=args.transfer_fee_rate,
            slippage_bps=args.slippage_bps,
            profile=args.profile,
            mainboard_only=not args.allow_non_mainboard,
            min_price=args.min_price,
            limit_eps=args.limit_eps,
            lot_size=args.lot_size,
            min_commission=args.min_commission,
            exclude_st=not args.allow_st,
            capacity_mode=payload["capacity_mode"],
            participation_rate=args.participation_rate,
            corporate_action_mode=args.corporate_action_mode,
            corporate_action_threshold=args.corporate_action_threshold,
        )

        print(f"[{i}/{len(configs)}] RUN {run_name}")
        try:
            result = backtest_prepared(
                payload["prepared"],
                exec_by_date,
                cfg,
                corporate_actions,
                collect_position_details=args.run_output_mode == "full",
            )
            write_run(
                run_dir, result, cfg, payload["meta"], args, prediction_sha,
                exec_panel, payload["capacity_precheck"], model_run,
            )
            log_path.write_text("[OK] inprocess_shared_rank_v1\n", encoding="utf-8")
            rows.append(legacy.flatten_summary(run_dir, cfg_tuple, "ok", returncode=0))
        except Exception as exc:
            log_path.write_text(
                f"[FAILED] {type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            rows.append(legacy.flatten_summary(run_dir, cfg_tuple, "failed", returncode=1))
            print(f"    FAILED {type(exc).__name__}: {exc}")

    summary = pd.DataFrame(rows)
    summary_csv = summary_root / "grid_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(out_root / "grid_summary.csv", index=False, encoding="utf-8-sig")
    legacy.write_leaderboards(summary_csv, out_root)

    manifest = {
        "engine": "inprocess_shared_rank_v1",
        "configs": len(configs),
        "signals": [x["signal_name"] for x in args.signal_specs],
        "execution_panel_built_once": True,
        "daily_rankings_built_once_per_signal": True,
        "non_rebalance_rank_maps_skipped_unless_full_output": True,
        "prediction_file_sha256": prediction_sha,
    }
    (out_root / "grid_engine_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.run_output_mode == "full":
        exec_report.to_csv(
            out_root / "execution_panel_build_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"[OK] configs={len(configs)} summary={summary_csv}")


if __name__ == "__main__":
    main()
