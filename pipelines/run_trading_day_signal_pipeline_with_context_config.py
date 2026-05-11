#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trading_day_signal_pipeline.py

交易日 14:55 实盘信号一键流水线。

核心目标：
1. 开始前扫描 saved_models，只采集有模型的标的；
2. 盘中提前采集股票实时快照和板块/外部上下文；
3. 14:52 左右提前 build-bars 和 context features；
4. 14:55 阶段只做 score-now，快速输出 buy_signals.csv；
5. 全程使用 cutoff_time，避免 14:55 之后的数据进入评分。

推荐运行：
    python pipelines/run_trading_day_signal_pipeline.py \
      --watchlist selected_watchlist.txt \
      --context-config configs/realtime_context_sources.toml \
      --cutoff-time 14:55 \
      --stock-collect-until 14:52 \
      --context-collect-until 14:52 \
      --build-time 14:52 \
      --score-time 14:54

如果你把本文件放在项目根目录，也可以：
    python run_trading_day_signal_pipeline.py ...

输出：
    saved_data/intraday_nextday_signals/YYYYMMDD/
      effective_watchlist.txt
      all_scores.csv
      buy_signals.csv
      rejected_scores.csv
      run_summary.json

    saved_data/realtime_context/YYYYMMDD/
      context_snapshots.csv
      context_features_asof.csv
      context_summary.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_yyyymmdd() -> str:
    return dt.datetime.now().strftime("%Y%m%d")


def parse_hhmm(value: str) -> dt.time:
    try:
        return dt.datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid HH:MM time: {value}") from exc


def seconds_until(hhmm: str) -> float:
    target_time = parse_hhmm(hhmm)
    now = dt.datetime.now()
    target = dt.datetime.combine(now.date(), target_time)
    return (target - now).total_seconds()


def sleep_until(hhmm: str, label: str, *, skip_if_past: bool = True) -> None:
    secs = seconds_until(hhmm)
    if secs <= 0:
        if skip_if_past:
            log(f"[TIME] {label}: {hhmm} already passed, continue immediately.")
            return
        raise RuntimeError(f"{label} time {hhmm} already passed.")
    log(f"[TIME] waiting {secs:.1f}s until {label} at {hhmm} ...")
    time.sleep(secs)


def log(msg: str) -> None:
    print(f"{now_ts()} {msg}", flush=True)


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(str(x) for x in cmd)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    log_file: Optional[Path] = None,
    dry_run: bool = False,
    check: bool = True,
    env: Optional[dict[str, str]] = None,
) -> int:
    log(f"[CMD] {quote_cmd(cmd)}")
    if dry_run:
        return 0

    start = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"\n\n===== {now_ts()} CMD: {quote_cmd(cmd)} =====\n")
            f.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=merged_env,
            )
    else:
        proc = subprocess.run(cmd, cwd=str(cwd), env=merged_env)

    elapsed = time.time() - start
    log(f"[DONE] returncode={proc.returncode}, elapsed={elapsed:.1f}s")

    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with returncode={proc.returncode}: {quote_cmd(cmd)}")
    return proc.returncode


def start_proc(
    cmd: list[str],
    *,
    cwd: Path,
    log_file: Path,
    dry_run: bool = False,
    env: Optional[dict[str, str]] = None,
) -> Optional[subprocess.Popen]:
    log(f"[START] {quote_cmd(cmd)}")
    if dry_run:
        return None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = log_file.open("a", encoding="utf-8")
    f.write(f"\n\n===== {now_ts()} START: {quote_cmd(cmd)} =====\n")
    f.flush()

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # start_new_session=True 方便必要时杀掉整个子进程组
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
        start_new_session=True,
    )


def wait_proc(proc: Optional[subprocess.Popen], name: str, *, dry_run: bool = False) -> None:
    if dry_run or proc is None:
        return
    start = time.time()
    log(f"[WAIT] {name} ...")
    rc = proc.wait()
    elapsed = time.time() - start
    log(f"[DONE] {name} returncode={rc}, wait_elapsed={elapsed:.1f}s")
    if rc != 0:
        raise RuntimeError(f"{name} failed with returncode={rc}")


def terminate_proc(proc: Optional[subprocess.Popen], name: str) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    log(f"[TERM] terminating {name} ...")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def resolve_project_root(args_root: Optional[str]) -> Path:
    if args_root:
        return Path(args_root).expanduser().resolve()
    here = Path(__file__).resolve()
    # 如果脚本在 pipelines/ 下，项目根目录是上一层
    if here.parent.name == "pipelines":
        return here.parent.parent
    return here.parent


def py_path(args_python: Optional[str]) -> str:
    return args_python or sys.executable


def ensure_exists(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="交易日从实时数据采集到 14:55 输出买入信号的一键流水线。"
    )

    ap.add_argument("--project-root", default=None, help="项目根目录；默认自动识别。")
    ap.add_argument("--python", default=None, help="Python 解释器路径；默认使用当前解释器。")
    ap.add_argument("--watchlist", default="selected_watchlist.txt", help="关注标的列表。")
    ap.add_argument("--models-dir", default="saved_models", help="saved_models 目录。")
    ap.add_argument("--saved-data-dir", default="saved_data", help="saved_data 根目录。")
    ap.add_argument("--realtime-cache-dir", default="saved_data/akshare_realtime_cache", help="AKShare 实时缓存目录。")
    ap.add_argument("--context-dir", default="saved_data/realtime_context", help="实时上下文目录。")
    ap.add_argument(
        "--context-config",
        default="configs/realtime_context_sources.toml",
        help="实时板块/外部上下文配置文件，传给 collect_realtime_context.py。",
    )
    ap.add_argument("--date", default=None, help="交易日，YYYYMMDD；默认今天。")

    ap.add_argument("--cutoff-time", default="14:55", help="评分数据硬截止时间，HH:MM。")
    ap.add_argument("--stock-collect-until", default="14:52", help="股票实时采集停止时间，HH:MM。")
    ap.add_argument("--context-collect-until", default="14:52", help="板块/外部上下文采集停止时间，HH:MM。")
    ap.add_argument("--build-time", default="14:52", help="开始 build-bars/context-features 的时间，HH:MM。")
    ap.add_argument("--score-time", default="14:54", help="开始 score-now 的时间，HH:MM。")

    ap.add_argument("--stock-interval-seconds", type=int, default=30, help="股票快照采集间隔。")
    ap.add_argument("--context-interval-seconds", type=int, default=60, help="上下文采集间隔。")
    ap.add_argument("--spot-source-priority", default="sina,ths,em,xq", help="股票快照数据源优先级。")
    ap.add_argument("--required-fields", default="close,open,high,low,volume,amount", help="核心字段。")
    ap.add_argument("--xq-max-symbols-per-round", type=int, default=10, help="每轮最多用 xq 补洞的标的数。")

    ap.add_argument("--max-missing-features", type=int, default=5, help="模型特征最多允许填充数量。")
    ap.add_argument("--min-amount-yuan", type=float, default=50_000_000, help="最低成交额过滤。")
    ap.add_argument("--max-abs-pct-chg", type=float, default=None, help="最大绝对涨跌幅过滤；不传则不启用。")

    ap.add_argument("--skip-stock-collect", action="store_true", help="跳过股票实时采集。")
    ap.add_argument("--skip-context-collect", action="store_true", help="跳过上下文实时采集。")
    ap.add_argument("--skip-build-bars", action="store_true", help="跳过股票 build-bars。")
    ap.add_argument("--skip-context-build", action="store_true", help="跳过上下文 build-features。")
    ap.add_argument("--skip-plan", action="store_true", help="跳过 plan 阶段。")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行。")
    ap.add_argument("--keep-going", action="store_true", help="部分非关键步骤失败时继续；score 仍会执行。")

    args = ap.parse_args()

    root = resolve_project_root(args.project_root)
    python = py_path(args.python)
    trade_date = args.date or today_yyyymmdd()

    watchlist = Path(args.watchlist)
    if not watchlist.is_absolute():
        watchlist = root / watchlist

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = root / models_dir

    saved_data_dir = Path(args.saved_data_dir)
    if not saved_data_dir.is_absolute():
        saved_data_dir = root / saved_data_dir

    realtime_cache_dir = Path(args.realtime_cache_dir)
    if not realtime_cache_dir.is_absolute():
        realtime_cache_dir = root / realtime_cache_dir

    context_dir = Path(args.context_dir)
    if not context_dir.is_absolute():
        context_dir = root / context_dir

    context_config = Path(args.context_config)
    if not context_config.is_absolute():
        context_config = root / context_config

    out_dir = saved_data_dir / "intraday_nextday_signals" / trade_date
    log_file = out_dir / "trading_day_pipeline.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(root / "pipelines" / "run_intraday_nextday_signals.py", "run_intraday_nextday_signals.py")
    ensure_exists(root / "data_collection" / "collect_akshare_l1_cache.py", "collect_akshare_l1_cache.py")
    ensure_exists(root / "data_collection" / "collect_realtime_context.py", "collect_realtime_context.py")
    ensure_exists(watchlist, "watchlist")
    ensure_exists(models_dir, "models_dir")
    ensure_exists(context_config, "context_config")

    log(f"[ROOT] {root}")
    log(f"[DATE] {trade_date}")
    log(f"[CONTEXT_CONFIG] {context_config}")
    log(f"[OUT]  {out_dir}")

    run_summary = {
        "trade_date": trade_date,
        "started_at": now_ts(),
        "cutoff_time": args.cutoff_time,
        "stock_collect_until": args.stock_collect_until,
        "context_collect_until": args.context_collect_until,
        "build_time": args.build_time,
        "score_time": args.score_time,
        "steps": [],
    }

    def mark(step: str, status: str, extra: Optional[dict] = None) -> None:
        item = {"step": step, "status": status, "time": now_ts()}
        if extra:
            item.update(extra)
        run_summary["steps"].append(item)
        with (out_dir / "trading_day_pipeline_summary.json").open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

    try:
        # 1. plan：扫描 saved_models，生成 effective_watchlist 和 context plan
        if not args.skip_plan:
            plan_cmd = [
                python, "pipelines/run_intraday_nextday_signals.py", "plan",
                "--watchlist", str(watchlist),
                "--models-dir", str(models_dir),
                "--cutoff-time", args.cutoff_time,
            ]
            run_cmd(plan_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)
            mark("plan", "ok")
        else:
            mark("plan", "skipped")

        effective_watchlist = out_dir / "effective_watchlist.txt"
        if args.dry_run:
            # dry-run 时 plan 不会真的创建文件，但后续命令仍要展示路径
            pass
        elif not effective_watchlist.exists():
            raise FileNotFoundError(
                f"effective_watchlist not found after plan: {effective_watchlist}. "
                f"请检查 saved_models 是否存在可用模型。"
            )

        # 2. 启动采集：股票和上下文并行采集到 14:52
        stock_proc = None
        context_proc = None

        if not args.skip_stock_collect:
            stock_collect_cmd = [
                python, "data_collection/collect_akshare_l1_cache.py", "collect-loop",
                "--symbols-file", str(effective_watchlist),
                "--out-dir", str(realtime_cache_dir),
                "--interval-seconds", str(args.stock_interval_seconds),
                "--until", args.stock_collect_until,
                "--allow-l1-only",
                "--disable-em-bid-ask",
                "--spot-source-priority", args.spot_source_priority,
                "--enable-source-short-circuit",
                "--required-fields", args.required_fields,
                "--xq-only-missing",
                "--xq-max-symbols-per-round", str(args.xq_max_symbols_per_round),
            ]
            stock_proc = start_proc(stock_collect_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run)
            mark("stock_collect_start", "ok")
        else:
            mark("stock_collect_start", "skipped")

        if not args.skip_context_collect:
            context_collect_cmd = [
                python, "data_collection/collect_realtime_context.py", "collect-loop",
                "--watchlist", str(watchlist),
                "--models-dir", str(models_dir),
                "--config", str(context_config),
                "--date", trade_date,
                "--cutoff-time", args.cutoff_time,
                "--interval-seconds", str(args.context_interval_seconds),
                "--until", args.context_collect_until,
            ]
            context_proc = start_proc(context_collect_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run)
            mark("context_collect_start", "ok")
        else:
            mark("context_collect_start", "skipped")

        # 3. 等到 build-time，并等待采集进程自然结束
        sleep_until(args.build_time, "build-time")
        if stock_proc is not None:
            wait_proc(stock_proc, "stock_collect", dry_run=args.dry_run)
            mark("stock_collect_wait", "ok")
        if context_proc is not None:
            wait_proc(context_proc, "context_collect", dry_run=args.dry_run)
            mark("context_collect_wait", "ok")

        # 4. build-bars：只用 cutoff 前数据
        if not args.skip_build_bars:
            build_cmd = [
                python, "data_collection/collect_akshare_l1_cache.py", "build-bars",
                "--out-dir", str(realtime_cache_dir),
                "--date", trade_date,
                "--cutoff-time", args.stock_collect_until,
                "--symbols-file", str(effective_watchlist),
            ]
            try:
                run_cmd(build_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)
                mark("build_bars", "ok")
            except Exception as exc:
                mark("build_bars", "failed", {"error": str(exc)})
                if not args.keep_going:
                    raise
        else:
            mark("build_bars", "skipped")

        # 5. build context features：只用 cutoff 前数据
        if not args.skip_context_build:
            context_build_cmd = [
                python, "data_collection/collect_realtime_context.py", "build-features",
                "--watchlist", str(watchlist),
                "--models-dir", str(models_dir),
                "--config", str(context_config),
                "--date", trade_date,
                "--cutoff-time", args.cutoff_time,
            ]
            try:
                run_cmd(context_build_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)
                mark("context_build_features", "ok")
            except Exception as exc:
                mark("context_build_features", "failed", {"error": str(exc)})
                if not args.keep_going:
                    raise
        else:
            mark("context_build_features", "skipped")

        # 6. 等到 score-time，只 score，不再采集、不再 build
        sleep_until(args.score_time, "score-time")

        score_cmd = [
            python, "pipelines/run_intraday_nextday_signals.py", "score-now",
            "--watchlist", str(watchlist),
            "--models-dir", str(models_dir),
            "--cutoff-time", args.cutoff_time,
            "--context-dir", str(context_dir),
            "--max-missing-features", str(args.max_missing_features),
            "--min-amount-yuan", str(args.min_amount_yuan),
        ]
        if args.max_abs_pct_chg is not None:
            score_cmd.extend(["--max-abs-pct-chg", str(args.max_abs_pct_chg)])

        run_cmd(score_cmd, cwd=root, log_file=log_file, dry_run=args.dry_run, check=True)
        mark("score_now", "ok")

        buy_path = out_dir / "buy_signals.csv"
        all_path = out_dir / "all_scores.csv"
        log(f"[RESULT] all_scores:  {all_path}")
        log(f"[RESULT] buy_signals: {buy_path}")
        mark("pipeline", "ok", {"buy_signals": str(buy_path), "all_scores": str(all_path)})
        return 0

    except KeyboardInterrupt:
        log("[INTERRUPT] received KeyboardInterrupt")
        terminate_proc(stock_proc, "stock_collect")
        terminate_proc(context_proc, "context_collect")
        mark("pipeline", "interrupted")
        return 130

    except Exception as exc:
        log(f"[ERROR] {exc}")
        terminate_proc(stock_proc, "stock_collect")
        terminate_proc(context_proc, "context_collect")
        mark("pipeline", "failed", {"error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
