#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_premarket_history_update.py

盘前历史数据准备流水线。

用途：
1. 开盘前扫描 saved_models，只更新“已有模型”的标的；
2. 根据 saved model 的 metadata / samples 路径 / pipeline_summary / realtime_context_sources.toml
   自动推断每个标的的 pipeline 参数；
3. 串行执行 run_nextday_pipeline.py 的数据更新与样本/特征构造阶段；
4. 不做模型搜索，不保存模型，不做实时采集，不输出交易信号；
5. 输出 premarket_history_update_report.csv/json，方便检查哪些标的更新成功。

推荐运行：
    python pipelines/run_premarket_history_update.py \
      --models-dir saved_models \
      --saved-data-dir saved_data \
      --context-config configs/realtime_context_sources.toml \
      --end-date today

常用：
    # 先看会跑哪些命令
    python pipelines/run_premarket_history_update.py --dry-run

    # 只更新指定股票
    python pipelines/run_premarket_history_update.py --symbols 601899.SH,600312.SH

    # 继续执行，某个标的失败也不中断后续标的
    python pipelines/run_premarket_history_update.py --keep-going

注意：
- 默认不加 --resume，因为盘前更新的目的就是刷新已有样本；
- 如果你确认只想跳过已有完整输出，可显式加 --resume；
- 该脚本不会在 14:55 前后执行，建议放在盘前/盘后定时任务里。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_iso() -> str:
    return dt.date.today().isoformat()


def yyyymmdd_to_iso(s: str) -> str:
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def log(msg: str) -> None:
    print(f"{now_ts()} {msg}", flush=True)


def resolve_project_root(args_root: Optional[str]) -> Path:
    if args_root:
        return Path(args_root).expanduser().resolve()
    here = Path(__file__).resolve()
    if here.parent.name == "pipelines":
        return here.parent.parent
    return here.parent


def norm_stock_code(x: str) -> str:
    x = str(x).strip()
    if not x:
        return x
    if "." in x:
        code, market = x.split(".", 1)
        return f"{code.zfill(6)}.{market.upper()}"
    code = x.zfill(6)
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def raw_code(stock_code: str) -> str:
    return norm_stock_code(stock_code).split(".")[0]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def split_csv_arg(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


@dataclass
class ArtifactInfo:
    stock_code: str
    artifact_name: str
    artifact_dir: Path
    metadata_path: Path
    metadata: dict[str, Any]
    samples_path: Optional[Path] = None


@dataclass
class SymbolPlan:
    stock_code: str
    raw_code: str
    pipeline_out: Path
    sector_symbol: str
    external: list[str] = field(default_factory=list)
    feature_pipeline: list[str] = field(default_factory=lambda: ["fundamental", "sector"])
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    source_notes: list[str] = field(default_factory=list)

    def only_stages(self) -> list[str]:
        stages = ["update_data", "samples"]
        if "fundamental" in self.feature_pipeline:
            stages.append("fundamental")
        if "sector" in self.feature_pipeline:
            stages.append("sector")
        if self.external:
            stages.append("external")
        return stages


def discover_artifacts(models_dir: Path) -> list[ArtifactInfo]:
    artifacts: list[ArtifactInfo] = []
    if not models_dir.exists():
        return artifacts

    for meta_path in models_dir.glob("*/*/metadata.json"):
        meta = read_json(meta_path)
        stock = meta.get("stock_code") or meta_path.parent.parent.name
        stock = norm_stock_code(stock)
        artifact_name = meta.get("artifact_name") or meta_path.parent.name
        samples_value = meta.get("samples")
        samples_path = Path(samples_value).expanduser() if samples_value else None
        artifacts.append(
            ArtifactInfo(
                stock_code=stock,
                artifact_name=artifact_name,
                artifact_dir=meta_path.parent,
                metadata_path=meta_path,
                metadata=meta,
                samples_path=samples_path,
            )
        )
    return artifacts


def infer_pipeline_out_from_samples(samples_path: Optional[Path], root: Path) -> Optional[Path]:
    if not samples_path:
        return None

    p = samples_path
    if not p.is_absolute():
        p = (root / p).resolve()

    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.endswith("_pipeline_out"):
            return Path(*parts[: i + 1])
    return None


def infer_external_from_samples(samples_path: Optional[Path]) -> list[str]:
    if not samples_path:
        return []
    parts = list(samples_path.parts)
    out: list[str] = []
    for i, part in enumerate(parts):
        if part == "04_external" and i + 1 < len(parts):
            out.append(parts[i + 1])
    # 去重保序
    seen = set()
    ans = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            ans.append(x)
    return ans


def load_pipeline_summary(pipeline_out: Optional[Path]) -> dict[str, Any]:
    if not pipeline_out:
        return {}
    for name in ["pipeline_summary.json", "run_summary.json"]:
        path = pipeline_out / name
        if path.exists():
            return read_json(path)
    return {}


def sector_from_context_config(stock_code: str, config: dict[str, Any]) -> Optional[str]:
    stocks = config.get("stocks", {})
    entry = stocks.get(stock_code) or stocks.get(norm_stock_code(stock_code))
    if not isinstance(entry, dict):
        return None
    sector_symbols = entry.get("sector_symbols") or []
    if isinstance(sector_symbols, list) and sector_symbols:
        return str(sector_symbols[0])
    return None


def external_from_context_config(stock_code: str, config: dict[str, Any]) -> list[str]:
    # 注意：这里的 context_groups 是实时上下文，不一定等同 run_nextday_pipeline 的 --external。
    # 不直接拿它作为 pipeline external，避免把 gold/copper 误当成 zijin_external。
    return []


def external_from_pipeline_summary(summary: dict[str, Any]) -> list[str]:
    v = summary.get("external")
    if isinstance(v, list):
        return [str(x) for x in v if str(x)]
    if isinstance(v, str) and v:
        return split_csv_arg(v)
    return []


def feature_pipeline_from_summary(summary: dict[str, Any]) -> list[str]:
    v = summary.get("feature_pipeline")
    if isinstance(v, list):
        xs = [str(x) for x in v if str(x)]
        return xs or ["fundamental", "sector"]
    if isinstance(v, str) and v:
        return split_csv_arg(v)
    return ["fundamental", "sector"]


def choose_symbol_plan(
    stock_code: str,
    artifacts: list[ArtifactInfo],
    *,
    root: Path,
    saved_data_dir: Path,
    context_config: dict[str, Any],
    default_sector_symbol: Optional[str],
    sector_overrides: dict[str, str],
    external_overrides: dict[str, list[str]],
) -> SymbolPlan:
    stock_code = norm_stock_code(stock_code)
    rcode = raw_code(stock_code)

    # 优先从任一 artifact metadata.samples 反推 pipeline_out
    pipeline_out: Optional[Path] = None
    for art in artifacts:
        pipeline_out = infer_pipeline_out_from_samples(art.samples_path, root)
        if pipeline_out is not None:
            break
    if pipeline_out is None:
        pipeline_out = saved_data_dir / f"{rcode}_pipeline_out"

    summary = load_pipeline_summary(pipeline_out)

    # sector 优先级：命令行覆盖 > pipeline_summary > realtime_context_sources.toml > default_sector_symbol
    sector_symbol = (
        sector_overrides.get(stock_code)
        or sector_overrides.get(rcode)
        or summary.get("sector_symbol")
        or sector_from_context_config(stock_code, context_config)
        or default_sector_symbol
    )
    if not sector_symbol:
        raise ValueError(
            f"Cannot infer sector_symbol for {stock_code}. "
            f"请在 pipeline_summary.json、configs/realtime_context_sources.toml 或 --sector-override 中提供。"
        )

    # external 优先级：命令行覆盖 > pipeline_summary > samples路径反推
    external = external_overrides.get(stock_code) or external_overrides.get(rcode)
    if external is None:
        external = external_from_pipeline_summary(summary)
    if not external:
        seen = set()
        external = []
        for art in artifacts:
            for x in infer_external_from_samples(art.samples_path):
                if x not in seen:
                    seen.add(x)
                    external.append(x)

    feature_pipeline = feature_pipeline_from_summary(summary)

    notes = []
    if summary:
        notes.append("pipeline_summary")
    if sector_from_context_config(stock_code, context_config):
        notes.append("context_config_sector")
    if external:
        notes.append("external=" + ",".join(external))

    return SymbolPlan(
        stock_code=stock_code,
        raw_code=rcode,
        pipeline_out=pipeline_out,
        sector_symbol=str(sector_symbol),
        external=list(external or []),
        feature_pipeline=feature_pipeline,
        artifacts=artifacts,
        source_notes=notes,
    )


def parse_key_value_list(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}, expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[norm_stock_code(k.strip()) if k.strip().isdigit() or "." in k.strip() else k.strip()] = v.strip()
    return out


def parse_external_overrides(values: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid external override: {item}, expected SYMBOL=ext1,ext2")
        k, v = item.split("=", 1)
        key = norm_stock_code(k.strip()) if k.strip().isdigit() or "." in k.strip() else k.strip()
        out[key] = split_csv_arg(v)
    return out


def write_plan_csv(plans: list[SymbolPlan], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stock_code",
                "raw_code",
                "sector_symbol",
                "external",
                "feature_pipeline",
                "only_stages",
                "pipeline_out",
                "n_artifacts",
                "artifact_names",
                "source_notes",
            ],
        )
        writer.writeheader()
        for p in plans:
            writer.writerow(
                {
                    "stock_code": p.stock_code,
                    "raw_code": p.raw_code,
                    "sector_symbol": p.sector_symbol,
                    "external": ",".join(p.external),
                    "feature_pipeline": ",".join(p.feature_pipeline),
                    "only_stages": ",".join(p.only_stages()),
                    "pipeline_out": str(p.pipeline_out),
                    "n_artifacts": len(p.artifacts),
                    "artifact_names": ",".join(a.artifact_name for a in p.artifacts),
                    "source_notes": ",".join(p.source_notes),
                }
            )


def write_report_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stock_code",
        "status",
        "returncode",
        "elapsed_seconds",
        "sector_symbol",
        "external",
        "pipeline_out",
        "command",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def build_command(
    plan: SymbolPlan,
    *,
    python: str,
    script_path: str,
    start_date: Optional[str],
    end_date: str,
    resume: bool,
    cache_mode: Optional[str],
    feature_cache_mode: Optional[str],
) -> list[str]:
    cmd = [
        python,
        script_path,
        "--symbol",
        plan.stock_code,
        "--sector-symbol",
        plan.sector_symbol,
        "--end-date",
        end_date,
        "--feature-pipeline",
        ",".join(plan.feature_pipeline),
        "--only-stages",
        ",".join(plan.only_stages()),
    ]

    if start_date:
        cmd.extend(["--start-date", start_date])

    if plan.external:
        cmd.extend(["--external", ",".join(plan.external)])

    if cache_mode:
        cmd.extend(["--cache-mode", cache_mode])
    if feature_cache_mode:
        cmd.extend(["--feature-cache-mode", feature_cache_mode])

    if resume:
        cmd.append("--resume")

    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description="盘前历史数据更新：只更新 saved_models 中已有模型的标的。")
    ap.add_argument("--project-root", default=None, help="项目根目录，默认自动识别。")
    ap.add_argument("--python", default=None, help="Python 解释器路径，默认当前解释器。")
    ap.add_argument("--models-dir", default="saved_models", help="saved_models 目录。")
    ap.add_argument("--saved-data-dir", default="saved_data", help="saved_data 目录。")
    ap.add_argument("--context-config", default="configs/realtime_context_sources.toml", help="实时上下文配置，用于推断 sector_symbol。")
    ap.add_argument("--symbols", default=None, help="只更新这些股票，逗号分隔，如 601899.SH,600312.SH。")
    ap.add_argument("--exclude-symbols", default=None, help="排除这些股票，逗号分隔。")
    ap.add_argument("--max-symbols", type=int, default=None, help="最多更新前 N 个标的，便于测试。")

    ap.add_argument("--start-date", default=None, help="传给 run_nextday_pipeline.py 的 start-date；不传则沿用 pipeline 默认。")
    ap.add_argument("--end-date", default="today", help="结束日期，默认 today。")
    ap.add_argument("--resume", action="store_true", help="给 run_nextday_pipeline.py 传 --resume；默认不传，保证盘前刷新。")
    ap.add_argument("--cache-mode", default=None, help="可选，传给 run_nextday_pipeline.py。")
    ap.add_argument("--feature-cache-mode", default=None, help="可选，传给 run_nextday_pipeline.py。")

    ap.add_argument("--default-sector-symbol", default=None, help="无法推断 sector_symbol 时使用的默认板块，不建议批量使用。")
    ap.add_argument(
        "--sector-override",
        action="append",
        default=[],
        help="手工指定板块，如 601899.SH=贵金属，可重复。",
    )
    ap.add_argument(
        "--external-override",
        action="append",
        default=[],
        help="手工指定 external，如 601899.SH=zijin_external，可重复。",
    )

    ap.add_argument("--dry-run", action="store_true", help="只生成计划并打印命令，不执行。")
    ap.add_argument("--keep-going", action="store_true", help="某个标的失败后继续更新后续标的。")
    ap.add_argument("--out-dir", default=None, help="报告输出目录，默认 saved_data/premarket_history_update/YYYYMMDD。")

    args = ap.parse_args()

    root = resolve_project_root(args.project_root)
    python = args.python or sys.executable

    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = root / models_dir

    saved_data_dir = Path(args.saved_data_dir)
    if not saved_data_dir.is_absolute():
        saved_data_dir = root / saved_data_dir

    context_config_path = Path(args.context_config)
    if not context_config_path.is_absolute():
        context_config_path = root / context_config_path

    script_path = "pipelines/run_nextday_pipeline.py"
    if not (root / script_path).exists():
        raise FileNotFoundError(f"run_nextday_pipeline.py not found: {root / script_path}")

    end_date = today_iso() if args.end_date == "today" else yyyymmdd_to_iso(args.end_date)
    report_date = dt.date.today().strftime("%Y%m%d")
    out_dir = Path(args.out_dir) if args.out_dir else saved_data_dir / "premarket_history_update" / report_date
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "premarket_history_update.log"

    log(f"[ROOT] {root}")
    log(f"[MODELS] {models_dir}")
    log(f"[END_DATE] {end_date}")
    log(f"[OUT] {out_dir}")

    context_config = read_toml(context_config_path)
    sector_overrides = parse_key_value_list(args.sector_override)
    external_overrides = parse_external_overrides(args.external_override)

    artifacts = discover_artifacts(models_dir)
    if not artifacts:
        raise RuntimeError(f"No saved model artifacts found under {models_dir}")

    by_symbol: dict[str, list[ArtifactInfo]] = {}
    for a in artifacts:
        by_symbol.setdefault(a.stock_code, []).append(a)

    include_symbols = set(norm_stock_code(x) for x in split_csv_arg(args.symbols))
    exclude_symbols = set(norm_stock_code(x) for x in split_csv_arg(args.exclude_symbols))

    symbols = sorted(by_symbol.keys())
    if include_symbols:
        symbols = [s for s in symbols if s in include_symbols]
    if exclude_symbols:
        symbols = [s for s in symbols if s not in exclude_symbols]
    if args.max_symbols is not None:
        symbols = symbols[: args.max_symbols]

    plans: list[SymbolPlan] = []
    skipped: list[dict[str, Any]] = []

    for s in symbols:
        try:
            plan = choose_symbol_plan(
                s,
                by_symbol[s],
                root=root,
                saved_data_dir=saved_data_dir,
                context_config=context_config,
                default_sector_symbol=args.default_sector_symbol,
                sector_overrides=sector_overrides,
                external_overrides=external_overrides,
            )
            plans.append(plan)
        except Exception as exc:
            skipped.append({"stock_code": s, "reason": str(exc)})
            log(f"[SKIP] {s}: {exc}")
            if not args.keep_going:
                raise

    if not plans:
        raise RuntimeError("No symbol plans to run.")

    write_plan_csv(plans, out_dir / "premarket_update_plan.csv")
    if skipped:
        with (out_dir / "skipped_symbols.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["stock_code", "reason"])
            writer.writeheader()
            writer.writerows(skipped)

    log(f"[PLAN] symbols={len(plans)}, skipped={len(skipped)}")
    for p in plans:
        log(
            f"[PLAN] {p.stock_code}: sector={p.sector_symbol}, "
            f"external={','.join(p.external) or '-'}, stages={','.join(p.only_stages())}"
        )

    report_rows: list[dict[str, Any]] = []

    for plan in plans:
        cmd = build_command(
            plan,
            python=python,
            script_path=script_path,
            start_date=args.start_date,
            end_date=end_date,
            resume=args.resume,
            cache_mode=args.cache_mode,
            feature_cache_mode=args.feature_cache_mode,
        )

        log(f"[RUN] {plan.stock_code}: {' '.join(cmd)}")
        start = time.time()
        rc = 0
        err = ""

        if args.dry_run:
            elapsed = 0.0
            status = "dry_run"
        else:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"\n\n===== {now_ts()} {plan.stock_code} CMD: {' '.join(cmd)} =====\n")
                f.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ.copy(),
                )
            rc = proc.returncode
            elapsed = time.time() - start
            status = "ok" if rc == 0 else "failed"
            if rc != 0:
                err = f"returncode={rc}"
                log(f"[ERROR] {plan.stock_code} failed: {err}")
                if not args.keep_going:
                    report_rows.append(
                        {
                            "stock_code": plan.stock_code,
                            "status": status,
                            "returncode": rc,
                            "elapsed_seconds": round(elapsed, 2),
                            "sector_symbol": plan.sector_symbol,
                            "external": ",".join(plan.external),
                            "pipeline_out": str(plan.pipeline_out),
                            "command": " ".join(cmd),
                            "error": err,
                        }
                    )
                    break

        report_rows.append(
            {
                "stock_code": plan.stock_code,
                "status": status,
                "returncode": rc,
                "elapsed_seconds": round(elapsed, 2),
                "sector_symbol": plan.sector_symbol,
                "external": ",".join(plan.external),
                "pipeline_out": str(plan.pipeline_out),
                "command": " ".join(cmd),
                "error": err,
            }
        )

        log(f"[DONE] {plan.stock_code}: status={status}, elapsed={elapsed:.1f}s")

    write_report_csv(report_rows, out_dir / "premarket_history_update_report.csv")
    with (out_dir / "premarket_history_update_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": now_ts(),
                "end_date": end_date,
                "n_plans": len(plans),
                "n_skipped": len(skipped),
                "dry_run": args.dry_run,
                "rows": report_rows,
                "skipped": skipped,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    n_failed = sum(1 for r in report_rows if r.get("status") == "failed")
    log(f"[SUMMARY] ok={len(report_rows)-n_failed}, failed={n_failed}, skipped={len(skipped)}")
    log(f"[REPORT] {out_dir / 'premarket_history_update_report.csv'}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
