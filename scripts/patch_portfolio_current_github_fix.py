
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKTEST = Path("portfolio_decision/backtest_historical_score_portfolio.py")
ADAPTER = Path("portfolio_decision/portfolio_confirm_from_buy_signals.py")
OPTIMIZER = Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py")


def log(s: str) -> None:
    print(s, flush=True)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def run(cmd: list[str]) -> None:
    log("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def function_bounds(text: str, func_name: str, next_func_name: str) -> tuple[int, int]:
    start_marker = f"\ndef {func_name}("
    next_marker = f"\ndef {next_func_name}("
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"function not found: {func_name}")
    start += 1
    end = text.find(next_marker, start)
    if end < 0:
        raise RuntimeError(f"next function not found after {func_name}: {next_func_name}")
    return start, end + 1


AS_TEXT_BLOCK = '''def as_text(x: Any, default: str = "") -> str:
    try:
        if x is None or pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return default
    return s


'''


def ensure_as_text(text: str) -> str:
    if "\ndef as_text(" in text or text.startswith("def as_text("):
        return text

    for marker in [
        "\ndef parse_rate_decimal(",
        "\ndef parse_bool_flag(",
        "\ndef find_metadata(",
        "\ndef load_json(",
    ]:
        pos = text.find(marker)
        if pos >= 0:
            return text[:pos + 1] + AS_TEXT_BLOCK + text[pos + 1:]

    pos = text.find("\ndef as_float(")
    if pos >= 0:
        next_def = text.find("\ndef ", pos + 1)
        if next_def > pos:
            return text[:next_def + 1] + AS_TEXT_BLOCK + text[next_def + 1:]

    raise RuntimeError("cannot insert as_text(): no stable insertion point found")


BUILD_INPUTS = '''def build_inputs(
    signal_dir: Path,
    saved_models: Path,
    out_input_dir: Path,
    use_all_scores: bool = False,
    context_config: Optional[Path] = None,
    model_overrides: Optional[Path] = None,
    recent_perf: Optional[Path] = None,
) -> Dict[str, Path]:
    src = signal_dir / ("all_scores.csv" if use_all_scores else "buy_signals.csv")
    if not src.exists():
        raise FileNotFoundError(f"missing upstream signal file: {src}")

    raw = pd.read_csv(src)
    out_input_dir.mkdir(parents=True, exist_ok=True)

    context_sector_map = load_context_sector_map(context_config)
    override_rows = load_rule_rows(model_overrides)
    recent_rows = load_rule_rows(recent_perf)

    signals_out = out_input_dir / "portfolio_signals.csv"
    metrics_out = out_input_dir / "portfolio_metrics.csv"
    prices_out = out_input_dir / "portfolio_prices.csv"

    signal_columns = [
        "stock_code", "model_name", "label_mode", "pred_return_bps", "pred_prob",
        "target_hit_bps", "price", "sector", "sector_source", "hit_score",
        "threshold", "score_margin", "entry_policy", "entry_vwap_premium_bps",
        "samples", "expected_return_col", "metadata_path",
    ]
    metric_columns = [
        "stock_code", "model_name", "label_mode", "trades", "win_rate",
        "avg_return_bps", "median_return_bps", "max_drawdown", "profit_factor",
        "target_hit_bps", "feature_group", "base_model_name", "entry_policy",
        "entry_vwap_premium_bps", "samples", "expected_return_col",
        "sector", "sector_source",
    ]

    if raw.empty:
        pd.DataFrame(columns=signal_columns).to_csv(signals_out, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=metric_columns).to_csv(metrics_out, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["stock_code", "price"]).to_csv(prices_out, index=False, encoding="utf-8-sig")
        return {"signals": signals_out, "metrics": metrics_out, "prices": prices_out}

    required = {"stock_code", "artifact_name"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{src} missing required columns: {sorted(missing)}")

    sig_rows, met_rows, price_rows = [], [], []

    for _, r in raw.iterrows():
        stock_code = normalize_stock_code(r.get("stock_code", ""))
        artifact = as_text(r.get("artifact_name"), "")
        if not stock_code or not artifact:
            continue

        meta_path = find_metadata(saved_models, stock_code, artifact)
        meta = load_metadata(meta_path)

        label_mode = as_text(meta.get("label_mode"), as_text(r.get("label_mode"), ""))
        if not label_mode:
            label_mode = "hit" if "hit" in artifact.lower() else "close_profit"

        validation = meta.get("validation_tail_trade_metrics", {}) or meta.get("validation_trade_metrics", {}) or {}
        avg_return_bps = metric_bps_from_return(validation.get("avg_return", np.nan))
        median_return_bps = metric_bps_from_return(validation.get("median_return", np.nan))
        trades = as_float(validation.get("trades", np.nan), np.nan)
        win_rate = as_float(validation.get("win_rate", np.nan), np.nan)
        max_drawdown = as_float(validation.get("max_drawdown", np.nan), np.nan)
        profit_factor = as_float(validation.get("profit_factor", np.nan), np.nan)

        target_hit_bps = as_float(
            meta.get("target_hit_bps", r.get("target_hit_bps", 80 if "80" in artifact else 50)),
            50.0,
        )

        price = as_float(r.get("close", np.nan), np.nan)
        if not np.isfinite(price) or price <= 0:
            price = as_float(r.get("daily_vwap", np.nan), np.nan)

        hit_score = as_float(r.get("hit_score", np.nan), np.nan)
        threshold = as_float(r.get("threshold", np.nan), np.nan)
        score_margin = as_float(r.get("score_margin", np.nan), np.nan)

        conf_mult = 1.0
        if np.isfinite(score_margin) and np.isfinite(threshold) and abs(threshold) > 1e-9:
            conf_mult = float(np.clip(1.0 + 0.20 * score_margin / max(abs(threshold), 1e-9), 0.80, 1.20))

        if str(label_mode).lower().startswith("hit") or str(label_mode).lower() == "hit":
            pred_prob = hit_score if np.isfinite(hit_score) else win_rate
            pred_return_bps = np.nan
        else:
            pred_return_bps = avg_return_bps * conf_mult if np.isfinite(avg_return_bps) else median_return_bps
            pred_prob = np.nan

        sector, sector_source = choose_sector(r, stock_code, context_sector_map)
        override_fields = apply_override_fields(stock_code, artifact, override_rows, recent_rows)

        entry_policy = as_text(r.get("entry_policy"), as_text(meta.get("entry_policy"), ""))
        entry_vwap_premium_bps = as_float(
            r.get("entry_vwap_premium_bps", meta.get("entry_vwap_premium_bps", 50.0)),
            50.0,
        )
        samples = as_text(
            r.get("samples"),
            as_text(r.get("sample_file"), as_text(meta.get("samples"), as_text(meta.get("sample_file"), ""))),
        )
        expected_return_col = as_text(r.get("expected_return_col"), "")
        if not expected_return_col:
            lm = str(label_mode).lower()
            expected_return_col = "trade_target_or_close_return" if lm.startswith("hit") or lm == "hit" else "trade_net_close_return"

        sig_rows.append({
            "stock_code": stock_code,
            "model_name": artifact,
            "label_mode": label_mode,
            "pred_return_bps": pred_return_bps,
            "pred_prob": pred_prob,
            "target_hit_bps": target_hit_bps,
            "price": price,
            "sector": sector,
            "sector_source": sector_source,
            "hit_score": hit_score,
            "threshold": threshold,
            "score_margin": score_margin,
            "entry_policy": entry_policy,
            "entry_vwap_premium_bps": entry_vwap_premium_bps,
            "samples": samples,
            "expected_return_col": expected_return_col,
            "metadata_path": str(meta_path) if meta_path else "",
            **override_fields,
        })

        met_rows.append({
            "stock_code": stock_code,
            "model_name": artifact,
            "label_mode": label_mode,
            "trades": trades,
            "win_rate": win_rate,
            "avg_return_bps": avg_return_bps,
            "median_return_bps": median_return_bps,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "target_hit_bps": target_hit_bps,
            "feature_group": meta.get("feature_group", ""),
            "base_model_name": meta.get("model_name", ""),
            "entry_policy": entry_policy,
            "entry_vwap_premium_bps": entry_vwap_premium_bps,
            "samples": samples,
            "expected_return_col": expected_return_col,
            "sector": sector,
            "sector_source": sector_source,
            **override_fields,
        })

        price_rows.append({"stock_code": stock_code, "price": price})

    sig_cols = list(dict.fromkeys(signal_columns + (list(sig_rows[0].keys()) if sig_rows else [])))
    met_cols = list(dict.fromkeys(metric_columns + (list(met_rows[0].keys()) if met_rows else [])))
    pd.DataFrame(sig_rows, columns=sig_cols).to_csv(signals_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(met_rows, columns=met_cols).to_csv(metrics_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(price_rows).drop_duplicates("stock_code", keep="last").to_csv(prices_out, index=False, encoding="utf-8-sig")

    return {"signals": signals_out, "metrics": metrics_out, "prices": prices_out}
'''


def patch_adapter() -> None:
    text = read(ADAPTER)
    text = ensure_as_text(text)
    start, end = function_bounds(text, "build_inputs", "make_account_template")
    write(ADAPTER, text[:start] + BUILD_INPUTS + "\n\n" + text[end:])


def patch_optimizer() -> None:
    text = read(OPTIMIZER)
    text = ensure_as_text(text)
    write(OPTIMIZER, text)


def verify_backtest_static() -> None:
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    text = read(BACKTEST)
    required = [
        "from model_training.optimize_nextday_vwap_model import add_trade_returns",
        "samples = add_trade_returns(",
        "entry_policy=lot.entry_policy",
        "expected_return_col=lot.expected_return_col",
        'buy_mask = (all_scores["signal"] == True)',
    ]
    missing = [x for x in required if x not in text]
    if missing:
        raise RuntimeError(f"backtest missing required markers: {missing}")

    run([sys.executable, "-m", "py_compile", str(BACKTEST)])

    mod = importlib.import_module("portfolio_decision.backtest_historical_score_portfolio")
    for cls_name in ["OpenLot", "TradeRecord"]:
        cls = getattr(mod, cls_name)
        if not is_dataclass(cls):
            raise RuntimeError(f"{cls_name} is not dataclass")
        seen_default = False
        bad = []
        for f in fields(cls):
            has_default = not (f.default.__class__.__name__ == "_MISSING_TYPE" and f.default_factory.__class__.__name__ == "_MISSING_TYPE")
            if has_default:
                seen_default = True
            elif seen_default:
                bad.append(f.name)
        if bad:
            raise RuntimeError(f"{cls_name} has non-default fields after default fields: {bad}")


def synthetic_adapter_test() -> None:
    sys.path.insert(0, str(Path.cwd()))
    mod = importlib.import_module("portfolio_decision.portfolio_confirm_from_buy_signals")
    mod = importlib.reload(mod)

    with tempfile.TemporaryDirectory(prefix="portfolio_adapter_test_") as td:
        base = Path(td)
        signal_dir = base / "signals"
        signal_dir.mkdir()
        saved_models = base / "saved_models"
        artifact = "artifact_close_profit_vwap_low"
        model_dir = saved_models / "600312.SH" / artifact
        model_dir.mkdir(parents=True)

        sample_path = base / "training_samples_with_sector.csv"
        sample_path.write_text("date,close,daily_vwap,next_day_close,next_day_high\n2026-01-05,10,10,10.1,10.2\n", encoding="utf-8")

        (model_dir / "metadata.json").write_text(json.dumps({
            "stock_code": "600312.SH",
            "label_mode": "close_profit",
            "entry_policy": "vwap_low",
            "entry_vwap_premium_bps": 50.0,
            "samples": str(sample_path),
            "validation_trade_metrics": {
                "avg_return": 0.01,
                "median_return": 0.005,
                "trades": 20,
                "win_rate": 0.6,
                "max_drawdown": -0.07,
                "profit_factor": 1.5,
            },
        }, ensure_ascii=False), encoding="utf-8")

        pd.DataFrame([{
            "stock_code": "600312.SH",
            "artifact_name": artifact,
            "close": 10.0,
            "daily_vwap": 10.0,
            "hit_score": 0.9,
            "threshold": 0.5,
            "score_margin": 0.4,
        }]).to_csv(signal_dir / "buy_signals.csv", index=False)

        paths = mod.build_inputs(signal_dir, saved_models, base / "out")
        for key in ["signals", "metrics"]:
            df = pd.read_csv(paths[key])
            for col in ["entry_policy", "entry_vwap_premium_bps", "samples", "expected_return_col"]:
                if col not in df.columns:
                    raise RuntimeError(f"{key} missing propagated column: {col}")
            row = df.iloc[0]
            if str(row["entry_policy"]) != "vwap_low":
                raise RuntimeError(f"{key} entry_policy mismatch: {row['entry_policy']}")
            if str(row["expected_return_col"]) != "trade_net_close_return":
                raise RuntimeError(f"{key} expected_return_col mismatch: {row['expected_return_col']}")
            if str(row["samples"]) != str(sample_path):
                raise RuntimeError(f"{key} samples mismatch: {row['samples']} != {sample_path}")


def synthetic_optimizer_test() -> None:
    sys.path.insert(0, str(Path.cwd()))
    mod = importlib.import_module("portfolio_decision.daily_portfolio_confirm_pyscipopt")
    mod = importlib.reload(mod)

    if not hasattr(mod, "as_text"):
        raise RuntimeError("optimizer still missing as_text()")

    row = pd.Series({
        "stock_code": "600312.SH",
        "model_name": "artifact_close_profit_vwap_low",
        "label_mode": "close_profit",
        "price": 10.0,
        "pred_return_bps": 80.0,
        "sector": "电网设备",
        "target_hit_bps": 50.0,
        "entry_policy": "vwap_low",
        "entry_vwap_premium_bps": 50.0,
        "samples": "/tmp/sample.csv",
        "expected_return_col": "trade_net_close_return",
        "metadata_path": "/tmp/metadata.json",
        "profit_factor": 1.5,
        "trades": 100,
        "median_return_bps": 10.0,
        "max_drawdown": -0.07,
    })
    df = pd.DataFrame([row])
    account = {"total_asset": 200000.0, "available_cash": 100000.0, "holdings": {}}
    candidates, rejected = mod.build_candidates(df, account, {"600312.SH": 0.02}, mod.DEFAULT_CONFIG)
    if not candidates:
        raise RuntimeError(f"optimizer synthetic candidate rejected: {rejected}")
    c = candidates[0]
    if c.entry_policy != "vwap_low":
        raise RuntimeError(f"optimizer entry_policy not propagated: {c.entry_policy}")
    if c.expected_return_col != "trade_net_close_return":
        raise RuntimeError(f"optimizer expected_return_col not propagated: {c.expected_return_col}")
    if c.samples != "/tmp/sample.csv":
        raise RuntimeError(f"optimizer samples not propagated: {c.samples}")


def validate_all() -> None:
    run([sys.executable, "-m", "py_compile", str(ADAPTER)])
    run([sys.executable, "-m", "py_compile", str(OPTIMIZER)])
    verify_backtest_static()
    synthetic_adapter_test()
    synthetic_optimizer_test()


def main() -> int:
    required = [BACKTEST, ADAPTER, OPTIMIZER]
    for p in required:
        if not p.exists():
            print(f"[ERROR] missing {p}", file=sys.stderr)
            return 2

    backup_root = Path("saved_data/patch_backups/portfolio_current_github_fix")
    backup_root.mkdir(parents=True, exist_ok=True)
    backups = {}
    for p in required:
        dst = backup_root / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        backups[p] = dst
        log(f"[BACKUP] {p} -> {dst}")

    try:
        patch_adapter()
        patch_optimizer()
        validate_all()
        log("[OK] portfolio current GitHub fix applied and self-tested")
        return 0
    except Exception as exc:
        for p, dst in backups.items():
            shutil.copy2(dst, p)
        print("[ROLLBACK] restored original files", file=sys.stderr)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
