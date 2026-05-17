#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path

import pandas as pd

PROJECT = Path.cwd()
BACKTEST = Path("portfolio_decision/backtest_historical_score_portfolio.py")
ADAPTER = Path("portfolio_decision/portfolio_confirm_from_buy_signals.py")
OPTIMIZER = Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str]) -> None:
    log("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def function_bounds(text: str, name: str, next_name: str) -> tuple[int, int]:
    start = text.find(f"\ndef {name}(")
    if start < 0:
        raise RuntimeError(f"function not found: {name}")
    start += 1
    end = text.find(f"\ndef {next_name}(", start)
    if end < 0:
        raise RuntimeError(f"next function not found after {name}: {next_name}")
    return start, end + 1


AS_TEXT_BLOCK = '''
def as_text(x: Any, default: str = "") -> str:
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


def insert_before_function(text: str, func_name: str, block: str) -> str:
    if block.strip() in text:
        return text
    pos = text.find(f"\ndef {func_name}(")
    if pos < 0:
        raise RuntimeError(f"cannot insert before missing function: {func_name}")
    return text[:pos + 1] + block.strip() + "\n\n" + text[pos + 1:]


def ensure_as_text(text: str) -> str:
    if "def as_text(" in text:
        return text
    for fn in ["find_metadata", "parse_rate_decimal", "load_json", "parse_bool_flag"]:
        if f"\ndef {fn}(" in text:
            return insert_before_function(text, fn, AS_TEXT_BLOCK)
    raise RuntimeError("cannot insert as_text: no stable helper function found")


def ensure_import_add_trade_returns(text: str) -> str:
    if "from model_training.optimize_nextday_vwap_model import add_trade_returns" in text:
        return text
    marker = """if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
"""
    if marker not in text:
        raise RuntimeError("cannot insert add_trade_returns import: PROJECT_DIR sys.path block not found")
    return text.replace(marker, marker + "\nfrom model_training.optimize_nextday_vwap_model import add_trade_returns\n", 1)


def ensure_samples_add_trade_returns(text: str) -> str:
    if "samples = add_trade_returns(" in text:
        return text
    start = text.find("\ndef load_artifact_states(")
    if start < 0:
        raise RuntimeError("load_artifact_states not found")
    end = text.find("\ndef build_history_close_from_states(", start)
    if end < 0:
        raise RuntimeError("load_artifact_states end marker not found")
    marker = "        cols = read_feature_columns(artifact_dir)"
    pos = text.find(marker, start, end)
    if pos < 0:
        raise RuntimeError("cannot insert add_trade_returns: read_feature_columns marker not found inside load_artifact_states")
    block = """        # Match model_training/search_walk_forward_model_complexity.py:
        # training_samples*.csv may not persist these columns, so rebuild them
        # using the artifact's own metadata before scoring and settlement.
        samples = add_trade_returns(
            samples,
            cost_bps=as_float(meta.get("round_trip_cost_bps"), 1.7),
            target_bps=as_float(meta.get("target_hit_bps"), 50.0),
            entry_policy=str(meta.get("entry_policy") or "vwap_low"),
            entry_vwap_premium_bps=as_float(meta.get("entry_vwap_premium_bps"), 50.0),
        )

"""
    return text[:pos] + block + text[pos:]


def ensure_usage_fields(text: str, class_name: str, anchor: str) -> str:
    pat = re.compile(rf"(@dataclass\s*\nclass {class_name}:\n)(.*?)(\n\n(?:@dataclass\s*\nclass |\ndef |\Z))", re.S)
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"dataclass not found: {class_name}")
    head, body, tail = m.group(1), m.group(2), m.group(3)
    names = {"entry_policy", "label_mode", "expected_return_col", "samples", "metadata_path"}
    kept = []
    for line in body.splitlines():
        stripped = line.strip()
        name = stripped.split(":", 1)[0] if ":" in stripped else ""
        if name not in names:
            kept.append(line)
    insert = [
        '    entry_policy: str = ""',
        '    label_mode: str = ""',
        '    expected_return_col: str = ""',
        '    samples: str = ""',
        '    metadata_path: str = ""',
    ]
    out, inserted = [], False
    for line in kept:
        out.append(line)
        if line.strip().startswith(anchor + ":"):
            out.extend(insert)
            inserted = True
    if not inserted:
        raise RuntimeError(f"anchor field {anchor} not found in {class_name}")
    return text[:m.start()] + head + "\n".join(out) + tail + text[m.end():]


def ensure_candidate_fields(text: str) -> str:
    if "expected_return_col: str" in text and "metadata_path: str" in text:
        return text
    pat = re.compile(r"(@dataclass\s*\nclass Candidate:\n)(.*?)(\n\ndef )", re.S)
    m = pat.search(text)
    if not m:
        raise RuntimeError("Candidate dataclass not found")
    head, body, tail = m.group(1), m.group(2), m.group(3)
    fields_to_add = [
        "    entry_policy: str",
        "    entry_vwap_premium_bps: float",
        "    samples: str",
        "    expected_return_col: str",
        "    metadata_path: str",
    ]
    out, inserted = [], False
    for line in body.splitlines():
        out.append(line)
        if line.strip().startswith("fail_loss_bps:"):
            out.extend(fields_to_add)
            inserted = True
    if not inserted:
        raise RuntimeError("fail_loss_bps not found in Candidate")
    return text[:m.start()] + head + "\n".join(out) + tail + text[m.end():]


def ensure_openlot_constructor_fields(text: str) -> str:
    if 'expected_return_col=str(r.get("expected_return_col"' in text:
        return text
    marker = '                    utility_bps=as_float(r.get("utility_bps", np.nan), np.nan),\n'
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError("OpenLot utility_bps constructor marker not found")
    insert = """                    entry_policy=str(r.get("entry_policy", "")),
                    label_mode=str(r.get("label_mode", "")),
                    expected_return_col=str(r.get("expected_return_col", "")),
                    samples=str(r.get("samples", "")),
                    metadata_path=str(r.get("metadata_path", "")),
"""
    return text[:pos + len(marker)] + insert + text[pos + len(marker):]


def ensure_traderecord_fields_after_exit(text: str, exit_reason: str, indent: str) -> str:
    marker = f'{indent}exit_reason="{exit_reason}",\n'
    if marker + f"{indent}entry_policy=lot.entry_policy" in text:
        return text
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError(f"TradeRecord exit_reason marker not found: {exit_reason}")
    insert = f"""{indent}entry_policy=lot.entry_policy,
{indent}label_mode=lot.label_mode,
{indent}expected_return_col=lot.expected_return_col,
{indent}samples=lot.samples,
{indent}metadata_path=lot.metadata_path,
"""
    return text[:pos + len(marker)] + insert + text[pos + len(marker):]


def ensure_rejected_mask(text: str) -> str:
    if 'buy_mask = (all_scores["signal"] == True)' in text:
        return text
    old = """            buy = all_scores[(all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")].copy()
            buy = buy.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~all_scores.index.isin(buy.index)].copy()
"""
    new = """            buy_mask = (all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")
            buy = all_scores[buy_mask].copy()
            buy = buy.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~buy_mask].copy()
"""
    if old not in text:
        raise RuntimeError("old rejected_scores block not found")
    return text.replace(old, new, 1)


BUILD_INPUTS = r'''
def build_inputs(
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

    if raw.empty:
        pd.DataFrame(columns=[
            "stock_code", "model_name", "label_mode",
            "entry_policy", "entry_vwap_premium_bps", "samples", "expected_return_col", "metadata_path"
        ]).to_csv(signals_out, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=[
            "stock_code", "model_name", "entry_policy", "entry_vwap_premium_bps", "samples", "expected_return_col"
        ]).to_csv(metrics_out, index=False, encoding="utf-8-sig")
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

        target_hit_bps = as_float(meta.get("target_hit_bps", r.get("target_hit_bps", 80 if "80" in artifact else 50)), 50.0)

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

    pd.DataFrame(sig_rows).to_csv(signals_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(met_rows).to_csv(metrics_out, index=False, encoding="utf-8-sig")
    pd.DataFrame(price_rows).drop_duplicates("stock_code", keep="last").to_csv(prices_out, index=False, encoding="utf-8-sig")

    return {"signals": signals_out, "metrics": metrics_out, "prices": prices_out}
'''


def patch_adapter() -> None:
    text = ensure_as_text(read(ADAPTER))
    start, end = function_bounds(text, "build_inputs", "make_account_template")
    write(ADAPTER, text[:start] + BUILD_INPUTS.strip() + "\n\n" + text[end:])
    log(f"[PATCHED] {ADAPTER}")


def patch_backtest() -> None:
    text = read(BACKTEST)
    text = ensure_import_add_trade_returns(text)
    text = ensure_samples_add_trade_returns(text)
    text = ensure_usage_fields(text, "OpenLot", "utility_bps")
    text = ensure_usage_fields(text, "TradeRecord", "exit_reason")
    text = ensure_openlot_constructor_fields(text)
    text = ensure_traderecord_fields_after_exit(text, "scheduled_hold_days", "                    ")
    text = ensure_traderecord_fields_after_exit(text, "force_close_at_end", "                ")
    text = ensure_rejected_mask(text)
    write(BACKTEST, text)
    log(f"[PATCHED] {BACKTEST}")


def patch_optimizer() -> None:
    text = ensure_as_text(read(OPTIMIZER))
    text = ensure_candidate_fields(text)
    if 'entry_policy = as_text(get_row_field(row, "entry_policy"' not in text:
        target = '        target_hit_bps = as_float(get_row_field(row, "target_hit_bps", 80.0 if "80" in label_mode else 50.0), 80.0)\n'
        pos = text.find(target)
        if pos < 0:
            raise RuntimeError("optimizer target_hit_bps line not found")
        insert = """        entry_policy = as_text(get_row_field(row, "entry_policy", ""))
        entry_vwap_premium_bps = as_float(get_row_field(row, "entry_vwap_premium_bps", 50.0), 50.0)
        samples = as_text(get_row_field(row, "samples", ""))
        expected_return_col = as_text(get_row_field(row, "expected_return_col", ""))
        metadata_path = as_text(get_row_field(row, "metadata_path", ""))
"""
        text = text[:pos + len(target)] + insert + text[pos + len(target):]

    if "entry_policy=entry_policy" not in text:
        marker = '            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps, ev_bps=ev_bps,\n'
        pos = text.find(marker)
        if pos < 0:
            raise RuntimeError("optimizer Candidate construction line not found")
        repl = """            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps,
            entry_policy=entry_policy, entry_vwap_premium_bps=entry_vwap_premium_bps,
            samples=samples, expected_return_col=expected_return_col, metadata_path=metadata_path,
            ev_bps=ev_bps,
"""
        text = text[:pos] + repl + text[pos + len(marker):]
    write(OPTIMIZER, text)
    log(f"[PATCHED] {OPTIMIZER}")


def validate_markers(path: Path, markers: list[str]) -> None:
    text = read(path)
    missing = [m for m in markers if m not in text]
    if missing:
        raise RuntimeError(f"{path} missing markers: {missing}")


def synthetic_adapter_test() -> None:
    sys.path.insert(0, str(PROJECT.resolve()))
    mod = importlib.import_module("portfolio_decision.portfolio_confirm_from_buy_signals")
    mod = importlib.reload(mod)
    with tempfile.TemporaryDirectory(prefix="portfolio_full_fix_test_") as td:
        base = Path(td)
        signal_dir = base / "signals"
        signal_dir.mkdir()
        saved_models = base / "saved_models"
        artifact = "nextday_vwap_low_close_profit_fake"
        model_dir = saved_models / "600312.SH" / artifact
        model_dir.mkdir(parents=True)
        sample_path = base / "training_samples.csv"
        sample_path.write_text("date,close,daily_vwap,next_day_close,next_day_high\n2026-01-05,10,10,10.1,10.2\n", encoding="utf-8")
        (model_dir / "metadata.json").write_text(json.dumps({
            "stock_code": "600312.SH", "label_mode": "close_profit", "entry_policy": "vwap_low",
            "entry_vwap_premium_bps": 50.0, "samples": str(sample_path),
            "validation_trade_metrics": {"avg_return": 0.01, "median_return": 0.005, "trades": 20, "win_rate": 0.6, "max_drawdown": -0.07, "profit_factor": 1.5},
        }, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame([{"stock_code": "600312.SH", "artifact_name": artifact, "close": 10.0, "daily_vwap": 10.0, "hit_score": 0.9, "threshold": 0.5, "score_margin": 0.4}]).to_csv(signal_dir / "buy_signals.csv", index=False)
        paths = mod.build_inputs(signal_dir=signal_dir, saved_models=saved_models, out_input_dir=base / "out")
        for key in ["signals", "metrics"]:
            df = pd.read_csv(paths[key])
            for col in ["entry_policy", "entry_vwap_premium_bps", "samples", "expected_return_col"]:
                if col not in df.columns:
                    raise RuntimeError(f"synthetic {key} missing {col}")
            if str(df.loc[0, "samples"]) != str(sample_path):
                raise RuntimeError(f"synthetic {key} samples mismatch: {df.loc[0, 'samples']} != {sample_path}")
            if str(df.loc[0, "entry_policy"]) != "vwap_low":
                raise RuntimeError(f"synthetic {key} entry_policy mismatch")
            if str(df.loc[0, "expected_return_col"]) != "trade_net_close_return":
                raise RuntimeError(f"synthetic {key} expected_return_col mismatch")


def validate_all() -> None:
    for f in [BACKTEST, ADAPTER, OPTIMIZER]:
        run([sys.executable, "-m", "py_compile", str(f)])
    validate_markers(ADAPTER, ["def as_text(", '"samples": samples', '"expected_return_col": expected_return_col', '"entry_vwap_premium_bps": entry_vwap_premium_bps'])
    validate_markers(BACKTEST, ["from model_training.optimize_nextday_vwap_model import add_trade_returns", "samples = add_trade_returns(", "entry_policy=lot.entry_policy", "expected_return_col=lot.expected_return_col", 'buy_mask = (all_scores["signal"] == True)'])
    validate_markers(OPTIMIZER, ["entry_policy: str", "entry_vwap_premium_bps: float", "samples: str", "expected_return_col: str", "metadata_path: str", "entry_policy=entry_policy", "expected_return_col=expected_return_col"])
    sys.path.insert(0, str(PROJECT.resolve()))
    mod = importlib.import_module("portfolio_decision.backtest_historical_score_portfolio")
    for cls_name in ["OpenLot", "TradeRecord"]:
        cls = getattr(mod, cls_name)
        if not is_dataclass(cls):
            raise RuntimeError(f"{cls_name} is not dataclass")
        seen_default, bad = False, []
        for f in fields(cls):
            has_default = not (f.default.__class__.__name__ == "_MISSING_TYPE" and f.default_factory.__class__.__name__ == "_MISSING_TYPE")
            if has_default:
                seen_default = True
            elif seen_default:
                bad.append(f.name)
        if bad:
            raise RuntimeError(f"{cls_name} dataclass field order invalid: {bad}")
    synthetic_adapter_test()


def main() -> int:
    paths = [BACKTEST, ADAPTER, OPTIMIZER]
    for p in paths:
        if not p.exists():
            print(f"[ERROR] missing {p}", file=sys.stderr)
            return 2
    backup_root = Path("saved_data/patch_backups/portfolio_complete_fix_v4")
    backup_root.mkdir(parents=True, exist_ok=True)
    backups = {}
    for p in paths:
        dst = backup_root / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        backups[p] = dst
        log(f"[BACKUP] {p} -> {dst}")
    try:
        patch_backtest()
        patch_adapter()
        patch_optimizer()
        validate_all()
        log("[OK] complete portfolio fix v4 applied and self-tested")
        return 0
    except Exception as exc:
        for p, dst in backups.items():
            shutil.copy2(dst, p)
        print("[ROLLBACK] restored original files", file=sys.stderr)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
