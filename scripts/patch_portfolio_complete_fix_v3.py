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

BACKTEST = Path('portfolio_decision/backtest_historical_score_portfolio.py')
ADAPTER = Path('portfolio_decision/portfolio_confirm_from_buy_signals.py')
OPTIMIZER = Path('portfolio_decision/daily_portfolio_confirm_pyscipopt.py')


def run(cmd: list[str]) -> None:
    print('[RUN]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read(p: Path) -> str:
    return p.read_text(encoding='utf-8')


def write(p: Path, s: str) -> None:
    p.write_text(s, encoding='utf-8')


def ensure_as_text(text: str) -> str:
    if 'def as_text(' in text:
        return text
    marker = '''def as_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None or pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


'''
    helper = marker + '''def as_text(x: Any, default: str = "") -> str:
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
    if marker not in text:
        raise RuntimeError('as_float block not found; cannot insert as_text')
    return text.replace(marker, helper, 1)


def replace_func(text: str, name: str, next_name: str, new_body: str) -> str:
    start_marker = f'\ndef {name}('
    next_marker = f'\ndef {next_name}('
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f'function not found: {name}')
    start += 1
    end = text.find(next_marker, start)
    if end < 0:
        raise RuntimeError(f'next function not found after {name}: {next_name}')
    return text[:start] + new_body.strip() + '\n\n' + text[end + 1:]


def ensure_add_trade_import(text: str) -> str:
    if 'from model_training.optimize_nextday_vwap_model import add_trade_returns' in text:
        return text
    marker = '''if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
'''
    if marker not in text:
        raise RuntimeError('PROJECT_DIR sys.path marker not found')
    return text.replace(marker, marker + '\nfrom model_training.optimize_nextday_vwap_model import add_trade_returns\n', 1)


def ensure_add_trade_returns_call(text: str) -> str:
    if 'samples = add_trade_returns(' in text:
        return text
    func_pos = text.find('\ndef load_artifact_states(')
    if func_pos < 0:
        raise RuntimeError('load_artifact_states not found')
    pos = text.find('        cols = read_feature_columns(artifact_dir)', func_pos)
    if pos < 0:
        raise RuntimeError('cols = read_feature_columns marker not found')
    block = '''        # Same return/entry semantics as model_training/search_walk_forward_model_complexity.py.
        # Raw samples may not persist these fields; rebuild them per artifact metadata.
        samples = add_trade_returns(
            samples,
            cost_bps=as_float(meta.get("round_trip_cost_bps"), 1.7),
            target_bps=as_float(meta.get("target_hit_bps"), 50.0),
            entry_policy=str(meta.get("entry_policy") or "vwap_low"),
            entry_vwap_premium_bps=as_float(meta.get("entry_vwap_premium_bps"), 50.0),
        )

'''
    return text[:pos] + block + text[pos:]


def ensure_dataclass_fields(text: str, cls: str, anchor: str) -> str:
    pat = re.compile(rf'(@dataclass\s*\nclass {cls}:\n)(.*?)(\n\n(?:@dataclass\s*\nclass |\ndef |\Z))', re.S)
    m = pat.search(text)
    if not m:
        raise RuntimeError(f'dataclass {cls} not found')
    head, body, tail = m.group(1), m.group(2), m.group(3)
    moved = {'entry_policy', 'label_mode', 'expected_return_col', 'samples', 'metadata_path'}
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        name = stripped.split(':', 1)[0] if ':' in stripped else ''
        if name not in moved:
            lines.append(line)
    new_fields = [
        '    entry_policy: str = ""',
        '    label_mode: str = ""',
        '    expected_return_col: str = ""',
        '    samples: str = ""',
        '    metadata_path: str = ""',
    ]
    out, inserted = [], False
    for line in lines:
        out.append(line)
        if line.strip().startswith(anchor + ':'):
            out.extend(new_fields)
            inserted = True
    if not inserted:
        raise RuntimeError(f'anchor {anchor} not found in {cls}')
    return text[:m.start()] + head + '\n'.join(out) + tail + text[m.end():]


def insert_after(text: str, marker: str, insert: str, label: str) -> str:
    if insert.strip() in text:
        return text
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError(f'marker not found: {label}')
    return text[:pos + len(marker)] + insert + text[pos + len(marker):]


def patch_backtest() -> None:
    text = read(BACKTEST)
    text = ensure_add_trade_import(text)
    text = ensure_add_trade_returns_call(text)
    text = ensure_dataclass_fields(text, 'OpenLot', 'utility_bps')
    text = ensure_dataclass_fields(text, 'TradeRecord', 'exit_reason')
    text = insert_after(text, '                    utility_bps=as_float(r.get("utility_bps", np.nan), np.nan),\n', '''                    entry_policy=str(r.get("entry_policy", "")),
                    label_mode=str(r.get("label_mode", "")),
                    expected_return_col=str(r.get("expected_return_col", "")),
                    samples=str(r.get("samples", "")),
                    metadata_path=str(r.get("metadata_path", "")),
''', 'OpenLot propagation')
    text = insert_after(text, '                    exit_reason="scheduled_hold_days",\n', '''                    entry_policy=lot.entry_policy,
                    label_mode=lot.label_mode,
                    expected_return_col=lot.expected_return_col,
                    samples=lot.samples,
                    metadata_path=lot.metadata_path,
''', 'TradeRecord scheduled propagation')
    text = insert_after(text, '                exit_reason="force_close_at_end",\n', '''                entry_policy=lot.entry_policy,
                label_mode=lot.label_mode,
                expected_return_col=lot.expected_return_col,
                samples=lot.samples,
                metadata_path=lot.metadata_path,
''', 'TradeRecord force-close propagation')
    if 'buy_mask = (all_scores["signal"] == True)' not in text:
        old = '''            buy = all_scores[(all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")].copy()
            buy = buy.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~all_scores.index.isin(buy.index)].copy()
'''
        new = '''            buy_mask = (all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")
            buy = all_scores[buy_mask].copy()
            buy = buy.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~buy_mask].copy()
'''
        if old not in text:
            raise RuntimeError('rejected_scores old block not found')
        text = text.replace(old, new, 1)
    write(BACKTEST, text)
    print(f'[PATCHED] {BACKTEST}')


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
    text = read(ADAPTER)
    text = ensure_as_text(text)
    text = replace_func(text, 'build_inputs', 'make_account_template', BUILD_INPUTS)
    write(ADAPTER, text)
    print(f'[PATCHED] {ADAPTER}')


def patch_optimizer() -> None:
    text = read(OPTIMIZER)
    text = ensure_as_text(text)
    if 'expected_return_col: str' not in text:
        old = '''    fail_loss_bps: float
    ev_bps: float
'''
        new = '''    fail_loss_bps: float
    entry_policy: str
    entry_vwap_premium_bps: float
    samples: str
    expected_return_col: str
    metadata_path: str
    ev_bps: float
'''
        if old not in text:
            raise RuntimeError('Candidate field marker not found')
        text = text.replace(old, new, 1)
    if 'entry_policy = as_text(get_row_field(row, "entry_policy"' not in text:
        old = '''        target_hit_bps = as_float(get_row_field(row, "target_hit_bps", 80.0 if "80" in label_mode else 50.0), 80.0)
        pred_prob = parse_rate_decimal(get_row_field(row, "pred_prob", get_row_field(row, "win_rate", cfg["default_hit_prob"])), cfg["default_hit_prob"])
'''
        new = '''        target_hit_bps = as_float(get_row_field(row, "target_hit_bps", 80.0 if "80" in label_mode else 50.0), 80.0)
        entry_policy = as_text(get_row_field(row, "entry_policy", ""))
        entry_vwap_premium_bps = as_float(get_row_field(row, "entry_vwap_premium_bps", 50.0), 50.0)
        samples = as_text(get_row_field(row, "samples", ""))
        expected_return_col = as_text(get_row_field(row, "expected_return_col", ""))
        metadata_path = as_text(get_row_field(row, "metadata_path", ""))
        pred_prob = parse_rate_decimal(get_row_field(row, "pred_prob", get_row_field(row, "win_rate", cfg["default_hit_prob"])), cfg["default_hit_prob"])
'''
        if old not in text:
            raise RuntimeError('Candidate extraction marker not found')
        text = text.replace(old, new, 1)
    if 'entry_policy=entry_policy' not in text:
        old = '''            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps, ev_bps=ev_bps,
            quality_weight=quality, tier=tier, tier_multiplier=tier_mult, vol_daily=vol_daily,
'''
        new = '''            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps,
            entry_policy=entry_policy, entry_vwap_premium_bps=entry_vwap_premium_bps,
            samples=samples, expected_return_col=expected_return_col, metadata_path=metadata_path,
            ev_bps=ev_bps,
            quality_weight=quality, tier=tier, tier_multiplier=tier_mult, vol_daily=vol_daily,
'''
        if old not in text:
            raise RuntimeError('Candidate construction marker not found')
        text = text.replace(old, new, 1)
    write(OPTIMIZER, text)
    print(f'[PATCHED] {OPTIMIZER}')


def validate_marker(path: Path, markers: list[str]) -> None:
    text = read(path)
    missing = [m for m in markers if m not in text]
    if missing:
        raise RuntimeError(f'{path} missing markers: {missing}')


def synthetic_adapter_test() -> None:
    sys.path.insert(0, str(Path.cwd().resolve()))
    mod = importlib.import_module('portfolio_decision.portfolio_confirm_from_buy_signals')
    mod = importlib.reload(mod)
    with tempfile.TemporaryDirectory(prefix='portfolio_full_fix_test_') as td:
        base = Path(td)
        sigdir = base / 'signals'; sigdir.mkdir()
        sm = base / 'saved_models'
        artifact = 'nextday_vwap_low_close_profit_fake'
        mdir = sm / '600312.SH' / artifact; mdir.mkdir(parents=True)
        sample = base / 'training_samples.csv'
        sample.write_text('date,close,daily_vwap,next_day_close,next_day_high\n2026-01-05,10,10,10.1,10.2\n', encoding='utf-8')
        (mdir / 'metadata.json').write_text(json.dumps({
            'stock_code':'600312.SH','label_mode':'close_profit','entry_policy':'vwap_low',
            'entry_vwap_premium_bps':50.0,'samples':str(sample),
            'validation_trade_metrics':{'avg_return':0.01,'median_return':0.005,'trades':20,'win_rate':0.6,'max_drawdown':-0.07,'profit_factor':1.5}
        }, ensure_ascii=False), encoding='utf-8')
        pd.DataFrame([{'stock_code':'600312.SH','artifact_name':artifact,'close':10.0,'daily_vwap':10.0,'hit_score':0.9,'threshold':0.5,'score_margin':0.4}]).to_csv(sigdir / 'buy_signals.csv', index=False)
        paths = mod.build_inputs(signal_dir=sigdir, saved_models=sm, out_input_dir=base / 'out')
        for key in ['signals', 'metrics']:
            df = pd.read_csv(paths[key])
            for col in ['entry_policy','entry_vwap_premium_bps','samples','expected_return_col']:
                if col not in df.columns:
                    raise RuntimeError(f'synthetic {key} missing {col}')
            if str(df.loc[0,'samples']) != str(sample):
                raise RuntimeError(f'synthetic {key} samples mismatch')
            if str(df.loc[0,'entry_policy']) != 'vwap_low':
                raise RuntimeError(f'synthetic {key} entry_policy mismatch')
            if str(df.loc[0,'expected_return_col']) != 'trade_net_close_return':
                raise RuntimeError(f'synthetic {key} expected_return_col mismatch')


def validate_all() -> None:
    for p in [BACKTEST, ADAPTER, OPTIMIZER]:
        run([sys.executable, '-m', 'py_compile', str(p)])
    validate_marker(BACKTEST, [
        'from model_training.optimize_nextday_vwap_model import add_trade_returns',
        'samples = add_trade_returns(',
        'entry_policy=lot.entry_policy',
        'expected_return_col=lot.expected_return_col',
        'buy_mask = (all_scores["signal"] == True)',
    ])
    validate_marker(ADAPTER, [
        'def as_text(',
        '"samples": samples',
        '"expected_return_col": expected_return_col',
        '"entry_vwap_premium_bps": entry_vwap_premium_bps',
    ])
    validate_marker(OPTIMIZER, [
        'entry_policy: str',
        'entry_vwap_premium_bps: float',
        'samples: str',
        'expected_return_col: str',
        'metadata_path: str',
        'entry_policy=entry_policy',
        'expected_return_col=expected_return_col',
    ])
    mod = importlib.import_module('portfolio_decision.backtest_historical_score_portfolio')
    for cls_name in ['OpenLot','TradeRecord']:
        cls = getattr(mod, cls_name)
        if not is_dataclass(cls):
            raise RuntimeError(f'{cls_name} not dataclass')
        seen_default, bad = False, []
        for f in fields(cls):
            has_default = not (f.default.__class__.__name__ == '_MISSING_TYPE' and f.default_factory.__class__.__name__ == '_MISSING_TYPE')
            if has_default:
                seen_default = True
            elif seen_default:
                bad.append(f.name)
        if bad:
            raise RuntimeError(f'{cls_name} dataclass order invalid: {bad}')
    synthetic_adapter_test()


def main() -> int:
    paths = [BACKTEST, ADAPTER, OPTIMIZER]
    for p in paths:
        if not p.exists():
            print(f'[ERROR] missing {p}', file=sys.stderr); return 2
    backup_root = Path('saved_data/patch_backups/portfolio_complete_fix_v3')
    backup_root.mkdir(parents=True, exist_ok=True)
    backups = {}
    for p in paths:
        b = backup_root / p
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, b)
        backups[p] = b
        print(f'[BACKUP] {p} -> {b}', flush=True)
    try:
        patch_backtest()
        patch_adapter()
        patch_optimizer()
        validate_all()
        print('[OK] portfolio complete fix v3 applied and self-tested')
        return 0
    except Exception as exc:
        for p,b in backups.items():
            shutil.copy2(b, p)
        print('[ROLLBACK] restored original files', file=sys.stderr)
        print(f'[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
