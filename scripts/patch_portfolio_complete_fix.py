#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

BACKTEST = Path("portfolio_decision/backtest_historical_score_portfolio.py")
ADAPTER = Path("portfolio_decision/portfolio_confirm_from_buy_signals.py")
OPTIMIZER = Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py")


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old not in text:
        print(f"[WARN] marker not found: {label}")
        return text, False
    return text.replace(old, new, 1), True


def ensure_as_text(text: str) -> tuple[str, bool]:
    if "def as_text(" in text:
        return text, False
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
        print("[WARN] as_float marker not found; cannot insert as_text")
        return text, False
    return text.replace(marker, helper, 1), True


def fix_trade_record_order_in_body(body: str) -> str:
    move_names = {"entry_policy", "label_mode", "expected_return_col", "samples", "metadata_path"}
    moved, kept = [], []
    for line in body.splitlines():
        stripped = line.strip()
        name = stripped.split(":", 1)[0] if ":" in stripped else ""
        if name in move_names:
            moved.append(line)
        else:
            kept.append(line)
    if not moved:
        return body
    out, inserted = [], False
    for line in kept:
        out.append(line)
        if line.strip().startswith("exit_reason:"):
            out.extend(moved)
            inserted = True
    return "\n".join(out) if inserted else body


def patch_dataclass(text: str, class_name: str, insert_after_field: str, fields_block: str) -> tuple[str, bool]:
    pat = re.compile(rf"(@dataclass\s*\nclass {class_name}:\n)(.*?)(\n\n(?:@dataclass\s*\nclass |def |class |\Z))", re.S)
    m = pat.search(text)
    if not m:
        print(f"[WARN] dataclass not found: {class_name}")
        return text, False
    head, body, tail = m.group(1), m.group(2), m.group(3)
    changed = False
    if class_name == "TradeRecord":
        fixed = fix_trade_record_order_in_body(body)
        if fixed != body:
            body = fixed
            changed = True
    if "expected_return_col" not in body or "metadata_path" not in body:
        lines, out, inserted = body.splitlines(), [], False
        for line in lines:
            out.append(line)
            if line.strip().startswith(insert_after_field + ":"):
                out.extend(fields_block.rstrip("\n").splitlines())
                inserted = True
        if inserted:
            body = "\n".join(out)
            changed = True
        else:
            print(f"[WARN] field {insert_after_field} not found in {class_name}")
    if not changed:
        return text, False
    return text[:m.start()] + head + body + tail + text[m.end():], True


def patch_backtest() -> None:
    if not BACKTEST.exists():
        raise SystemExit(f"[ERROR] missing {BACKTEST}")
    text = BACKTEST.read_text(encoding="utf-8")
    original = text

    import_marker = '''if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


DEFAULT_EVAL_CONFIG = {
'''
    import_new = '''if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model_training.optimize_nextday_vwap_model import add_trade_returns


DEFAULT_EVAL_CONFIG = {
'''
    if "from model_training.optimize_nextday_vwap_model import add_trade_returns" not in text:
        text, _ = replace_once(text, import_marker, import_new, "backtest import add_trade_returns")

    load_marker = '''        samples = samples.dropna(subset=["date"]).sort_values("date")

        cols = read_feature_columns(artifact_dir)
'''
    load_new = '''        samples = samples.dropna(subset=["date"]).sort_values("date")

        # Match model_training/search_walk_forward_model_complexity.py:
        # training_samples*.csv may not persist these columns, so rebuild them
        # using the artifact's own metadata before scoring and settlement.
        samples = add_trade_returns(
            samples,
            cost_bps=as_float(meta.get("round_trip_cost_bps"), 1.7),
            target_bps=as_float(meta.get("target_hit_bps"), 50.0),
            entry_policy=str(meta.get("entry_policy") or "vwap_low"),
            entry_vwap_premium_bps=as_float(meta.get("entry_vwap_premium_bps"), 50.0),
        )

        cols = read_feature_columns(artifact_dir)
'''
    if "training_samples*.csv may not persist these columns" not in text:
        text, _ = replace_once(text, load_marker, load_new, "backtest add_trade_returns")

    default_fields = '''    entry_policy: str = ""
    label_mode: str = ""
    expected_return_col: str = ""
    samples: str = ""
    metadata_path: str = ""
'''
    text, _ = patch_dataclass(text, "OpenLot", "utility_bps", default_fields)
    text, _ = patch_dataclass(text, "TradeRecord", "exit_reason", default_fields)

    openlot_marker = '''                    utility_bps=as_float(r.get("utility_bps", np.nan), np.nan),
                ))
'''
    openlot_new = '''                    utility_bps=as_float(r.get("utility_bps", np.nan), np.nan),
                    entry_policy=str(r.get("entry_policy", "")),
                    label_mode=str(r.get("label_mode", "")),
                    expected_return_col=str(r.get("expected_return_col", "")),
                    samples=str(r.get("samples", "")),
                    metadata_path=str(r.get("metadata_path", "")),
                ))
'''
    if 'expected_return_col=str(r.get("expected_return_col"' not in text:
        text, _ = replace_once(text, openlot_marker, openlot_new, "backtest OpenLot fields")

    scheduled_marker = '''                    utility_bps=lot.utility_bps,
                    exit_reason="scheduled_hold_days",
'''
    scheduled_new = '''                    utility_bps=lot.utility_bps,
                    exit_reason="scheduled_hold_days",
                    entry_policy=lot.entry_policy,
                    label_mode=lot.label_mode,
                    expected_return_col=lot.expected_return_col,
                    samples=lot.samples,
                    metadata_path=lot.metadata_path,
'''
    if 'exit_reason="scheduled_hold_days",\n                    entry_policy=lot.entry_policy' not in text:
        text, _ = replace_once(text, scheduled_marker, scheduled_new, "backtest scheduled TradeRecord fields")

    force_marker = '''                utility_bps=lot.utility_bps,
                exit_reason="force_close_at_end",
'''
    force_new = '''                utility_bps=lot.utility_bps,
                exit_reason="force_close_at_end",
                entry_policy=lot.entry_policy,
                label_mode=lot.label_mode,
                expected_return_col=lot.expected_return_col,
                samples=lot.samples,
                metadata_path=lot.metadata_path,
'''
    if 'exit_reason="force_close_at_end",\n                entry_policy=lot.entry_policy' not in text:
        text, _ = replace_once(text, force_marker, force_new, "backtest force-close TradeRecord fields")

    rejected_old = '''            buy = all_scores[(all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")].copy()
            buy = buy.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~all_scores.index.isin(buy.index)].copy()
'''
    rejected_new = '''            buy_mask = (all_scores["signal"] == True) & (all_scores["reject_reason"].astype(str) == "")
            buy = all_scores[buy_mask].copy()
            buy = buy.sort_values(["score_margin", "hit_score"], ascending=False).reset_index(drop=True)
            buy["rank"] = np.arange(1, len(buy) + 1)
            rejected = all_scores.loc[~buy_mask].copy()
'''
    if 'buy_mask = (all_scores["signal"] == True)' not in text:
        text, _ = replace_once(text, rejected_old, rejected_new, "backtest rejected mask")

    if text != original:
        BACKTEST.write_text(text, encoding="utf-8")
        print(f"[PATCHED] {BACKTEST}")
    else:
        print(f"[UNCHANGED] {BACKTEST}")


def patch_adapter() -> None:
    if not ADAPTER.exists():
        raise SystemExit(f"[ERROR] missing {ADAPTER}")
    text = ADAPTER.read_text(encoding="utf-8")
    original = text
    text, _ = ensure_as_text(text)

    label_old = '''        label_mode = str(meta.get("label_mode", r.get("label_mode", ""))).strip()
        if not label_mode:
            label_mode = "hit" if "hit" in artifact.lower() else "close_profit"
'''
    label_new = '''        label_mode = as_text(meta.get("label_mode"), as_text(r.get("label_mode"), ""))
        if not label_mode:
            label_mode = "hit" if "hit" in artifact.lower() else "close_profit"
'''
    if label_old in text:
        text = text.replace(label_old, label_new, 1)

    sector_pattern = re.compile(
        r'        sector = r\.get\("sector", r\.get\("industry", "UNKNOWN"\)\)\n\n'
        r'(?:        entry_policy = .*?\n        sig_rows\.append\(\{\n|        sig_rows\.append\(\{\n)',
        re.S,
    )
    sector_block = '''        sector = r.get("sector", r.get("industry", "UNKNOWN"))

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
'''
    if 'entry_policy = as_text(r.get("entry_policy")' not in text:
        text, n = sector_pattern.subn(sector_block, text, count=1)
        if n == 0:
            print("[WARN] adapter sector block not patched")

    sig_marker = '''            "metadata_path": str(meta_path) if meta_path else "",
        })
'''
    sig_new = '''            "entry_policy": entry_policy,
            "entry_vwap_premium_bps": entry_vwap_premium_bps,
            "samples": samples,
            "expected_return_col": expected_return_col,
            "metadata_path": str(meta_path) if meta_path else "",
        })
'''
    if '"expected_return_col": expected_return_col' not in text:
        text, _ = replace_once(text, sig_marker, sig_new, "adapter sig rows")

    met_marker = '''            "entry_policy": meta.get("entry_policy", ""),
            "sector": sector,
        })
'''
    met_new = '''            "entry_policy": entry_policy,
            "entry_vwap_premium_bps": entry_vwap_premium_bps,
            "samples": samples,
            "expected_return_col": expected_return_col,
            "sector": sector,
        })
'''
    if '"samples": samples' not in text:
        text, _ = replace_once(text, met_marker, met_new, "adapter metrics rows")

    if text != original:
        ADAPTER.write_text(text, encoding="utf-8")
        print(f"[PATCHED] {ADAPTER}")
    else:
        print(f"[UNCHANGED] {ADAPTER}")


def patch_optimizer() -> None:
    if not OPTIMIZER.exists():
        raise SystemExit(f"[ERROR] missing {OPTIMIZER}")
    text = OPTIMIZER.read_text(encoding="utf-8")
    original = text
    text, _ = ensure_as_text(text)

    if "expected_return_col: str" not in text:
        marker = '''    fail_loss_bps: float
    ev_bps: float
'''
        repl = '''    fail_loss_bps: float
    entry_policy: str
    entry_vwap_premium_bps: float
    samples: str
    expected_return_col: str
    metadata_path: str
    ev_bps: float
'''
        text, _ = replace_once(text, marker, repl, "optimizer Candidate fields")

    extract_pat = re.compile(
        r'        target_hit_bps = as_float\(get_row_field\(row, "target_hit_bps".*?\n'
        r'(?:        entry_policy = .*?\n        metadata_path = .*?\n)?'
        r'        pred_prob = parse_rate_decimal\(get_row_field\(row, "pred_prob".*?\n',
        re.S,
    )
    extract_new = '''        target_hit_bps = as_float(get_row_field(row, "target_hit_bps", 80.0 if "80" in label_mode else 50.0), 80.0)
        entry_policy = as_text(get_row_field(row, "entry_policy", ""))
        entry_vwap_premium_bps = as_float(get_row_field(row, "entry_vwap_premium_bps", 50.0), 50.0)
        samples = as_text(get_row_field(row, "samples", ""))
        expected_return_col = as_text(get_row_field(row, "expected_return_col", ""))
        metadata_path = as_text(get_row_field(row, "metadata_path", ""))
        pred_prob = parse_rate_decimal(get_row_field(row, "pred_prob", get_row_field(row, "win_rate", cfg["default_hit_prob"])), cfg["default_hit_prob"])
'''
    if 'entry_policy = as_text(get_row_field(row, "entry_policy"' not in text:
        text, n = extract_pat.subn(extract_new, text, count=1)
        if n == 0:
            print("[WARN] optimizer extraction block not patched")

    construct_marker = '''            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps, ev_bps=ev_bps,
            quality_weight=quality, tier=tier, tier_multiplier=tier_mult, vol_daily=vol_daily,
'''
    construct_new = '''            target_hit_bps=target_hit_bps, fail_loss_bps=fail_loss_bps,
            entry_policy=entry_policy, entry_vwap_premium_bps=entry_vwap_premium_bps,
            samples=samples, expected_return_col=expected_return_col, metadata_path=metadata_path,
            ev_bps=ev_bps,
            quality_weight=quality, tier=tier, tier_multiplier=tier_mult, vol_daily=vol_daily,
'''
    if "entry_policy=entry_policy" not in text:
        text, _ = replace_once(text, construct_marker, construct_new, "optimizer Candidate construction")

    if text != original:
        OPTIMIZER.write_text(text, encoding="utf-8")
        print(f"[PATCHED] {OPTIMIZER}")
    else:
        print(f"[UNCHANGED] {OPTIMIZER}")


def main() -> int:
    patch_backtest()
    patch_adapter()
    patch_optimizer()
    print("[OK] complete portfolio fix patch applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
