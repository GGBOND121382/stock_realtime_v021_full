#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if new in text:
        return text
    raise SystemExit(f"[ERROR] cannot find block for {label}")


def patch_adapter() -> None:
    p = Path("portfolio_decision/portfolio_confirm_from_buy_signals.py")
    txt = p.read_text(encoding="utf-8")

    if "import re\n" not in txt:
        txt = txt.replace("import json\n", "import json\nimport re\n", 1)

    if "import tomllib" not in txt and "import tomli as tomllib" not in txt:
        marker = "import pandas as pd\n"
        inj = '''import pandas as pd

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore
'''
        txt = replace_exact(txt, marker, inj, "tomllib import")

    helpers = r'''
def _is_unknown_sector(x: Any) -> bool:
    s = str(x or "").strip()
    return (not s) or s.upper() in {"UNKNOWN", "NAN", "NONE", "NULL", "无", "-"}


def _truthy(x: Any, default: bool = True) -> bool:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip().lower()
    if s == "":
        return default
    if s in {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


def _float_or_nan(x: Any) -> float:
    return as_float(x, np.nan)


def load_context_sector_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    stocks = cfg.get("stocks", {}) if isinstance(cfg, dict) else {}
    out: Dict[str, str] = {}
    if not isinstance(stocks, dict):
        return out
    for code, entry in stocks.items():
        if not isinstance(entry, dict):
            continue
        sectors = entry.get("sector_symbols") or entry.get("sectors") or []
        if isinstance(sectors, str):
            sector = sectors.strip()
        elif isinstance(sectors, list) and sectors:
            sector = str(sectors[0]).strip()
        else:
            sector = ""
        if sector:
            out[normalize_stock_code(code)] = sector
    return out


def load_rule_rows(path: Optional[Path]) -> list[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        df = pd.read_csv(path, comment="#")
    except pd.errors.EmptyDataError:
        return []
    except Exception as exc:
        print(f"[WARN] failed to read rule file {path}: {type(exc).__name__}: {exc}")
        return []
    if df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        row = {str(k): v for k, v in r.to_dict().items()}
        if not any(str(v).strip() for v in row.values() if v is not None and not pd.isna(v)):
            continue
        rows.append(row)
    return rows


def _rule_matches(row: Dict[str, Any], stock_code: str, artifact: str) -> bool:
    stock_pat = str(row.get("stock_code", row.get("symbol", "*")) or "*").strip()
    if stock_pat and stock_pat != "*":
        allowed = {normalize_stock_code(x.strip()) for x in stock_pat.replace(";", ",").split(",") if x.strip()}
        if normalize_stock_code(stock_code) not in allowed:
            return False

    pat = str(row.get("artifact_pattern", row.get("artifact_name", row.get("model_name", "*"))) or "*").strip()
    if not pat or pat == "*":
        return True
    if pat == artifact:
        return True
    try:
        return re.search(pat, artifact) is not None
    except re.error:
        return pat in artifact


def merged_matching_rules(rows: list[Dict[str, Any]], stock_code: str, artifact: str) -> Dict[str, Any]:
    # Later matching rows override earlier rows.  This allows a broad stock rule
    # followed by a more specific artifact rule in the CSV.
    out: Dict[str, Any] = {}
    for row in rows:
        if _rule_matches(row, stock_code, artifact):
            out.update({k: v for k, v in row.items() if k and not (isinstance(v, float) and pd.isna(v))})
    return out


def choose_sector(row: pd.Series, stock_code: str, context_sector_map: Dict[str, str]) -> tuple[str, str]:
    for col in ["sector", "industry", "sector_symbol"]:
        if col in row.index and not _is_unknown_sector(row.get(col)):
            return str(row.get(col)).strip(), f"signal:{col}"
    sector = context_sector_map.get(normalize_stock_code(stock_code))
    if sector:
        return sector, "context_config"
    return "UNKNOWN", "missing"


def apply_override_fields(
    stock_code: str,
    artifact: str,
    override_rows: list[Dict[str, Any]],
    recent_rows: list[Dict[str, Any]],
) -> Dict[str, Any]:
    ov = merged_matching_rules(override_rows, stock_code, artifact)
    rp = merged_matching_rules(recent_rows, stock_code, artifact)

    enabled = _truthy(ov.get("enabled", 1), True) and _truthy(rp.get("enabled", rp.get("recent_enabled", 1)), True)
    weight_multiplier = as_float(ov.get("weight_multiplier", 1.0), 1.0) * as_float(
        rp.get("weight_multiplier", rp.get("recent_weight_multiplier", 1.0)), 1.0
    )

    out = {
        "enabled": int(bool(enabled)),
        "weight_multiplier": float(weight_multiplier),
        "max_weight_override": _float_or_nan(ov.get("max_weight_override", ov.get("max_weight", np.nan))),
        "max_add_weight_override": _float_or_nan(ov.get("max_add_weight_override", ov.get("max_add_weight", np.nan))),
        "model_override_reason": str(ov.get("notes", ov.get("reason", "")) or ""),
        "recent_perf_note": str(rp.get("notes", rp.get("reason", "")) or ""),
    }
    for k in ["recent_trades", "recent_win_rate", "recent_profit_factor", "recent_pnl", "recent_avg_return"]:
        if k in rp:
            out[k] = rp.get(k)
    return out

'''
    if "def load_context_sector_map" not in txt:
        txt = txt.replace("def metric_bps_from_return", helpers + "\ndef metric_bps_from_return", 1)

    old_sig = "def build_inputs(signal_dir: Path, saved_models: Path, out_input_dir: Path, use_all_scores: bool = False) -> Dict[str, Path]:"
    new_sig = '''def build_inputs(
    signal_dir: Path,
    saved_models: Path,
    out_input_dir: Path,
    use_all_scores: bool = False,
    context_config: Optional[Path] = None,
    model_overrides: Optional[Path] = None,
    recent_perf: Optional[Path] = None,
) -> Dict[str, Path]:'''
    txt = replace_exact(txt, old_sig, new_sig, "build_inputs signature")

    old = '''    raw = pd.read_csv(src)
    out_input_dir.mkdir(parents=True, exist_ok=True)

    signals_out = out_input_dir / "portfolio_signals.csv"
'''
    new = '''    raw = pd.read_csv(src)
    out_input_dir.mkdir(parents=True, exist_ok=True)

    context_sector_map = load_context_sector_map(context_config)
    override_rows = load_rule_rows(model_overrides)
    recent_rows = load_rule_rows(recent_perf)

    signals_out = out_input_dir / "portfolio_signals.csv"
'''
    txt = replace_exact(txt, old, new, "load sector and override inputs")

    old = '''        sector = r.get("sector", r.get("industry", "UNKNOWN"))

        sig_rows.append({
'''
    new = '''        sector, sector_source = choose_sector(r, stock_code, context_sector_map)
        override_fields = apply_override_fields(stock_code, artifact, override_rows, recent_rows)

        sig_rows.append({
'''
    txt = replace_exact(txt, old, new, "sector and override selection")

    old = '''            "metadata_path": str(meta_path) if meta_path else "",
        })
'''
    new = '''            "metadata_path": str(meta_path) if meta_path else "",
            "sector_source": sector_source,
            **override_fields,
        })
'''
    txt = replace_exact(txt, old, new, "signal override columns")

    old = '''            "sector": sector,
        })
'''
    new = '''            "sector": sector,
            "sector_source": sector_source,
            **override_fields,
        })
'''
    txt = replace_exact(txt, old, new, "metric override columns")

    old = '''    ap.add_argument("--config", default="configs/portfolio_confirm_config.json")
    ap.add_argument("--out-dir", default="portfolio_reports")
'''
    new = '''    ap.add_argument("--config", default="configs/portfolio_confirm_config.json")
    ap.add_argument("--context-config", default="configs/realtime_context_sources.toml")
    ap.add_argument("--model-overrides", default="configs/portfolio_model_overrides.csv")
    ap.add_argument("--recent-perf", default=None, help="Optional recent model performance CSV used as an additional weight/enable rule table.")
    ap.add_argument("--out-dir", default="portfolio_reports")
'''
    txt = replace_exact(txt, old, new, "adapter CLI args")

    old = '''        out_input_dir=input_dir,
        use_all_scores=bool(args.use_all_scores),
    )
'''
    new = '''        out_input_dir=input_dir,
        use_all_scores=bool(args.use_all_scores),
        context_config=Path(args.context_config) if args.context_config else None,
        model_overrides=Path(args.model_overrides) if args.model_overrides else None,
        recent_perf=Path(args.recent_perf) if args.recent_perf else None,
    )
'''
    txt = replace_exact(txt, old, new, "build_inputs call")

    p.write_text(txt, encoding="utf-8")
    print("[PATCHED]", p)


def patch_optimizer() -> None:
    p = Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py")
    txt = p.read_text(encoding="utf-8")

    txt = txt.replace('"max_positions": 3,', '"max_positions": 7,')
    txt = txt.replace('"max_positions": 10,', '"max_positions": 7,')
    if '"max_policy_weight": 0.15,' not in txt:
        txt = txt.replace(
            '"max_daily_buy_pct_of_cash": 0.70,\n',
            '"max_daily_buy_pct_of_cash": 0.70,\n    "max_policy_weight": 0.15,\n',
            1,
        )

    old = '''    median_return_bps: float
    utility_bps: float
    current_value: float
'''
    new = '''    median_return_bps: float
    utility_bps: float
    enabled: bool
    weight_multiplier: float
    override_reason: str
    current_value: float
'''
    txt = replace_exact(txt, old, new, "Candidate dataclass override fields")

    if "def parse_bool_flag" not in txt:
        old = "def parse_drawdown_abs(x: Any, default: float = 0.12) -> float:\n"
        new = '''def parse_bool_flag(x: Any, default: bool = True) -> bool:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip().lower()
    if s == "":
        return default
    if s in {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", "disable", "disabled"}:
        return False
    return default


''' + old
        txt = replace_exact(txt, old, new, "parse_bool_flag")

    old = '''    if model_type == "hit":
        max_weight = min(max_weight, 0.10)
        max_add_weight = min(max_add_weight, 0.08)
    if model_type == "observation":
        max_weight = min(max_weight, 0.05)
        max_add_weight = min(max_add_weight, 0.05)
    return tier, max_weight, max_add_weight
'''
    new = '''    max_policy_weight = cfg.get("max_policy_weight")
    if max_policy_weight is not None:
        max_weight = min(max_weight, as_float(max_policy_weight, max_weight))

    if model_type == "hit":
        max_weight = min(max_weight, 0.10)
        max_add_weight = min(max_add_weight, 0.08)
    if model_type == "observation":
        max_weight = min(max_weight, 0.05)
        max_add_weight = min(max_add_weight, 0.05)
    return tier, max_weight, max_add_weight
'''
    if 'max_policy_weight = cfg.get("max_policy_weight")' not in txt:
        txt = replace_exact(txt, old, new, "max_policy_weight in get_tier_and_caps")

    old = '''        if not np.isfinite(price) or price <= 0:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "missing_or_invalid_price"})
            continue

        model_type = infer_model_type(label_mode, row, cfg, code)
'''
    new = '''        if not np.isfinite(price) or price <= 0:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "missing_or_invalid_price"})
            continue

        enabled = parse_bool_flag(get_row_field(row, "enabled", 1), True)
        weight_multiplier = as_float(get_row_field(row, "weight_multiplier", 1.0), 1.0)
        override_reason = str(get_row_field(row, "model_override_reason", get_row_field(row, "recent_perf_note", "")) or "")
        if not enabled:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "disabled_by_override", "override_reason": override_reason})
            continue
        if weight_multiplier <= 0:
            rejected.append({"stock_code": code, "model_name": model_name, "reason": "non_positive_weight_multiplier", "weight_multiplier": weight_multiplier})
            continue

        model_type = infer_model_type(label_mode, row, cfg, code)
'''
    txt = replace_exact(txt, old, new, "candidate enabled and multiplier")

    old = '''        quality = calc_quality_weight(row, cfg)
        tier, max_weight, max_add_weight = get_tier_and_caps(row, code, model_type, cfg)
        tier_mult = as_float(cfg.get("tier_multiplier", {}).get(str(tier), 1.0), 1.0)

        vol_daily = float(vol_map.get(code, np.nan))
'''
    new = '''        quality = calc_quality_weight(row, cfg)
        tier, max_weight, max_add_weight = get_tier_and_caps(row, code, model_type, cfg)
        max_weight_override = as_float(get_row_field(row, "max_weight_override", np.nan), np.nan)
        max_add_weight_override = as_float(get_row_field(row, "max_add_weight_override", np.nan), np.nan)
        if np.isfinite(max_weight_override) and max_weight_override > 0:
            max_weight = min(float(max_weight_override), as_float(cfg.get("max_policy_weight", max_weight), max_weight))
        if np.isfinite(max_add_weight_override) and max_add_weight_override > 0:
            max_add_weight = float(max_add_weight_override)
        tier_mult = as_float(cfg.get("tier_multiplier", {}).get(str(tier), 1.0), 1.0)

        vol_daily = float(vol_map.get(code, np.nan))
'''
    txt = replace_exact(txt, old, new, "cap overrides")

    old = '''        utility_bps = (
            ev_bps * quality * tier_mult
            - float(cfg.get("vol_penalty_lambda", 0.02)) * vol_bps
            - float(cfg.get("dd_penalty_lambda", 0.02)) * dd_bps
        )
        if utility_bps <= float(cfg.get("min_utility_bps", 0.0)):
'''
    new = '''        utility_bps = (
            ev_bps * quality * tier_mult
            - float(cfg.get("vol_penalty_lambda", 0.02)) * vol_bps
            - float(cfg.get("dd_penalty_lambda", 0.02)) * dd_bps
        )
        utility_bps *= weight_multiplier
        if utility_bps <= float(cfg.get("min_utility_bps", 0.0)):
'''
    txt = replace_exact(txt, old, new, "utility multiplier")

    old = '''            profit_factor=pf, trades=trades, median_return_bps=median_bps,
            utility_bps=utility_bps, current_value=current_value, current_shares=current_shares,
'''
    new = '''            profit_factor=pf, trades=trades, median_return_bps=median_bps,
            utility_bps=utility_bps, enabled=True, weight_multiplier=weight_multiplier,
            override_reason=override_reason, current_value=current_value, current_shares=current_shares,
'''
    txt = replace_exact(txt, old, new, "Candidate instantiation")

    p.write_text(txt, encoding="utf-8")
    print("[PATCHED]", p)


def patch_wrapper() -> None:
    p = Path("scripts/run_portfolio_confirm_from_signals.sh")
    txt = p.read_text(encoding="utf-8")
    old = '''CONFIG="${CONFIG:-configs/portfolio_confirm_config.json}"
OUT_DIR="${OUT_DIR:-portfolio_reports}"
'''
    new = '''CONFIG="${CONFIG:-configs/portfolio_confirm_config.json}"
CONTEXT_CONFIG="${CONTEXT_CONFIG:-configs/realtime_context_sources.toml}"
MODEL_OVERRIDES="${MODEL_OVERRIDES:-configs/portfolio_model_overrides.csv}"
RECENT_PERF="${RECENT_PERF:-}"
OUT_DIR="${OUT_DIR:-portfolio_reports}"
'''
    txt = replace_exact(txt, old, new, "portfolio wrapper variables")

    old = '''if [[ -f "$HISTORY" ]]; then
  CMD+=(--history "$HISTORY")
else
  echo "[WARN] history file not found: $HISTORY; risk model will use conservative fallbacks."
fi

CMD+=("${EXTRA_ARGS[@]}")
'''
    new = '''if [[ -f "$HISTORY" ]]; then
  CMD+=(--history "$HISTORY")
else
  echo "[WARN] history file not found: $HISTORY; risk model will use conservative fallbacks."
fi

if [[ -f "$CONTEXT_CONFIG" ]]; then
  CMD+=(--context-config "$CONTEXT_CONFIG")
fi

if [[ -f "$MODEL_OVERRIDES" ]]; then
  CMD+=(--model-overrides "$MODEL_OVERRIDES")
fi

if [[ -n "$RECENT_PERF" && -f "$RECENT_PERF" ]]; then
  CMD+=(--recent-perf "$RECENT_PERF")
fi

CMD+=("${EXTRA_ARGS[@]}")
'''
    txt = replace_exact(txt, old, new, "portfolio wrapper adapter flags")
    p.write_text(txt, encoding="utf-8")
    print("[PATCHED]", p)


def patch_config() -> None:
    p = Path("configs/portfolio_confirm_config.json")
    cfg = {}
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["max_policy_weight"] = 0.15
    cfg["max_positions"] = 7
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[PATCHED]", p)


def patch_backtest() -> None:
    p = Path("portfolio_decision/backtest_historical_score_portfolio.py")
    if not p.exists():
        print("[SKIP] no backtest_historical_score_portfolio.py")
        return
    txt = p.read_text(encoding="utf-8")
    perf_func = r'''
def _profit_factor_from_pnl(pnl: pd.Series) -> float:
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    profit = float(pnl[pnl > 0].sum())
    loss = float(-pnl[pnl < 0].sum())
    if loss <= 0:
        return float("inf") if profit > 0 else 0.0
    return profit / loss


def write_perf_summaries(trades_df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    paths = {
        "model_perf": out_dir / "model_perf_summary.csv",
        "stock_perf": out_dir / "stock_perf_summary.csv",
        "suggested_recent_perf": out_dir / "suggested_portfolio_model_recent_perf.csv",
    }
    if trades_df.empty:
        for path in paths.values():
            pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
        return {k: str(v) for k, v in paths.items()}

    df = trades_df.copy()
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")

    def summarize_group(g: pd.DataFrame) -> pd.Series:
        pnl = pd.to_numeric(g["pnl"], errors="coerce")
        ret = pd.to_numeric(g["net_return"], errors="coerce")
        return pd.Series({
            "trades": int(len(g)),
            "pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()) if len(g) else 0.0,
            "profit_factor": _profit_factor_from_pnl(pnl),
            "avg_return": float(ret.mean()) if len(g) else 0.0,
            "median_return": float(ret.median()) if len(g) else 0.0,
            "max_loss": float(pnl.min()) if len(g) else 0.0,
        })

    model_cols = [c for c in ["stock_code", "model_name"] if c in df.columns]
    model_perf = df.groupby(model_cols, dropna=False).apply(summarize_group).reset_index() if model_cols else pd.DataFrame()
    stock_perf = df.groupby(["stock_code"], dropna=False).apply(summarize_group).reset_index() if "stock_code" in df.columns else pd.DataFrame()

    suggested = model_perf.copy()
    if not suggested.empty:
        def mult(row):
            trades = float(row.get("trades", 0))
            pf = float(row.get("profit_factor", 0))
            pnl = float(row.get("pnl", 0))
            if trades >= 10 and (pf < 0.9 or pnl < 0):
                return 0.30
            if trades >= 10 and pf < 1.1:
                return 0.60
            if trades >= 20 and pf > 1.5 and pnl > 0:
                return 1.10
            return 1.00
        suggested["artifact_pattern"] = suggested.get("model_name", "")
        suggested["enabled"] = 1
        suggested["weight_multiplier"] = suggested.apply(mult, axis=1)
        suggested["notes"] = "generated_from_backtest_perf; review before using as RECENT_PERF"
        suggested = suggested[["stock_code", "artifact_pattern", "enabled", "weight_multiplier", "trades", "pnl", "win_rate", "profit_factor", "avg_return", "median_return", "notes"]]

    model_perf.to_csv(paths["model_perf"], index=False, encoding="utf-8-sig")
    stock_perf.to_csv(paths["stock_perf"], index=False, encoding="utf-8-sig")
    suggested.to_csv(paths["suggested_recent_perf"], index=False, encoding="utf-8-sig")
    return {k: str(v) for k, v in paths.items()}

'''
    if "def write_perf_summaries" not in txt:
        marker = "def read_watchlist("
        if marker not in txt:
            print("[WARN] cannot patch backtest perf summaries: read_watchlist marker not found")
            return
        txt = txt.replace(marker, perf_func + "\n" + marker, 1)

    old = '''    summary = sim["summary"]
    summary.update({
'''
    new = '''    perf_summary_paths = write_perf_summaries(sim["trades"], out_dir)
    summary = sim["summary"]
    summary.update({
        "perf_summary_paths": perf_summary_paths,
'''
    if "perf_summary_paths = write_perf_summaries" not in txt and old in txt:
        txt = txt.replace(old, new, 1)
    p.write_text(txt, encoding="utf-8")
    print("[PATCHED]", p)


def main() -> int:
    patch_adapter()
    patch_optimizer()
    patch_wrapper()
    patch_config()
    patch_backtest()
    print("[OK] all-in-one portfolio patch applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
