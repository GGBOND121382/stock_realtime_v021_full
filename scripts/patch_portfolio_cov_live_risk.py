#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path

OPTIMIZER = Path("portfolio_decision/daily_portfolio_confirm_pyscipopt.py")
WRAPPER = Path("scripts/run_portfolio_confirm_from_signals.sh")
BACKTEST = Path("portfolio_decision/backtest_historical_score_portfolio.py")


LINEAR_COV_BLOCK = '''
    if bool(cfg.get("use_covariance_penalty", False)) and not cov_matrix.empty:
        # PySCIPOpt setObjective() in this environment rejects nonlinear
        # objectives such as amount_i times amount_j. Use a linear marginal
        # covariance-risk proxy so the model remains MILP-compatible.
        risk_aversion = float(cfg.get("cov_risk_aversion", 3.0))
        self_weight = float(cfg.get("cov_linear_self_weight", 0.05))

        current_weights = {}
        for h_code, h in holdings.items():
            code = normalize_stock_code(h_code)
            mv = as_float(h.get("market_value", 0.0), 0.0)
            if mv > 0 and total_asset > 0:
                current_weights[code] = mv / total_asset

        cov_linear_penalty_bps = {}
        for c in candidates:
            code = c.stock_code
            if code not in cov_matrix.index:
                continue

            current_cov = 0.0
            for h_code, w_h in current_weights.items():
                if h_code in cov_matrix.columns:
                    current_cov += float(cov_matrix.loc[code, h_code]) * float(w_h)

            var_i = float(cov_matrix.loc[code, code]) if code in cov_matrix.columns else 0.0
            marginal_variance = max(0.0, 2.0 * current_cov + self_weight * var_i)
            cov_linear_penalty_bps[code] = 10000.0 * risk_aversion * marginal_variance

        if cov_linear_penalty_bps:
            obj = obj - quicksum(
                (cov_linear_penalty_bps.get(c.stock_code, 0.0) / 10000.0) * amount[c.stock_code]
                for c in candidates
            )
'''


BACKTEST_HELPER = r'''
def write_point_in_time_risk_history(
    history: pd.DataFrame,
    date: pd.Timestamp,
    out_dir: Path,
    include_current_day: bool = True,
    min_rows: int = 20,
) -> Optional[Path]:
    # Intermediate risk input for the optimizer subprocess.
    # The file is clipped to the current backtest date to avoid lookahead.
    if history is None or history.empty:
        return None

    cutoff = pd.Timestamp(date).normalize()
    hist = history.copy()
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    hist = hist[hist.index.notna()].sort_index()
    hist = hist.loc[hist.index <= cutoff] if include_current_day else hist.loc[hist.index < cutoff]
    hist = hist.dropna(axis=1, how="all")
    if len(hist) < int(min_rows) or hist.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"risk_history_until_{cutoff.strftime('%Y%m%d')}.csv"
    export = hist.reset_index()
    if export.columns[0] != "date":
        export = export.rename(columns={export.columns[0]: "date"})
    export["date"] = pd.to_datetime(export["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    export.to_csv(path, index=False, encoding="utf-8-sig")
    return path

'''


def patch_optimizer() -> None:
    if not OPTIMIZER.exists():
        raise SystemExit(f"[ERROR] missing {OPTIMIZER}")
    txt = OPTIMIZER.read_text(encoding="utf-8")

    if '"covariance_penalty_mode"' not in txt:
        if '"cov_risk_aversion": 3.0,' in txt:
            txt = txt.replace(
                '"cov_risk_aversion": 3.0,\n',
                '"cov_risk_aversion": 3.0,\n'
                '    "covariance_penalty_mode": "linear",\n'
                '    "cov_linear_self_weight": 0.05,\n',
                1,
            )
        else:
            print("[WARN] cannot find cov_risk_aversion default marker")

    if "cov_linear_penalty_bps" not in txt:
        pattern = re.compile(
            r'\n    if\s+bool\(cfg\.get\("use_covariance_penalty",\s*False\)\)\s+and\s+not\s+cov_matrix\.empty:\n'
            r'(?:        .*\n)+?'
            r'(?=\n    m\.setObjective\(obj,\s*["\']maximize["\']\))'
        )
        m = pattern.search(txt)
        if not m:
            raise SystemExit("[ERROR] cannot find covariance penalty objective block")
        old = m.group(0)
        if "amount[i]" in old and "amount[j]" in old:
            txt = txt[:m.start()] + "\n" + LINEAR_COV_BLOCK.rstrip("\n") + txt[m.end():]
            print("[PATCHED] optimizer nonlinear covariance objective -> linear proxy")
        else:
            print("[WARN] covariance block found but does not look nonlinear; left unchanged")

    if '"covariance_penalty_mode": cfg.get("covariance_penalty_mode", "linear")' not in txt:
        old = '''        "use_covariance_penalty": bool(cfg.get("use_covariance_penalty", False)),
        "cov_risk_aversion": float(cfg.get("cov_risk_aversion", 3.0)),
'''
        new = '''        "use_covariance_penalty": bool(cfg.get("use_covariance_penalty", False)),
        "cov_risk_aversion": float(cfg.get("cov_risk_aversion", 3.0)),
        "covariance_penalty_mode": cfg.get("covariance_penalty_mode", "linear"),
        "cov_linear_self_weight": float(cfg.get("cov_linear_self_weight", 0.05)),
'''
        if old in txt:
            txt = txt.replace(old, new, 1)

    OPTIMIZER.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {OPTIMIZER}")


def insert_var_after(txt: str, marker: str, addition: str) -> str:
    first_line = addition.strip().split("\n")[0]
    if first_line in txt:
        return txt
    if marker not in txt:
        print(f"[WARN] marker not found for variable insertion: {marker.strip()}")
        return txt
    return txt.replace(marker, marker + addition, 1)


def patch_wrapper() -> None:
    if not WRAPPER.exists():
        raise SystemExit(f"[ERROR] missing {WRAPPER}")
    txt = WRAPPER.read_text(encoding="utf-8")

    txt = txt.replace('HISTORY="${HISTORY:-history_close.csv}"', 'HISTORY="${HISTORY:-}"')

    txt = insert_var_after(
        txt,
        'SAVED_MODELS="${SAVED_MODELS:-saved_models}"\n',
        'SAVED_DATA_DIR="${SAVED_DATA_DIR:-saved_data}"\n',
    )
    txt = insert_var_after(
        txt,
        'OUT_DIR="${OUT_DIR:-portfolio_reports}"\n',
        'AUTO_RISK_HISTORY="${AUTO_RISK_HISTORY:-1}"\nRISK_HISTORY_DIR="${RISK_HISTORY_DIR:-${OUT_DIR}/risk_history}"\n',
    )

    auto_block = r'''
if [[ "${AUTO_RISK_HISTORY:-1}" == "1" && ( -z "${HISTORY:-}" || ! -f "$HISTORY" ) ]]; then
  mkdir -p "$RISK_HISTORY_DIR"
  AUTO_HISTORY="${RISK_HISTORY_DIR}/risk_history_for_portfolio_${DATE_COMPACT}.csv"
  SIGNAL_FILE="$SIGNAL_DIR/buy_signals.csv"
  if [[ ! -f "$SIGNAL_FILE" ]]; then
    SIGNAL_FILE="$SIGNAL_DIR/all_scores.csv"
  fi

  echo "[RISK_HISTORY] building point-in-time live risk history: $AUTO_HISTORY"
  "$PYTHON" scripts/build_portfolio_risk_history.py \
    --saved-models "$SAVED_MODELS" \
    --saved-data-dir "$SAVED_DATA_DIR" \
    --signals "$SIGNAL_FILE" \
    --date "$DATE_DASH" \
    --out "$AUTO_HISTORY"

  if [[ -f "$AUTO_HISTORY" ]]; then
    HISTORY="$AUTO_HISTORY"
  fi
fi

'''
    if "scripts/build_portfolio_risk_history.py" not in txt:
        marker = "CMD=(\n"
        if marker not in txt:
            raise SystemExit("[ERROR] cannot find CMD=( marker in portfolio wrapper")
        txt = txt.replace(marker, auto_block + marker, 1)

    if 'CMD+=(--history "$HISTORY")' not in txt:
        marker = 'CMD+=("${EXTRA_ARGS[@]}")\n'
        history_block = '''if [[ -f "$HISTORY" ]]; then
  CMD+=(--history "$HISTORY")
else
  echo "[WARN] no risk history available; risk model will use conservative fallbacks."
fi

'''
        if marker in txt:
            txt = txt.replace(marker, history_block + marker, 1)

    WRAPPER.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] {WRAPPER}")


def patch_backtest_best_effort() -> None:
    if not BACKTEST.exists():
        print("[SKIP] backtest script not found")
        return
    txt = BACKTEST.read_text(encoding="utf-8")
    changed = False

    if "def write_point_in_time_risk_history(" not in txt:
        marker = "def run_portfolio_adapter("
        if marker in txt:
            txt = txt.replace(marker, BACKTEST_HELPER + "\n" + marker, 1)
            changed = True
        else:
            print("[WARN] cannot insert backtest risk-history helper")

    if "risk_history_path = write_point_in_time_risk_history(history, date, day_out)" not in txt:
        anchor = (
            '        account_path = day_out / f"sim_account_{ymd}.json"\n'
            '        account_path.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")\n'
        )
        if anchor in txt:
            txt = txt.replace(anchor, anchor + '\n        risk_history_path = write_point_in_time_risk_history(history, date, day_out)\n', 1)
            changed = True
        else:
            print("[WARN] cannot find account_path write block in backtest")

    if "history_path=risk_history_path" not in txt and "history_path = risk_history_path" not in txt:
        pattern = re.compile(
            r'(orders_path\s*=\s*run_portfolio_adapter\([\s\S]*?history_path\s*=\s*)history_path(,)',
            flags=re.MULTILINE,
        )
        if pattern.search(txt):
            txt = pattern.sub(r"\1risk_history_path\2", txt, count=1)
            changed = True
        else:
            print("[WARN] cannot switch backtest run_portfolio_adapter history_path argument")

    if changed:
        BACKTEST.write_text(txt, encoding="utf-8")
        print(f"[PATCHED] {BACKTEST}")
    else:
        print(f"[KEEP] {BACKTEST}")


def main() -> int:
    patch_optimizer()
    patch_wrapper()
    patch_backtest_best_effort()
    print("[OK] covariance linear fix + live risk history patch applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
