#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python3}"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="saved_data/patch_backups/daily_flow_readme_patch_${TS}"
mkdir -p "$BACKUP_DIR"

FILES=(
  "pipelines/run_intraday_nextday_signals.py"
  "scripts/run_trading_day_signal_and_portfolio_all_models.sh"
  "README.md"
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f" >&2
    exit 2
  fi
  mkdir -p "$BACKUP_DIR/$(dirname "$f")"
  cp -p "$f" "$BACKUP_DIR/$f"
  echo "[BACKUP] $f -> $BACKUP_DIR/$f"
done

"$PYTHON" - <<'PY'
from pathlib import Path
import re

changed = []

def write_if_changed(path: str, new_text: str) -> None:
    p = Path(path)
    old = p.read_text(encoding="utf-8")
    if old != new_text:
        p.write_text(new_text, encoding="utf-8")
        changed.append(path)

# 1. run_intraday_nextday_signals.py default source
p = Path("pipelines/run_intraday_nextday_signals.py")
txt = p.read_text(encoding="utf-8")
old = 'p.add_argument("--spot-source-priority", default="sina,ths,em,xq")'
new = 'p.add_argument("--spot-source-priority", default="sina_batch,ths_etf,xq")'
if old in txt:
    txt = txt.replace(old, new, 1)
elif new in txt:
    pass
else:
    pat = r'(p\.add_argument\(\s*"--spot-source-priority"\s*,\s*default=)"sina,ths,em,xq"'
    txt2, n = re.subn(pat, r'\1"sina_batch,ths_etf,xq"', txt, count=1)
    if n == 0 and 'default="sina_batch,ths_etf,xq"' not in txt:
        raise SystemExit("[ERROR] could not find --spot-source-priority default")
    txt = txt2
write_if_changed(str(p), txt)

# 2. wrapper passes --date
p = Path("scripts/run_trading_day_signal_and_portfolio_all_models.sh")
txt = p.read_text(encoding="utf-8")
if '--date "$DATE_COMPACT"' not in txt:
    anchor = '"$PYTHON" pipelines/run_trading_day_signal_pipeline.py \\\n  --watchlist "$WATCHLIST" \\'
    repl = '"$PYTHON" pipelines/run_trading_day_signal_pipeline.py \\\n  --date "$DATE_COMPACT" \\\n  --watchlist "$WATCHLIST" \\'
    if anchor not in txt:
        raise SystemExit("[ERROR] could not find wrapper pipeline invocation anchor")
    txt = txt.replace(anchor, repl, 1)
write_if_changed(str(p), txt)

# 3. README updates
p = Path("README.md")
txt = p.read_text(encoding="utf-8")
txt = txt.replace("--spot-source-priority sina,ths,em,xq", "--spot-source-priority sina_batch,ths_etf,xq")

section_marker = "### 0. 推荐一键入口：信号 + 组合确认"
if section_marker not in txt:
    block = '''```text
盘前：只更新历史数据、样本、基本面/板块/外部历史特征；不做实时信号。
盘中：提前采集实时快照和实时上下文；14:55 前只做快速 score。
14:55：不再联网做慢采集，不再 build-bars，只读取 cutoff 前缓存输出 buy_signals.csv。
```'''
    insert = block + '''

### 0. 推荐一键入口：信号 + 组合确认

每天实盘优先使用这个入口，它会串联：

```text
实时股票快照采集
实时板块/外部上下文采集
5min bar 构建与 OHLCV 修正
score-now 生成 all_scores / buy_signals / rejected_scores
portfolio optimizer 生成最终组合订单
```

推荐命令：

```bash
PYTHON=python3 bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

指定交易日回放/补跑时同时指定紧凑日期和横线日期：

```bash
DATE_COMPACT=20260515 DATE_DASH=2026-05-15 PYTHON=python3 \\
bash scripts/run_trading_day_signal_and_portfolio_all_models.sh
```

该入口内部会把 `DATE_COMPACT` 传给信号流水线的 `--date`，并把 `DATE_DASH` 传给组合确认模块，避免信号目录日期和组合报告日期错位。

默认实时股票源为：

```text
sina_batch,ths_etf,xq
```

含义：

```text
sina_batch：A 股/ETF 小批量目标代码实时源，默认主源
ths_etf：THS ETF 表，只用于 ETF 补充
xq：雪球慢速补洞源，只补仍缺核心字段的少量标的
```

不建议在 14:55 主流程中使用 `em` 或全市场接口。`em` 在实时路径中被显式禁用，`em_full` 只适合人工诊断，不适合作为临近收盘主流程数据源。
'''
    if block in txt:
        txt = txt.replace(block, insert, 1)
    else:
        header = "## 交易日 14:55 实盘信号流水线"
        if header not in txt:
            raise SystemExit("[ERROR] README trading-day section not found")
        txt = txt.replace(header, header + "\n\n" + insert, 1)

note_marker = "### 每日流程关键检查点"
if note_marker not in txt:
    txt = txt.rstrip() + '''

### 每日流程关键检查点

1. `scripts/run_trading_day_signal_and_portfolio_all_models.sh` 是推荐的一键入口。
2. `pipelines/run_intraday_nextday_signals.py` 的默认 `--spot-source-priority` 已对齐为 `sina_batch,ths_etf,xq`。
3. 历史补跑时必须同时设置：
   - `DATE_COMPACT=YYYYMMDD`
   - `DATE_DASH=YYYY-MM-DD`
4. 信号输出目录：
   - `saved_data/intraday_nextday_signals/YYYYMMDD/all_scores.csv`
   - `saved_data/intraday_nextday_signals/YYYYMMDD/buy_signals.csv`
   - `saved_data/intraday_nextday_signals/YYYYMMDD/rejected_scores.csv`
5. 组合输出目录：
   - `portfolio_reports/daily_portfolio_orders_YYYY-MM-DD.csv`
   - `portfolio_reports/daily_portfolio_selected_YYYY-MM-DD.csv`
   - `portfolio_reports/daily_portfolio_rejected_YYYY-MM-DD.csv`
   - `portfolio_reports/daily_portfolio_report_YYYY-MM-DD.json`
6. 清理模型库不是每日交易流程的一部分。`cleanup-apply` 只应在检查 `cleanup-preview` 报告后单独执行。
''' + "\n"
write_if_changed(str(p), txt)

# Validate changed files
intraday = Path("pipelines/run_intraday_nextday_signals.py").read_text(encoding="utf-8")
wrapper = Path("scripts/run_trading_day_signal_and_portfolio_all_models.sh").read_text(encoding="utf-8")
readme = Path("README.md").read_text(encoding="utf-8")
assert 'default="sina_batch,ths_etf,xq"' in intraday
assert '--date "$DATE_COMPACT"' in wrapper
assert "推荐一键入口：信号 + 组合确认" in readme
assert "sina_batch,ths_etf,xq" in readme
print("[OK] changed_files=" + ",".join(changed) if changed else "[OK] already up to date")
PY

echo "[VALIDATE] bash -n scripts/run_trading_day_signal_and_portfolio_all_models.sh"
bash -n scripts/run_trading_day_signal_and_portfolio_all_models.sh

echo "[VALIDATE] python compile pipelines/run_intraday_nextday_signals.py"
"$PYTHON" -m py_compile pipelines/run_intraday_nextday_signals.py

echo "[DONE] patch applied"
echo "[BACKUP_DIR] $BACKUP_DIR"
