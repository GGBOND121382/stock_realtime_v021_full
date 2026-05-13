#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "[1/4] backup files ..."
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p backups/ane_live_board_v2_${TS}

cp feature_building/build_stock_external_features.py "backups/ane_live_board_v2_${TS}/build_stock_external_features.py"
cp configs/realtime_context_sources.toml "backups/ane_live_board_v2_${TS}/realtime_context_sources.toml"

echo "[2/4] patch training profile + realtime context config ..."

"$PYTHON" - <<'PY'
from pathlib import Path

# 1) 训练侧 external profile
p1 = Path("feature_building/build_stock_external_features.py")
s1 = p1.read_text(encoding="utf-8")

old1 = 'boards=("国防军工", "航空装备", "通用设备", "专用设备"),'
new1 = 'boards=("军工装备", "军工电子", "通用设备", "专用设备"),'

if old1 not in s1 and new1 not in s1:
    raise SystemExit("[ERROR] cannot find old/new board tuple in build_stock_external_features.py")

if old1 in s1:
    s1 = s1.replace(old1, new1)

p1.write_text(s1, encoding="utf-8")


# 2) 实时侧 context 配置
p2 = Path("configs/realtime_context_sources.toml")
s2 = p2.read_text(encoding="utf-8")

old2 = 'symbols = ["国防军工", "航空装备", "通用设备", "专用设备"]'
new2 = 'symbols = ["军工装备", "军工电子", "通用设备", "专用设备"]'

if old2 not in s2 and new2 not in s2:
    raise SystemExit("[ERROR] cannot find old/new ane_boards symbols in realtime_context_sources.toml")

if old2 in s2:
    s2 = s2.replace(old2, new2)

# 加一行说明，避免以后又改回去
marker = "# 603308 live-board-v2: use THS realtime available boards; old 国防军工/航空装备 were unavailable."
if marker not in s2:
    s2 = s2.replace(
        "[contexts.ane_boards]\n",
        marker + "\n[contexts.ane_boards]\n",
    )

p2.write_text(s2, encoding="utf-8")

print("[OK] patched files")
PY

echo "[3/4] verify patch ..."
grep -n 'boards=.*军工装备.*军工电子.*通用设备.*专用设备' feature_building/build_stock_external_features.py
grep -n 'symbols = \["军工装备", "军工电子", "通用设备", "专用设备"\]' configs/realtime_context_sources.toml

echo "[4/4] done."
echo
echo "Backup saved at: backups/ane_live_board_v2_${TS}"
echo
echo "Next: rebuild 603308 external features and retrain a NEW artifact, do not overwrite old model."
