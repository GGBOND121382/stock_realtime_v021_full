#!/usr/bin/env bash
set -euo pipefail

# setup_stock_runtime_ubuntu_aliyun.sh
#
# Purpose:
#   Initialize a fresh Ubuntu ECS environment for the stock_realtime codebase.
#   - Use Aliyun apt mirror
#   - Use Aliyun PyPI mirror
#   - Install system packages
#   - Install Python packages without creating a virtualenv
#   - Verify imports
#
# Usage:
#   chmod +x setup_stock_runtime_ubuntu_aliyun.sh
#   ./setup_stock_runtime_ubuntu_aliyun.sh
#
# Options via env:
#   SKIP_APT_MIRROR=1      Do not rewrite apt sources
#   SKIP_APT_INSTALL=1     Do not install apt packages
#   SKIP_PIP_CONFIG=1      Do not configure pip mirror
#   SKIP_PIP_INSTALL=1     Do not install Python packages
#   USE_USER_SITE=0        Install Python packages globally instead of --user
#
# Notes:
#   - Designed for Ubuntu/Debian-like systems.
#   - Does NOT create a Python virtual environment.
#   - On newer Ubuntu versions, pip may require --break-system-packages.
#     This script detects and uses it when available.

log() {
  echo
  echo "============================================================"
  echo "$*"
  echo "============================================================"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

require_cmd bash
require_cmd uname

if [[ ! -f /etc/os-release ]]; then
  die "/etc/os-release not found. This script expects Ubuntu/Debian-like Linux."
fi

# shellcheck disable=SC1091
. /etc/os-release

if [[ "${ID:-}" != "ubuntu" && "${ID_LIKE:-}" != *"debian"* ]]; then
  warn "Detected ID=${ID:-unknown}, ID_LIKE=${ID_LIKE:-unknown}. This script is mainly for Ubuntu/Debian."
fi

CODENAME="${VERSION_CODENAME:-}"
if [[ -z "$CODENAME" ]]; then
  die "Cannot detect Ubuntu codename from /etc/os-release."
fi

log "Detected system"
echo "ID=${ID:-unknown}"
echo "VERSION=${VERSION:-unknown}"
echo "CODENAME=${CODENAME}"

if [[ "${SKIP_APT_MIRROR:-0}" != "1" ]]; then
  log "Configuring Aliyun apt mirror"

  TS="$(date +%Y%m%d_%H%M%S)"

  # Backup common apt source files.
  if [[ -f /etc/apt/sources.list ]]; then
    $SUDO cp /etc/apt/sources.list "/etc/apt/sources.list.bak_${TS}"
  fi

  if [[ -f /etc/apt/sources.list.d/ubuntu.sources ]]; then
    $SUDO cp /etc/apt/sources.list.d/ubuntu.sources "/etc/apt/sources.list.d/ubuntu.sources.bak_${TS}"
    # Disable the default deb822 source to avoid duplicate/conflicting entries.
    $SUDO mv /etc/apt/sources.list.d/ubuntu.sources "/etc/apt/sources.list.d/ubuntu.sources.disabled_${TS}"
  fi

  # Disable previous aliyun list if any, then create a clean one.
  if [[ -f /etc/apt/sources.list.d/aliyun-ubuntu.list ]]; then
    $SUDO cp /etc/apt/sources.list.d/aliyun-ubuntu.list "/etc/apt/sources.list.d/aliyun-ubuntu.list.bak_${TS}"
  fi

  $SUDO tee /etc/apt/sources.list.d/aliyun-ubuntu.list > /dev/null <<EOF
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME} main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME}-security main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME}-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ ${CODENAME}-backports main restricted universe multiverse
EOF

  # Keep /etc/apt/sources.list minimal to avoid duplicates.
  $SUDO tee /etc/apt/sources.list > /dev/null <<EOF
# Main Ubuntu source moved to /etc/apt/sources.list.d/aliyun-ubuntu.list
EOF

  $SUDO apt update
else
  log "Skipping apt mirror configuration"
fi

if [[ "${SKIP_APT_INSTALL:-0}" != "1" ]]; then
  log "Installing system packages"
  $SUDO apt update
  $SUDO apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    git \
    unzip \
    curl \
    wget \
    ca-certificates \
    libgomp1
else
  log "Skipping apt package installation"
fi

require_cmd python3

log "Python version"
python3 --version

# Make sure pip exists.
if ! python3 -m pip --version >/dev/null 2>&1; then
  log "pip not available; installing python3-pip"
  $SUDO apt update
  $SUDO apt install -y python3-pip
fi

log "pip version"
python3 -m pip --version

if [[ "${SKIP_PIP_CONFIG:-0}" != "1" ]]; then
  log "Configuring Aliyun PyPI mirror"
  python3 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
  python3 -m pip config set install.trusted-host mirrors.aliyun.com
else
  log "Skipping pip mirror configuration"
fi

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  log "Installing Python runtime packages"

  PIP_FLAGS=(-U)

  if [[ "${USE_USER_SITE:-1}" != "0" ]]; then
    PIP_FLAGS+=(--user)
  fi

  # Newer Ubuntu may require this when installing into system-managed Python.
  if python3 -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages"; then
    PIP_FLAGS+=(--break-system-packages)
  fi

  python3 -m pip install "${PIP_FLAGS[@]}" \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    joblib \
    matplotlib \
    openpyxl \
    baostock \
    akshare \
    py7zr \
    requests \
    yfinance
else
  log "Skipping Python package installation"
fi

log "Verifying Python imports"
python3 - <<'PY'
import sys

packages = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
    ("openpyxl", "openpyxl"),
    ("baostock", "baostock"),
    ("akshare", "akshare"),
    ("py7zr", "py7zr"),
    ("requests", "requests"),
    ("yfinance", "yfinance"),
]

failed = []
for module_name, display_name in packages:
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"OK: {display_name} {version}")
    except Exception as exc:
        failed.append((display_name, repr(exc)))
        print(f"FAIL: {display_name}: {exc}")

if failed:
    print("\nSome imports failed:")
    for name, err in failed:
        print(f"  - {name}: {err}")
    sys.exit(1)

print("\nOK: all runtime packages imported successfully")
PY

log "Optional project syntax check"
if [[ -d data_collection || -d feature_building || -d model_training || -d pipelines ]]; then
  python3 -m compileall -q \
    data_collection \
    feature_building \
    model_training \
    model_saving \
    prediction \
    pipelines \
    scripts 2>/dev/null || {
      warn "compileall failed. Check whether all project directories exist and patches are fully applied."
      exit 1
    }
  echo "OK: project compileall passed"
else
  echo "Project directories not found in current path; skipped compileall."
fi

log "Done"
echo "Next commands, from your project root:"
echo "  chmod +x scripts/run_all_14_v2_pipelines.sh"
echo "  PYTHON=python3 END_DATE=2026-05-12 JOB_TIMEOUT=8h ./scripts/run_all_14_v2_pipelines.sh"
