#!/usr/bin/env bash
# Stable repository-root entry point. The implementation lives in proxy_tools/.
_PROXYCTL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROXYCTL_IMPL="${_PROXYCTL_ROOT}/proxy_tools/proxyctl.sh"

if [[ ! -f "$_PROXYCTL_IMPL" ]]; then
    echo "[ERROR] missing proxy implementation: $_PROXYCTL_IMPL" >&2
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return 1
    else
        exit 1
    fi
fi

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    source "$_PROXYCTL_IMPL" "$@"
    unset _PROXYCTL_ROOT _PROXYCTL_IMPL
else
    exec bash "$_PROXYCTL_IMPL" "$@"
fi
