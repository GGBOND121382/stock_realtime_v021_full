#!/usr/bin/env python3
"""Minimal read-only HTTP API for the AS1455 Android executor.

The default production request only publishes the committed
``execution_batch.json``.  An explicit ``?experiment=...`` request may build an
in-memory test batch from an already materialized dashboard plan when no
committed batch exists.  The fallback never writes files and is never used by
the default production request.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

DEFAULT_LIVE_ROOT = Path("saved_data/ashare_ml4t/live_as1455")
DEFAULT_EXPERIMENT = "r21_best_reb21_fold0_4_forward"
PROTOCOL = "as1455_execution_batch_v1"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EXPERIMENT_RE = re.compile(
    r"^r\d{2}_(?:all5|first3|best)_reb\d+_fold0_[45]_forward$"
)


def shanghai_today() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


def validate_experiment_name(value: str) -> str:
    experiment = str(value).strip()
    if not EXPERIMENT_RE.fullmatch(experiment):
        raise ValueError(f"invalid experiment: {value}")
    return experiment


def request_has_experiment_override(query: str) -> bool:
    return "experiment" in parse_qs(query, keep_blank_values=True)


def select_request_experiment(query: str, default_experiment: str) -> str:
    params = parse_qs(query, keep_blank_values=True)
    values = params.get("experiment")
    if values is None:
        return validate_experiment_name(default_experiment)
    if len(values) != 1 or not values[0].strip():
        raise ValueError(
            "experiment query parameter must appear exactly once and be non-empty"
        )
    return validate_experiment_name(values[0])


def batch_path(live_root: Path, trade_date: str, experiment: str) -> Path:
    experiment = validate_experiment_name(experiment)
    token = trade_date.replace("-", "")
    return (
        live_root
        / token
        / "nine_strategy"
        / "strategies"
        / experiment
        / "execution_batch.json"
    )


def load_ready_batch(live_root: Path, trade_date: str, experiment: str) -> dict[str, Any] | None:
    if not DATE_RE.fullmatch(trade_date):
        raise ValueError(f"invalid trade date: {trade_date}")
    experiment = validate_experiment_name(experiment)
    path = batch_path(live_root, trade_date, experiment)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ready":
        return None
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unexpected execution protocol in {path}: {payload.get('protocol')}")
    if payload.get("trade_date") != trade_date:
        raise RuntimeError(f"execution trade_date mismatch in {path}")
    if payload.get("experiment") != experiment:
        raise RuntimeError(f"execution experiment mismatch in {path}")
    orders = payload.get("orders")
    if not isinstance(orders, list) or int(payload.get("order_count", -1)) != len(orders):
        raise RuntimeError(f"invalid order payload in {path}")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _code6(value: object) -> str:
    matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", str(value or "").strip())
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve six-digit stock code from {value!r}")
    return matches[0]


def _positive_decimal(value: object, field: str) -> Decimal:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise RuntimeError(f"invalid {field}={value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise RuntimeError(f"invalid {field}={value!r}")
    return number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _positive_integer(row: dict[str, str]) -> int:
    for name in ("filled_shares", "shares", "intended_shares"):
        raw = str(row.get(name, "") or "").strip()
        if not raw:
            continue
        try:
            number = Decimal(raw)
        except InvalidOperation as exc:
            raise RuntimeError(f"invalid {name}={raw!r}") from exc
        if not number.is_finite() or number <= 0 or number != number.to_integral_value():
            raise RuntimeError(f"invalid {name}={raw!r}")
        return int(number)
    raise RuntimeError("planned order lacks a positive integer share quantity")


def _reference_price(row: dict[str, str]) -> Decimal:
    for name in ("raw_exec_price", "raw_close_1500"):
        raw = str(row.get(name, "") or "").strip()
        if raw:
            return _positive_decimal(raw, name)
    raise RuntimeError("planned order lacks raw_exec_price/raw_close_1500")


def _optional_float(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        number = Decimal(raw)
    except InvalidOperation as exc:
        raise RuntimeError(f"invalid numeric value={value!r}") from exc
    if not number.is_finite():
        return None
    return float(number)


def _signal_id(
    trade_date: str,
    experiment: str,
    code: str,
    side: str,
    qty: int,
    submit_price: Decimal,
    rank: object,
    reason: object,
) -> str:
    stable = "|".join(
        [
            trade_date,
            experiment,
            code,
            side,
            str(qty),
            format(submit_price, "f"),
            str(rank or "").strip(),
            str(reason or "").strip(),
        ]
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _temporary_plan_root(
    live_root: Path, trade_date: str, experiment: str
) -> Path | None:
    token = trade_date.replace("-", "")
    nine_root = live_root / token / "nine_strategy"
    candidates = (
        nine_root / "start_date_plan" / "strategies" / experiment,
        nine_root / "strategies" / experiment,
    )
    for root in candidates:
        if (root / "strategy_manifest.json").is_file() and (root / "16_live_orders.csv").is_file():
            return root
    return None


def build_temporary_execution_batch(
    live_root: Path, trade_date: str, experiment: str
) -> dict[str, Any] | None:
    """Build one read-only test batch from an existing materialized/live plan."""
    if not DATE_RE.fullmatch(trade_date):
        raise ValueError(f"invalid trade date: {trade_date}")
    experiment = validate_experiment_name(experiment)
    plan_root = _temporary_plan_root(live_root, trade_date, experiment)
    if plan_root is None:
        return None

    manifest = json.loads((plan_root / "strategy_manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("trade_date", trade_date)) != trade_date:
        raise RuntimeError(f"temporary plan trade_date mismatch in {plan_root}")
    if str(manifest.get("experiment", experiment)) != experiment:
        raise RuntimeError(f"temporary plan experiment mismatch in {plan_root}")

    token = trade_date.replace("-", "")
    sidecar_path = live_root / token / "08_live_execution_sidecar.csv"
    if not sidecar_path.is_file():
        raise RuntimeError(f"temporary execution sidecar missing: {sidecar_path}")

    sidecar_by_code: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(sidecar_path):
        raw_symbol = row.get("symbol") or row.get("code")
        code = _code6(raw_symbol)
        if code in sidecar_by_code:
            raise RuntimeError(f"duplicate execution sidecar stock code: {code}")
        sidecar_by_code[code] = row

    prepared: list[dict[str, Any]] = []
    for source_index, row in enumerate(_read_csv_rows(plan_root / "16_live_orders.csv")):
        side = str(row.get("side", "") or "").strip().lower()
        if side not in {"buy", "sell"}:
            raise RuntimeError(f"unsupported planned order side for {experiment}: {side!r}")
        raw_symbol = row.get("symbol") or row.get("code")
        code = _code6(raw_symbol)
        execution_row = sidecar_by_code.get(code)
        if execution_row is None:
            raise RuntimeError(f"order stock missing from execution sidecar: {code}")

        upper = _positive_decimal(execution_row.get("up_limit"), f"up_limit[{code}]")
        lower = _positive_decimal(execution_row.get("down_limit"), f"down_limit[{code}]")
        if lower >= upper:
            raise RuntimeError(f"invalid daily limit range for {code}: lower={lower} upper={upper}")
        reference = _reference_price(row)
        if not (lower <= reference <= upper):
            raise RuntimeError(
                f"reference price outside server daily limits for {code}: "
                f"reference={reference} lower={lower} upper={upper}"
            )

        qty = _positive_integer(row)
        submit_price = upper if side == "buy" else lower
        rank_raw = row.get("rank", "")
        reason = row.get("reason", "")
        prepared.append(
            {
                "_source_index": source_index,
                "signal_id": _signal_id(
                    trade_date,
                    experiment,
                    code,
                    side,
                    qty,
                    submit_price,
                    rank_raw,
                    reason,
                ),
                "code": code,
                "symbol": str(raw_symbol or code),
                "side": side,
                "qty": qty,
                "submit_price": format(submit_price, ".2f"),
                "reference_price": format(reference, ".2f"),
                "upper_limit": format(upper, ".2f"),
                "lower_limit": format(lower, ".2f"),
                "rank": _optional_float(rank_raw),
                "position_before": _optional_float(row.get("position_before", "")),
                "price_source": "server_execution_sidecar_limits",
            }
        )

    prepared.sort(key=lambda item: (0 if item["side"] == "sell" else 1, item["_source_index"]))
    orders: list[dict[str, Any]] = []
    for sequence, item in enumerate(prepared, start=1):
        final = dict(item)
        final.pop("_source_index", None)
        final["sequence"] = sequence
        orders.append(final)

    return {
        "status": "ready",
        "protocol": PROTOCOL,
        "trade_date": trade_date,
        "experiment": experiment,
        "order_count": len(orders),
        "submit_sequence": "sells_then_buys",
        "price_semantics": (
            "server_final_limit_price: buy=upper_limit, sell=lower_limit; "
            "reference_price is audit/research only"
        ),
        "price_source": "server_execution_sidecar_limits",
        "broker_orders_submitted": False,
        "windows_required_files": [],
        "windows_quote_lookup_required": False,
        "windows_account_reads_required": False,
        "temporary_test_batch": True,
        "temporary_plan_source": str(plan_root),
        "orders": orders,
    }


def load_request_batch(
    live_root: Path,
    trade_date: str,
    default_experiment: str,
    query: str,
) -> dict[str, Any] | None:
    """Resolve a request without changing the default production semantics."""
    experiment = select_request_experiment(query, default_experiment)
    committed = load_ready_batch(live_root, trade_date, experiment)
    if committed is not None:
        return committed
    if not request_has_experiment_override(query):
        return None
    return build_temporary_execution_batch(live_root, trade_date, experiment)


class ExecutionAPIHandler(BaseHTTPRequestHandler):
    server_version = "AS1455ExecutionAPI/1"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.api_token  # type: ignore[attr-defined]
        if not expected:
            return True
        raw = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not raw.startswith(prefix):
            return False
        return hmac.compare_digest(raw[len(prefix):], expected)

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return

        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok", "service": "as1455_execution_api"})
            return
        if path == "/api/v1/execution/latest":
            trade_date = shanghai_today()
        elif path.startswith("/api/v1/execution/"):
            trade_date = path.rsplit("/", 1)[-1]
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        try:
            batch = load_request_batch(
                self.server.live_root,  # type: ignore[attr-defined]
                trade_date,
                self.server.experiment,  # type: ignore[attr-defined]
                parsed.query,
            )
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__, "detail": str(exc)})
            return
        if batch is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self._json(HTTPStatus.OK, batch)

    def log_message(self, fmt: str, *args: object) -> None:
        print("[AS1455-EXEC-API] " + fmt % args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("AS1455_EXECUTION_API_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("AS1455_EXECUTION_API_PORT", "8510")))
    p.add_argument("--live-root", default=os.environ.get("AS1455_LIVE_ROOT", str(DEFAULT_LIVE_ROOT)))
    p.add_argument("--experiment", default=os.environ.get("AS1455_PRODUCTION_EXPERIMENT", DEFAULT_EXPERIMENT))
    p.add_argument("--token", default=os.environ.get("AS1455_EXECUTION_API_TOKEN", ""))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ExecutionAPIHandler)
    server.live_root = Path(args.live_root).expanduser().resolve()  # type: ignore[attr-defined]
    server.experiment = validate_experiment_name(str(args.experiment))  # type: ignore[attr-defined]
    server.api_token = str(args.token)  # type: ignore[attr-defined]
    print(
        json.dumps(
            {
                "status": "listening",
                "host": args.host,
                "port": args.port,
                "live_root": str(server.live_root),  # type: ignore[attr-defined]
                "experiment": server.experiment,  # type: ignore[attr-defined]
                "token_required": bool(server.api_token),  # type: ignore[attr-defined]
                "experiment_query_override": True,
                "temporary_test_batch_fallback": True,
            },
            ensure_ascii=False,
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
