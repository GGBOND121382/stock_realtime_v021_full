#!/usr/bin/env python3
"""Minimal read-only HTTP API for the AS1455 Android executor.

No strategy calculation happens here. The API only publishes the already
committed ``execution_batch.json`` produced by the server-side production job.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
from datetime import datetime
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
            experiment = select_request_experiment(
                parsed.query,
                self.server.experiment,  # type: ignore[attr-defined]
            )
            batch = load_ready_batch(
                self.server.live_root,  # type: ignore[attr-defined]
                trade_date,
                experiment,
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
            },
            ensure_ascii=False,
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
