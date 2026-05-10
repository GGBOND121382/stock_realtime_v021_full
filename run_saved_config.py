#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run command configs saved under saved_configs/.

The config format is TOML so each command can keep comments next to the
parameters that matter. Commands run from the stock_realtime project directory
by default, which keeps relative paths stable after moving outputs into
saved_data/ and saved_models/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Python 3.11+ has tomllib built in. For Python 3.10 or earlier, install tomli: "
            "pip install tomli"
        ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_DIR / "saved_configs"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    commands = data.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"{path} must contain at least one [[commands]] item")
    return data


def resolve_config(name_or_path: str) -> Path:
    raw = Path(name_or_path)
    candidates = []
    if raw.suffix:
        candidates.append(raw)
        candidates.append(CONFIG_DIR / raw.name)
    else:
        candidates.extend([
            raw,
            CONFIG_DIR / f"{name_or_path}.toml",
            CONFIG_DIR / name_or_path / "config.toml",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"config not found: {name_or_path}")


def expand_value(value: str, python_override: str | None = None) -> str:
    python = python_override or sys.executable
    replacements = {
        "python": python,
        "project_dir": str(PROJECT_DIR),
        "saved_data": str(PROJECT_DIR / "saved_data"),
        "saved_models": str(PROJECT_DIR / "saved_models"),
    }
    out = value
    for key, repl in replacements.items():
        out = out.replace("{" + key + "}", repl)
    return os.path.expandvars(out)


def expand_args(args: list[Any], python_override: str | None = None) -> list[str]:
    return [expand_value(str(x), python_override=python_override) for x in args]


def iter_config_files() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.toml"))


def list_configs() -> None:
    for path in iter_config_files():
        try:
            data = load_config(path)
            title = data.get("title") or path.stem
            command_count = len(data.get("commands", []))
            print(f"{path.stem}\tcommands={command_count}\t{title}")
        except Exception as exc:
            print(f"{path.stem}\tERROR {type(exc).__name__}: {exc}")


def print_command(config_path: Path, command: dict[str, Any], python_override: str | None = None) -> None:
    args = expand_args(command["args"], python_override=python_override)
    print(f"[{config_path.name}] {command.get('id', '<unnamed>')}: {command.get('description', '')}")
    print(" ".join(quote_arg(x) for x in args))


def quote_arg(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def select_commands(data: dict[str, Any], only: str) -> list[dict[str, Any]]:
    commands = data["commands"]
    if only == "all":
        return commands
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    selected = [cmd for cmd in commands if str(cmd.get("id", "")) in wanted]
    missing = wanted - {str(cmd.get("id", "")) for cmd in selected}
    if missing:
        raise KeyError(f"command id not found: {sorted(missing)}")
    return selected


def run_config(args: argparse.Namespace) -> int:
    config_path = resolve_config(args.config)
    data = load_config(config_path)
    commands = select_commands(data, args.command)
    cwd = Path(expand_value(str(data.get("cwd", PROJECT_DIR)), python_override=args.python)).resolve()
    env = os.environ.copy()
    for key, value in (data.get("env") or {}).items():
        env[str(key)] = expand_value(str(value), python_override=args.python)

    for command in commands:
        cmd_args = expand_args(command["args"], python_override=args.python)
        print_command(config_path, command, python_override=args.python)
        if args.dry_run:
            continue
        subprocess.run(cmd_args, cwd=str(cwd), env=env, check=True)
    return 0


def show_config(args: argparse.Namespace) -> int:
    config_path = resolve_config(args.config)
    data = load_config(config_path)
    for command in select_commands(data, args.command):
        print_command(config_path, command, python_override=args.python)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run saved stock_realtime command configs")
    sub = p.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="List saved config files")
    list_p.set_defaults(func=lambda _args: (list_configs() or 0))

    for name, help_text in [("show", "Print command(s) without running"), ("run", "Run command(s)")]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("config", help="Config stem or path, e.g. 600312_pipeline")
        sp.add_argument("--command", default="all", help="Command id, comma list, or all")
        sp.add_argument("--python", default=None, help="Override {python}; default is current interpreter")
        if name == "run":
            sp.add_argument("--dry-run", action="store_true", help="Print commands but do not execute")
        sp.set_defaults(func=show_config if name == "show" else run_config)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(args.func(args))
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] command failed with returncode={exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
