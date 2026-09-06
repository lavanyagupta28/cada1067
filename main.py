#!/usr/bin/env python3
"""LLM-assisted EDA netlist exploration and transformation entrypoint."""

from __future__ import annotations

import argparse
import os
import sys


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _load_config(path: str) -> dict:
    """Load and return the YAML configuration mapping."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        sys.exit("ERROR: PyYAML is not installed in cad_env.")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except FileNotFoundError:
        sys.exit(f"ERROR: Config file not found: {path!r}")
    except Exception as exc:
        sys.exit(f"ERROR: Could not parse config file {path!r}: {exc}")

    if not isinstance(config, dict):
        sys.exit(f"ERROR: Config file {path!r} must contain a YAML mapping.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cada1067",
        description="LLM-assisted EDA netlist exploration and transformation.",
    )
    parser.add_argument(
        "-config",
        "--config",
        metavar="<config_file_path>",
        default=os.path.join(_SCRIPT_DIR, "config.yaml"),
        help="Configuration file path (defaults to config.yaml beside this entrypoint).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    from src.agent import EDAAgent
    from src.eda_engine import EDAEngine
    from src.io_handler import IOHandler

    engine = EDAEngine()
    agent = EDAAgent(config, engine)
    IOHandler(agent).run()


if __name__ == "__main__":
    main()
