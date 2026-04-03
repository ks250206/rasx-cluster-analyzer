"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rasx_cluster_analyzer.config import ConfigError, load_config
from rasx_cluster_analyzer.pipeline import run_analysis


def _default_config_path() -> Path:
    return Path("config.toml").resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cluster Rigaku rasX map patterns and export a Plotly HTML map.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ./config.toml in the current working directory)",
    )
    parser.add_argument(
        "rasx_dir",
        type=Path,
        help="Directory containing .rasx map files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    ns = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if ns.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg_path = ns.config if ns.config is not None else _default_config_path()
    if not cfg_path.is_file():
        logging.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    try:
        cfg = load_config(cfg_path)
    except ConfigError as e:
        logging.error("%s", e)
        sys.exit(1)

    try:
        out = run_analysis(ns.rasx_dir, cfg)
    except (OSError, ValueError, NotADirectoryError) as e:
        logging.error("%s", e)
        sys.exit(1)

    logging.info("Done: %s", out)


if __name__ == "__main__":
    main()
