"""CLI entry point for production runs and article figures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import signalplot

from mine_thermal_anomaly.config import load_config
from mine_thermal_anomaly.paths import DEFAULT_CONFIG_PATH
from mine_thermal_anomaly.pipeline import run_production
from mine_thermal_anomaly.plots_blog06 import (
    create_main_visualization,
    create_trend_visualization,
)
from mine_thermal_anomaly.plots_blog16 import (
    create_main_thermal_time_series,
    create_spatial_thermal_heatmap,
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _cmd_run(config_path: Path) -> int:
    config = load_config(config_path)
    _configure_logging(config.logging_level)
    np.random.seed(config.random_seed)
    run_production(config)
    return 0


def _cmd_viz06(config_path: Path) -> int:
    config = load_config(config_path)
    _configure_logging(config.logging_level)
    np.random.seed(config.random_seed)
    signalplot.apply(font_family=config.font_family)

    logger = logging.getLogger(__name__)
    logger.info("THERMAL ANOMALY DETECTION - BLOG 06 FIGURES")

    paths = [
        create_main_visualization(config),
        create_trend_visualization(config),
    ]
    saved = [p for p in paths if p is not None]
    if saved:
        logger.info("Wrote %s figure(s) to %s", len(saved), config.output.figures_dir)
        for path in saved:
            logger.info("  - %s", path.name)
    return 0


def _cmd_viz16(config_path: Path) -> int:
    config = load_config(config_path)
    _configure_logging(config.logging_level)
    np.random.seed(config.random_seed)
    signalplot.apply(font_family=config.font_family)

    logger = logging.getLogger(__name__)
    logger.info("MINE THERMAL ANOMALY (MODIS) - BLOG 16 FIGURES")

    paths = [
        create_main_thermal_time_series(config),
        create_spatial_thermal_heatmap(config),
    ]
    saved = [p for p in paths if p is not None]
    if saved:
        logger.info("Wrote %s figure(s) to %s", len(saved), config.output.figures_dir)
        for path in saved:
            logger.info("  - %s", path.name)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Thermal anomaly detection at mine sites (MODIS LST demos)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: repo config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Production pipeline: detect, assess, export CSVs")
    subparsers.add_parser("viz-06", help="Generate blog 06 article figures")
    subparsers.add_parser("viz-16", help="Generate blog 16 MODIS article figures")

    args = parser.parse_args(argv)

    if args.command == "run":
        return _cmd_run(args.config)
    if args.command == "viz-06":
        return _cmd_viz06(args.config)
    if args.command == "viz-16":
        return _cmd_viz16(args.config)

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
