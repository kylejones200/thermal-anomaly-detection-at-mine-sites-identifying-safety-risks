"""Write pipeline outputs to disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mine_thermal_anomaly.config import AppConfig


def tables_dir(config: AppConfig) -> Path:
    path = config.output.tables_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_tables(
    config: AppConfig,
    *,
    baseline: pd.DataFrame,
    anomalies: pd.DataFrame,
    site_analysis: pd.DataFrame,
) -> list[Path]:
    if not config.output.save_tables:
        return []

    out = tables_dir(config)
    paths = [
        out / config.output.baseline_csv,
        out / config.output.anomalies_csv,
        out / config.output.site_analysis_csv,
    ]
    baseline.to_csv(paths[0], index=False)
    anomalies.to_csv(paths[1], index=False)
    site_analysis.to_csv(paths[2], index=False)
    return paths
