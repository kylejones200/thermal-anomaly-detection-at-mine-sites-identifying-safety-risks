"""Thermal anomaly scoring against seasonal baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mine_thermal_anomaly.config import DetectionConfig, RiskConfig


def detect_thermal_anomalies(
    thermal_data: pd.DataFrame,
    baseline: dict,
    detection: DetectionConfig,
) -> pd.DataFrame:
    """Detect thermal anomalies using statistical thresholds."""
    result = thermal_data.copy()
    result["week_of_year"] = result["date"].dt.isocalendar().week
    result = result.merge(baseline["seasonal"], left_on="week_of_year", right_on="week", how="left")
    result["day_z_score"] = (result["lst_day_celsius"] - result["day_mean"]) / result["day_std"]
    result["night_z_score"] = (result["lst_night_celsius"] - result["night_mean"]) / result[
        "night_std"
    ]
    threshold = detection.threshold_sigma
    result["day_anomaly"] = result["day_z_score"] > threshold
    result["night_anomaly"] = result["night_z_score"] > threshold
    result["any_anomaly"] = result["day_anomaly"] | result["night_anomaly"]
    result["anomaly_score"] = np.clip(result["day_z_score"] * 20, 0, 100)
    return result


def risk_level(score: float, risk: RiskConfig) -> str:
    if score > risk.high_score:
        return "HIGH"
    if score > risk.medium_score:
        return "MEDIUM"
    return "LOW"
