"""Multi-feature site analysis and thermal trend diagnostics."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from mine_thermal_anomaly.baseline import calculate_thermal_baseline
from mine_thermal_anomaly.config import AppConfig, DetectionConfig
from mine_thermal_anomaly.detection import detect_thermal_anomalies, risk_level
from mine_thermal_anomaly.modis import apply_feature_thermal_effects, fetch_site_thermal


def analyze_mine_site_thermal(config: AppConfig) -> pd.DataFrame:
    """Analyze thermal patterns across multiple mine features."""
    all_results = []
    detection = config.detection

    for feature in config.features:
        thermal = fetch_site_thermal(config, feature.lat, feature.lon)
        thermal = apply_feature_thermal_effects(thermal, feature)

        baseline = calculate_thermal_baseline(thermal)
        anomalies = detect_thermal_anomalies(thermal, baseline, detection)

        recent_period = anomalies["date"] > (
            anomalies["date"].max() - timedelta(days=detection.recent_analysis_days)
        )
        recent_anomalies = anomalies[recent_period]

        max_score = recent_anomalies["anomaly_score"].max()
        all_results.append(
            {
                "site": config.site.name,
                "feature_name": feature.name,
                "feature_type": feature.type,
                "latitude": feature.lat,
                "longitude": feature.lon,
                "recent_mean_temp": recent_anomalies["lst_day_celsius"].mean(),
                "recent_max_temp": recent_anomalies["lst_day_celsius"].max(),
                "anomaly_count_90d": int(recent_anomalies["any_anomaly"].sum()),
                "max_anomaly_score": max_score,
                "mean_z_score": recent_anomalies["day_z_score"].mean(),
                "risk_level": risk_level(float(max_score), config.risk),
            }
        )

    return pd.DataFrame(all_results)


def analyze_thermal_trends(
    thermal_data: pd.DataFrame, baseline: dict, window_days: int
) -> dict:
    """Analyze thermal trends to identify developing problems."""
    thermal_sorted = thermal_data.sort_values("date").copy()
    roll_window = max(window_days // 8, 1)
    thermal_sorted["rolling_mean"] = (
        thermal_sorted["lst_day_celsius"]
        .rolling(window=roll_window, min_periods=3)
        .mean()
    )

    recent_6mo = thermal_sorted[
        thermal_sorted["date"] > (thermal_sorted["date"].max() - timedelta(days=180))
    ]

    if len(recent_6mo) >= 10:
        x = np.arange(len(recent_6mo))
        y = recent_6mo["lst_day_celsius"].values
        coefficients = np.polyfit(x, y, 1)
        trend_slope = coefficients[0]
        observations_per_year = 365 / 8
        annual_trend = trend_slope * observations_per_year
    else:
        annual_trend = 0.0

    recent_30d = thermal_sorted[
        thermal_sorted["date"] > (thermal_sorted["date"].max() - timedelta(days=30))
    ]
    recent_mean = recent_30d["lst_day_celsius"].mean()
    baseline_mean = baseline["overall"]["day_mean"]
    deviation_from_baseline = recent_mean - baseline_mean

    if annual_trend > 2:
        trend_status, urgency = "WARMING", "HIGH"
    elif annual_trend > 0.5:
        trend_status, urgency = "SLIGHT_WARMING", "MEDIUM"
    elif annual_trend < -0.5:
        trend_status, urgency = "COOLING", "LOW"
    else:
        trend_status, urgency = "STABLE", "LOW"

    return {
        "annual_trend_celsius": annual_trend,
        "trend_status": trend_status,
        "urgency": urgency,
        "recent_mean": recent_mean,
        "baseline_mean": baseline_mean,
        "deviation": deviation_from_baseline,
    }


def inject_demo_anomaly(thermal: pd.DataFrame, detection: DetectionConfig) -> pd.DataFrame:
    """Add a recent temperature spike for demonstration anomaly detection."""
    result = thermal.copy()
    recent_dates = result["date"] > (
        result["date"].max() - timedelta(days=detection.recent_anomaly_days)
    )
    result.loc[recent_dates, "lst_day_celsius"] += detection.demo_injected_anomaly_celsius
    return result


def inject_tailings_warming(thermal: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Add gradual warming on tailings for trend-analysis demos."""
    result = thermal.copy()
    detection = config.detection
    recent_mask = result["date"] > detection.tailings_warming_start
    days_recent = (
        result.loc[recent_mask, "date"] - result.loc[recent_mask, "date"].min()
    ).dt.days
    rate = detection.tailings_warming_rate_celsius / detection.tailings_warming_days
    result.loc[recent_mask, "lst_day_celsius"] += days_recent * rate
    return result
