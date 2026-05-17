"""Tests for anomaly detection logic."""

import pandas as pd

from mine_thermal_anomaly.baseline import calculate_thermal_baseline
from mine_thermal_anomaly.config import DetectionConfig, load_config
from mine_thermal_anomaly.detection import detect_thermal_anomalies, risk_level
from mine_thermal_anomaly.paths import DEFAULT_CONFIG_PATH


def test_detect_thermal_anomalies_flags_spike():
    # Same ISO week so seasonal std is well-defined for z-scores.
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2022-06-06", periods=6, freq="D"),
            "lst_day_celsius": [20.0, 20.0, 20.0, 20.0, 20.0, 45.0],
            "lst_night_celsius": [10.0, 10.0, 10.0, 10.0, 10.0, 35.0],
        }
    )
    baseline = calculate_thermal_baseline(frame.iloc[:5])
    detection = DetectionConfig(
        threshold_sigma=2.5,
        recent_anomaly_days=90,
        recent_analysis_days=90,
        trend_window_days=90,
        demo_injected_anomaly_celsius=8,
        tailings_warming_start="2023-06-01",
        tailings_warming_rate_celsius=6,
        tailings_warming_days=180,
    )
    result = detect_thermal_anomalies(frame, baseline, detection)
    assert result["any_anomaly"].iloc[-1]


def test_risk_level_thresholds():
    risk = load_config(DEFAULT_CONFIG_PATH).risk
    assert risk_level(70, risk) == "HIGH"
    assert risk_level(50, risk) == "MEDIUM"
    assert risk_level(10, risk) == "LOW"
