"""Production thermal anomaly detection pipeline."""

from __future__ import annotations

import logging
import time

from mine_thermal_anomaly.analysis import (
    analyze_mine_site_thermal,
    analyze_thermal_trends,
    inject_demo_anomaly,
    inject_tailings_warming,
)
from mine_thermal_anomaly.baseline import calculate_thermal_baseline
from mine_thermal_anomaly.config import AppConfig
from mine_thermal_anomaly.detection import detect_thermal_anomalies
from mine_thermal_anomaly.io import export_tables
from mine_thermal_anomaly.modis import fetch_site_thermal

logger = logging.getLogger(__name__)


def run_production(config: AppConfig) -> dict:
    """Execute the full thermal anomaly detection workflow."""
    logger.info("THERMAL ANOMALY DETECTION - PRODUCTION RUN")
    start_time = time.time()
    site = config.site
    detection = config.detection
    logger.info("\n1. Fetching MODIS LST Data...")
    thermal_data = fetch_site_thermal(config, site.latitude, site.longitude)
    logger.info("   Collected %s observations", len(thermal_data))
    logger.info(
        "   Temperature range: %.1f°C to %.1f°C",
        thermal_data["lst_day_celsius"].min(),
        thermal_data["lst_day_celsius"].max(),
    )
    logger.info("\n2. Calculating Thermal Baseline...")
    baseline = calculate_thermal_baseline(thermal_data)
    logger.info("   Day Mean: %.2f°C", baseline["overall"]["day_mean"])
    logger.info("   Day Std Dev: %.2f°C", baseline["overall"]["day_std"])
    logger.info("   Day 95th Percentile: %.2f°C", baseline["overall"]["day_p95"])
    logger.info("\n3. Detecting Thermal Anomalies...")
    thermal_with_anomaly = inject_demo_anomaly(thermal_data, detection)
    anomalies = detect_thermal_anomalies(thermal_with_anomaly, baseline, detection)
    anomaly_count = int(anomalies["any_anomaly"].sum())
    anomaly_pct = (anomaly_count / len(anomalies)) * 100
    logger.info("   Anomalies Detected: %s (%.1f%%)", anomaly_count, anomaly_pct)
    logger.info(
        "   Maximum Severity Score: %.1f/100",
        anomalies["anomaly_score"].max(),
    )
    logger.info("\n4. Analyzing Multiple Mine Features...")
    site_analysis = analyze_mine_site_thermal(config)
    high_risk_count = int((site_analysis["risk_level"] == "HIGH").sum())
    medium_risk_count = int((site_analysis["risk_level"] == "MEDIUM").sum())
    logger.info("   Features Analyzed: %s", len(site_analysis))
    logger.info("   High Risk Features: %s", high_risk_count)
    logger.info("   Medium Risk Features: %s", medium_risk_count)
    logger.info("\n5. Analyzing Thermal Trends...")
    tailings = next(f for f in config.features if f.type == "tailings_dam")
    tailings_thermal = fetch_site_thermal(config, tailings.lat, tailings.lon)
    tailings_thermal = inject_tailings_warming(tailings_thermal, config)
    baseline_tailings = calculate_thermal_baseline(tailings_thermal)
    trend_analysis = analyze_thermal_trends(
        tailings_thermal, baseline_tailings, detection.trend_window_days
    )
    logger.info("   Trend Status: %s", trend_analysis["trend_status"])
    logger.info("   Urgency Level: %s", trend_analysis["urgency"])
    logger.info(
        "   Annual Trend: %+.2f°C/year",
        trend_analysis["annual_trend_celsius"],
    )
    logger.info("\n6. Exporting Results...")
    table_paths = export_tables(
        config,
        baseline=thermal_data,
        anomalies=anomalies,
        site_analysis=site_analysis,
    )
    for path in table_paths:
        logger.info("   Exported: %s", path.name)

    execution_time = time.time() - start_time
    logger.info("=== PERFORMANCE METRICS ===")
    logger.info("Total Execution Time: %.3f seconds", execution_time)
    logger.info(
        "Observations Processed: %s",
        len(thermal_data) * len(config.features),
    )
    logger.info("Anomaly Detection Rate: %.2f%%", anomaly_pct)
    logger.info(
        "Features at High Risk: %s/%s",
        high_risk_count,
        len(config.features),
    )
    return {
        "anomaly_pct": float(anomaly_pct),
        "high_risk_count": high_risk_count,
        "execution_time": execution_time,
        "table_paths": table_paths,
        "trend_analysis": trend_analysis,
    }
