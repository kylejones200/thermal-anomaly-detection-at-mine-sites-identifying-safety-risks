"""Figures for the thermal anomaly detection article (blog 06)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from mine_thermal_anomaly.analysis import inject_demo_anomaly, inject_tailings_warming
from mine_thermal_anomaly.baseline import calculate_thermal_baseline
from mine_thermal_anomaly.config import AppConfig
from mine_thermal_anomaly.detection import detect_thermal_anomalies
from mine_thermal_anomaly.modis import fetch_site_thermal
from mine_thermal_anomaly.style import apply_minimalist_style

logger = logging.getLogger(__name__)


def _figures_dir(config: AppConfig) -> Path:
    path = config.output.figures_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_main_visualization(config: AppConfig) -> Path | None:
    """Temperature time series with baseline and anomaly scores."""
    logger.info("Creating main thermal anomaly visualization...")
    site = config.site
    thermal_data = fetch_site_thermal(config, site.latitude, site.longitude)
    baseline = calculate_thermal_baseline(thermal_data)
    thermal_with_anomaly = inject_demo_anomaly(thermal_data, config.detection)
    anomalies = detect_thermal_anomalies(thermal_with_anomaly, baseline, config.detection)
    if not config.output.save_figures:
        logger.info("  Skipping save (save_figures=false)")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(
        thermal_data["date"],
        thermal_data["lst_day_celsius"],
        color="black",
        linewidth=1,
        label="Historical Temperature",
    )
    ax1.axhline(
        y=baseline["overall"]["day_mean"],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Baseline Mean",
    )
    ax1.axhline(
        y=baseline["overall"]["day_p95"],
        color="gray",
        linestyle=":",
        linewidth=0.8,
        label="95th Percentile",
    )
    anomaly_period = anomalies[anomalies["any_anomaly"]]
    ax1.scatter(
        anomaly_period["date"],
        anomaly_period["lst_day_celsius"],
        color="black",
        s=50,
        marker="o",
        facecolors="white",
        edgecolors="black",
        linewidths=1.5,
        label="Detected Anomalies",
        zorder=5,
    )
    apply_minimalist_style(ax1)
    ax1.set_title(
        "Thermal Anomaly Detection at Mine Tailings Dam",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Land Surface Temperature (°C)", fontsize=10)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.fill_between(anomalies["date"], 0, anomalies["anomaly_score"], color="gray", alpha=0.3)
    ax2.plot(anomalies["date"], anomalies["anomaly_score"], color="black", linewidth=1)
    ax2.axhline(y=40, color="gray", linestyle="--", linewidth=0.8, label="Medium Risk")
    ax2.axhline(y=60, color="gray", linestyle="-.", linewidth=0.8, label="High Risk")
    apply_minimalist_style(ax2)
    ax2.set_title("Anomaly Severity Score", fontsize=12, fontweight="bold", loc="left")
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Anomaly Score (0-100)", fontsize=10)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_ylim(0, 105)
    out = _figures_dir(config) / config.output.blog06_main_figure
    fig.savefig(out, dpi=config.output.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Wrote %s", out.name)
    return out


def create_trend_visualization(config: AppConfig) -> Path | None:
    """Rolling mean and deviation-from-baseline trend figure."""
    logger.info("Creating thermal trend visualization...")
    thermal_data = fetch_site_thermal(config, config.site.latitude, config.site.longitude)
    thermal_data = inject_tailings_warming(thermal_data, config)
    thermal_sorted = thermal_data.sort_values("date").copy()
    thermal_sorted["rolling_mean"] = (
        thermal_sorted["lst_day_celsius"].rolling(window=6, min_periods=3).mean()
    )
    warming_start = config.detection.tailings_warming_start
    baseline_data = thermal_data[thermal_data["date"] < warming_start]
    baseline_mean = baseline_data["lst_day_celsius"].mean()
    if not config.output.save_figures:
        logger.info("  Skipping save (save_figures=false)")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(
        thermal_sorted["date"],
        thermal_sorted["lst_day_celsius"],
        color="lightgray",
        linewidth=0.8,
        label="Raw Temperature",
    )
    ax1.plot(
        thermal_sorted["date"],
        thermal_sorted["rolling_mean"],
        color="black",
        linewidth=1.5,
        label="6-Period Moving Average",
    )
    ax1.axhline(
        y=baseline_mean,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Historical Baseline",
    )
    warming_period = thermal_sorted[thermal_sorted["date"] > warming_start]
    ax1.axvspan(
        warming_period["date"].min(),
        warming_period["date"].max(),
        alpha=0.1,
        color="gray",
        label="Warming Period",
    )
    apply_minimalist_style(ax1)
    ax1.set_title(
        "Thermal Trend Analysis: Tailings Dam",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Land Surface Temperature (°C)", fontsize=10)
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    thermal_sorted["deviation"] = thermal_sorted["lst_day_celsius"] - baseline_mean
    colors = ["black" if x >= 0 else "gray" for x in thermal_sorted["deviation"]]
    ax2.bar(
        thermal_sorted["date"],
        thermal_sorted["deviation"],
        color=colors,
        width=6,
        alpha=0.6,
    )
    ax2.axhline(y=0, color="black", linewidth=0.8)
    apply_minimalist_style(ax2)
    ax2.set_title(
        "Temperature Deviation from Baseline",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Temperature Deviation (°C)", fontsize=10)
    out = _figures_dir(config) / config.output.blog06_trend_figure
    fig.savefig(out, dpi=config.output.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Wrote %s", out.name)
    return out
