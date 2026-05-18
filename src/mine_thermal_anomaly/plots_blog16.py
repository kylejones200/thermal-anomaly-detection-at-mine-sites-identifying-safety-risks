"""Figures for distributed MODIS thermal monitoring article (blog 16)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mine_thermal_anomaly.config import AppConfig
from mine_thermal_anomaly.style import apply_minimalist_style

logger = logging.getLogger(__name__)


def _figures_dir(config: AppConfig) -> Path:
    path = config.output.figures_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_mine_thermal_data(seed: int) -> tuple:
    """Generate synthetic MODIS LST data for mine sites with anomalies."""
    np.random.seed(seed)
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
    days = np.arange(365)
    baseline = 298 + 15 * np.sin((days - 80) * 2 * np.pi / 365)
    baseline += np.random.randn(365) * 2.5
    normal_site = baseline + np.random.randn(365) * 1.5
    tailings_site = baseline.copy()
    anomaly_periods = [(120, 145), (220, 235), (310, 330)]
    for start, end in anomaly_periods:
        intensity = np.random.uniform(15, 25)
        tailings_site[start:end] += intensity * np.exp(
            -((np.arange(end - start) - (end - start) / 2) ** 2) / 20
        )
    tailings_site += np.random.randn(365) * 2.0
    waste_site = baseline + 0.03 * days + np.random.randn(365) * 2.0
    return dates, baseline, normal_site, tailings_site, waste_site


def create_main_thermal_time_series(config: AppConfig) -> Path | None:
    """Time series plot showing thermal anomalies across mine sites."""
    logger.info("Generating main thermal time series visualization...")
    dates, baseline, normal, tailings, waste = generate_mine_thermal_data(config.random_seed)
    baseline_c = baseline - 273.15
    normal_c = normal - 273.15
    tailings_c = tailings - 273.15
    waste_c = waste - 273.15
    z_scores = (tailings - baseline) / np.std(tailings - baseline)
    anomalies = np.abs(z_scores) > 3
    anomaly_dates = [d for d, flag in zip(dates, anomalies) if flag]
    anomaly_temps = [t for t, flag in zip(tailings_c, anomalies) if flag]
    if not config.output.save_figures:
        logger.info("  Skipping save (save_figures=false); %s anomalies", len(anomaly_dates))
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(
        dates, baseline_c, "--", color="black", linewidth=2, label="Seasonal Baseline", alpha=0.7
    )
    ax1.plot(
        dates, normal_c, "-", color="black", linewidth=1.5, label="Normal Mine Site", alpha=0.8
    )
    ax1.plot(
        dates, waste_c, "-", color="black", linewidth=1.5, label="Waste Dump (Oxidation)", alpha=0.8
    )
    ax1.plot(dates, tailings_c, "-", color="black", linewidth=1.5, label="Tailings Dam", alpha=0.8)
    ax1.scatter(
        anomaly_dates,
        anomaly_temps,
        s=100,
        facecolors="none",
        edgecolors="#FF4136",
        linewidths=2,
        label=f"Thermal Anomalies (n={len(anomaly_dates)})",
        zorder=5,
    )
    apply_minimalist_style(ax1)
    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.set_title(
        "Mine-Site Thermal Monitoring with MODIS LST",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax1.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    event_idx = 128
    ax1.annotate(
        "Spontaneous Combustion Event",
        xy=(dates[event_idx], tailings_c[event_idx]),
        xytext=(dates[event_idx + 50], tailings_c[event_idx] + 8),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5},
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1,
        },
    )
    ax2.plot(
        dates,
        z_scores,
        "o-",
        color="black",
        linewidth=1,
        markersize=2,
        markerfacecolor="white",
        markeredgecolor="black",
        label="Z-Score (Tailings Dam)",
    )
    ax2.axhline(y=3, color="black", linestyle="--", linewidth=2, label="Anomaly Threshold (±3σ)")
    ax2.axhline(y=-3, color="black", linestyle="--", linewidth=2)
    ax2.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax2.fill_between(
        dates,
        -10,
        10,
        where=np.abs(z_scores) > 3,
        color="black",
        alpha=0.2,
        label="Anomaly Regions",
    )
    apply_minimalist_style(ax2)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Z-Score (σ)", fontsize=11)
    ax2.set_title(
        "Statistical Anomaly Detection",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=15,
    )
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_ylim(-6, 8)
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = _figures_dir(config) / config.output.blog16_timeseries_figure
    fig.savefig(out, dpi=config.output.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Wrote %s (%s anomalies)", out.name, len(anomaly_dates))
    return out


def create_spatial_thermal_heatmap(config: AppConfig) -> Path | None:
    """Spatial heatmap showing thermal patterns across a mine site."""
    logger.info("Generating spatial thermal heatmap visualization...")
    np.random.seed(config.random_seed)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    temp = 25 + 5 * np.sin(x_grid * 0.5) + 3 * np.cos(y_grid * 0.7)
    tailings_x, tailings_y = 3.5, 7.0
    dist_tailings = np.sqrt((x_grid - tailings_x) ** 2 + (y_grid - tailings_y) ** 2)
    temp += 20 * np.exp(-(dist_tailings**2) / 0.4)
    waste_x, waste_y = 7.5, 3.5
    dist_waste = np.sqrt((x_grid - waste_x) ** 2 + (y_grid - waste_y) ** 2)
    temp += 12 * np.exp(-(dist_waste**2) / 0.8)
    plant_x, plant_y = 5.0, 5.0
    dist_plant = np.sqrt((x_grid - plant_x) ** 2 + (y_grid - plant_y) ** 2)
    temp += 8 * np.exp(-(dist_plant**2) / 0.3)
    temp += np.random.randn(100, 100) * 1.5
    z_scores = (temp - np.mean(temp)) / np.std(temp)
    if not config.output.save_figures:
        logger.info("  Skipping save (save_figures=false)")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    im1 = ax1.contourf(x_grid, y_grid, temp, levels=20, cmap="hot")
    ax1.plot(
        tailings_x,
        tailings_y,
        "c^",
        markersize=15,
        markeredgecolor="black",
        markeredgewidth=2,
        label="Tailings Dam",
    )
    ax1.plot(
        waste_x,
        waste_y,
        "cs",
        markersize=15,
        markeredgecolor="black",
        markeredgewidth=2,
        label="Waste Dump",
    )
    ax1.plot(
        plant_x,
        plant_y,
        "co",
        markersize=15,
        markeredgecolor="black",
        markeredgewidth=2,
        label="Processing Plant",
    )
    apply_minimalist_style(ax1)
    ax1.set_xlabel("Easting (km)", fontsize=10)
    ax1.set_ylabel("Northing (km)", fontsize=10)
    ax1.set_title("MODIS LST Temperature Map", fontsize=12, fontweight="bold", loc="center", pad=15)
    ax1.legend(loc="upper right", frameon=False, fontsize=8)
    ax1.set_aspect("equal")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Temperature (°C)", fontsize=10)
    cbar1.outline.set_visible(False)
    im2 = ax2.contourf(x_grid, y_grid, z_scores, levels=20, cmap="gray")
    ax2.contour(x_grid, y_grid, z_scores, levels=[3], colors="red", linewidths=3, linestyles="--")
    ax2.plot(
        tailings_x,
        tailings_y,
        "k^",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Tailings Dam",
    )
    ax2.plot(
        waste_x,
        waste_y,
        "ks",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Waste Dump",
    )
    ax2.plot(
        plant_x,
        plant_y,
        "ko",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=2,
        label="Processing Plant",
    )
    apply_minimalist_style(ax2)
    ax2.set_xlabel("Easting (km)", fontsize=10)
    ax2.set_ylabel("Northing (km)", fontsize=10)
    ax2.set_title("Z-Score Anomaly Detection", fontsize=12, fontweight="bold", loc="center", pad=15)
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.set_aspect("equal")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Z-Score (σ)", fontsize=10)
    cbar2.outline.set_visible(False)
    fig.suptitle("Spatial Thermal Anomaly Analysis", fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = _figures_dir(config) / config.output.blog16_spatial_figure
    fig.savefig(out, dpi=config.output.figure_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Wrote %s", out.name)
    return out
