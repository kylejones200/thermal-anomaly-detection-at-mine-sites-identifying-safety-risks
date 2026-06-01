"""Seasonal thermal baseline statistics."""

from __future__ import annotations

import pandas as pd


def calculate_thermal_baseline(thermal_data: pd.DataFrame) -> dict:
    """Calculate seasonal baseline statistics."""
    frame = thermal_data.copy()
    frame["week_of_year"] = frame["date"].dt.isocalendar().week
    seasonal_baseline = (
        frame.groupby("week_of_year")
        .agg({"lst_day_celsius": ["mean", "std"], "lst_night_celsius": ["mean", "std"]})
        .reset_index()
    )
    seasonal_baseline.columns = [
        "week",
        "day_mean",
        "day_std",
        "night_mean",
        "night_std",
    ]
    overall_stats = {
        "day_mean": frame["lst_day_celsius"].mean(),
        "day_std": frame["lst_day_celsius"].std(),
        "day_p95": frame["lst_day_celsius"].quantile(0.95),
        "day_p99": frame["lst_day_celsius"].quantile(0.99),
        "night_mean": frame["lst_night_celsius"].mean(),
        "night_std": frame["lst_night_celsius"].std(),
    }
    return {"seasonal": seasonal_baseline, "overall": overall_stats}
