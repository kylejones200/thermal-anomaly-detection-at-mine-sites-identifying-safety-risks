"""Synthetic MODIS land-surface temperature series for demonstrations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mine_thermal_anomaly.config import AppConfig, DataConfig, FeatureConfig


def kelvin_to_celsius(kelvin: float | np.ndarray) -> float | np.ndarray:
    return kelvin - 273.15


def fetch_modis_lst_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    data: DataConfig,
    *,
    freq_days: int | None = None,
) -> pd.DataFrame:
    """Generate realistic MODIS LST data matching satellite revisit characteristics."""
    freq = f"{freq_days or data.modis_freq_days}D"
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    temperatures = []
    for date in dates:
        day_of_year = date.timetuple().tm_yday
        seasonal = data.seasonal_amplitude_k * np.sin(
            2 * np.pi * (day_of_year - 80) / 365
        )
        weather_noise = np.random.normal(0, data.weather_noise_std_k)
        temp_k = data.base_temp_kelvin + seasonal + weather_noise

        temperatures.append(
            {
                "date": date,
                "lst_day_kelvin": temp_k,
                "lst_night_kelvin": temp_k - data.day_night_delta_k,
                "quality_flag": 0,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    return pd.DataFrame(temperatures)


def add_celsius_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["lst_day_celsius"] = kelvin_to_celsius(result["lst_day_kelvin"])
    result["lst_night_celsius"] = kelvin_to_celsius(result["lst_night_kelvin"])
    return result


def fetch_site_thermal(config: AppConfig, lat: float, lon: float) -> pd.DataFrame:
    frame = fetch_modis_lst_data(
        lat,
        lon,
        config.site.start_date,
        config.site.end_date,
        config.data,
    )
    return add_celsius_columns(frame)


def apply_feature_thermal_effects(
    thermal: pd.DataFrame, feature: FeatureConfig
) -> pd.DataFrame:
    """Inject feature-type thermal signatures for multi-site demos."""
    result = thermal.copy()
    if feature.type == "waste_dump":
        result["lst_day_celsius"] = result["lst_day_celsius"] + np.random.normal(
            3, 1, len(result)
        )
    elif feature.type == "tailings_dam":
        recent = result["date"] > "2023-10-01"
        result.loc[recent, "lst_day_celsius"] += np.random.normal(5, 2, recent.sum())
    return result
