"""Load and validate project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mine_thermal_anomaly.paths import DEFAULT_CONFIG_PATH, resolve_project_path


@dataclass(frozen=True)
class SiteConfig:
    name: str
    latitude: float
    longitude: float
    start_date: str
    end_date: str


@dataclass(frozen=True)
class DataConfig:
    modis_freq_days: int
    base_temp_kelvin: float
    seasonal_amplitude_k: float
    weather_noise_std_k: float
    day_night_delta_k: float


@dataclass(frozen=True)
class DetectionConfig:
    threshold_sigma: float
    recent_anomaly_days: int
    recent_analysis_days: int
    trend_window_days: int
    demo_injected_anomaly_celsius: float
    tailings_warming_start: str
    tailings_warming_rate_celsius: float
    tailings_warming_days: int


@dataclass(frozen=True)
class RiskConfig:
    high_score: float
    medium_score: float


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    type: str
    lat: float
    lon: float


@dataclass(frozen=True)
class OutputConfig:
    tables_dir: Path
    figures_dir: Path
    save_tables: bool
    save_figures: bool
    figure_dpi: int
    baseline_csv: str
    anomalies_csv: str
    site_analysis_csv: str
    blog06_main_figure: str
    blog06_trend_figure: str
    blog16_timeseries_figure: str
    blog16_spatial_figure: str


@dataclass(frozen=True)
class AppConfig:
    logging_level: str
    random_seed: int
    site: SiteConfig
    data: DataConfig
    detection: DetectionConfig
    risk: RiskConfig
    features: tuple[FeatureConfig, ...]
    output: OutputConfig
    font_family: str


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    site_raw = _require(raw, "site")
    data_raw = _require(raw, "data")
    detection_raw = _require(raw, "detection")
    risk_raw = _require(raw, "risk")
    output_raw = _require(raw, "output")
    style_raw = raw.get("style", {})
    features = tuple(
        FeatureConfig(
            name=str(_require(item, "name")),
            type=str(_require(item, "type")),
            lat=float(_require(item, "lat")),
            lon=float(_require(item, "lon")),
        )
        for item in _require(raw, "features")
    )
    return AppConfig(
        logging_level=raw.get("logging", {}).get("level", "INFO"),
        random_seed=int(raw.get("random_seed", 42)),
        site=SiteConfig(
            name=str(_require(site_raw, "name")),
            latitude=float(_require(site_raw, "latitude")),
            longitude=float(_require(site_raw, "longitude")),
            start_date=str(_require(site_raw, "start_date")),
            end_date=str(_require(site_raw, "end_date")),
        ),
        data=DataConfig(
            modis_freq_days=int(_require(data_raw, "modis_freq_days")),
            base_temp_kelvin=float(_require(data_raw, "base_temp_kelvin")),
            seasonal_amplitude_k=float(_require(data_raw, "seasonal_amplitude_k")),
            weather_noise_std_k=float(_require(data_raw, "weather_noise_std_k")),
            day_night_delta_k=float(_require(data_raw, "day_night_delta_k")),
        ),
        detection=DetectionConfig(
            threshold_sigma=float(_require(detection_raw, "threshold_sigma")),
            recent_anomaly_days=int(_require(detection_raw, "recent_anomaly_days")),
            recent_analysis_days=int(_require(detection_raw, "recent_analysis_days")),
            trend_window_days=int(_require(detection_raw, "trend_window_days")),
            demo_injected_anomaly_celsius=float(
                _require(detection_raw, "demo_injected_anomaly_celsius")
            ),
            tailings_warming_start=str(_require(detection_raw, "tailings_warming_start")),
            tailings_warming_rate_celsius=float(
                _require(detection_raw, "tailings_warming_rate_celsius")
            ),
            tailings_warming_days=int(_require(detection_raw, "tailings_warming_days")),
        ),
        risk=RiskConfig(
            high_score=float(_require(risk_raw, "high_score")),
            medium_score=float(_require(risk_raw, "medium_score")),
        ),
        features=features,
        output=OutputConfig(
            tables_dir=resolve_project_path(_require(output_raw, "tables_dir")),
            figures_dir=resolve_project_path(_require(output_raw, "figures_dir")),
            save_tables=bool(output_raw.get("save_tables", True)),
            save_figures=bool(output_raw.get("save_figures", True)),
            figure_dpi=int(output_raw.get("figure_dpi", 300)),
            baseline_csv=str(_require(output_raw, "baseline_csv")),
            anomalies_csv=str(_require(output_raw, "anomalies_csv")),
            site_analysis_csv=str(_require(output_raw, "site_analysis_csv")),
            blog06_main_figure=str(_require(output_raw, "blog06_main_figure")),
            blog06_trend_figure=str(_require(output_raw, "blog06_trend_figure")),
            blog16_timeseries_figure=str(_require(output_raw, "blog16_timeseries_figure")),
            blog16_spatial_figure=str(_require(output_raw, "blog16_spatial_figure")),
        ),
        font_family=str(style_raw.get("font_family", "serif")),
    )
