"""Tests for configuration loading."""

from mine_thermal_anomaly.config import load_config
from mine_thermal_anomaly.paths import DEFAULT_CONFIG_PATH, FIGURES_DIR, TABLES_DIR


def test_load_default_config():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.site.name == "Golden Grove Mine"
    assert config.site.latitude == -30.5
    assert len(config.features) == 5
    assert config.detection.threshold_sigma == 2.5
    assert config.output.figures_dir == FIGURES_DIR
    assert config.output.tables_dir == TABLES_DIR
    assert config.output.save_tables is True
