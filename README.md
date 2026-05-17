# Thermal Anomaly Detection at Mine Sites Identifying Safety Risks

Published: 2025-10-07  
Medium: [Thermal Anomaly Detection at Mine Sites Identifying Safety Risks](https://medium.com/@kyle-t-jones/thermal-anomaly-detection-at-mine-sites-identifying-safety-risks-35d561499694)

Companion code for the article (`article.md`). Demonstrates MODIS-style land-surface temperature monitoring, seasonal baselines, z-score anomaly detection, multi-feature risk scoring, and trend analysis at mine sites (synthetic data for reproducibility).

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run mine-thermal run
uv run mine-thermal viz-06
uv run mine-thermal viz-16
```

| Command | Output |
|---------|--------|
| `run` | CSV tables in `outputs/tables/` |
| `viz-06` | Blog 06 figures in `outputs/figures/` |
| `viz-16` | Blog 16 MODIS figures in `outputs/figures/` |

## Project layout

```
config.yaml              # site, features, detection thresholds, output paths
pyproject.toml / uv.lock
src/mine_thermal_anomaly/  # MODIS synthesis, detection, pipeline, plots, CLI
tests/
outputs/
  figures/               # generated PNGs (gitignored except .gitkeep)
  tables/                # generated CSVs (gitignored except .gitkeep)
docs/                    # blog drafts and LinkedIn posts
article.md               # Medium export
```

## Configuration

Edit `config.yaml` to change mine coordinates, feature list, `detection.threshold_sigma`, risk score cutoffs, or output filenames. Set `output.save_tables: false` or `output.save_figures: false` to run without writing artifacts.

```bash
uv run mine-thermal run --config /path/to/config.yaml
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

CI runs ruff and pytest on push/PR (see `.github/workflows/ci.yml`).

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
