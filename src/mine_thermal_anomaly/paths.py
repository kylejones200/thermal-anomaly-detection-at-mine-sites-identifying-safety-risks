"""Repository root and default paths for config and outputs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"


def resolve_project_path(rel: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    path = Path(rel)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
