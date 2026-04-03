"""Load and validate ``config.toml``."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Invalid configuration file or values."""


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Output and optional path overrides."""

    output_html: str | None


@dataclass(frozen=True, slots=True)
class GridConfig:
    """Common 2θ grid for feature extraction."""

    theta_min: float
    theta_max: float
    n_points: int
    exclude_ranges: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Per-spectrum intensity normalization before column-wise scaling."""

    intensity_normalization: str  # "none" | "l2" | "max"


@dataclass(frozen=True, slots=True)
class PcaConfig:
    """PCA parameters (t-SNE への入力次元もここで決める)."""

    n_components: int
    random_state: int


@dataclass(frozen=True, slots=True)
class TsneConfig:
    """t-SNE parameters."""

    perplexity: float
    max_iter: int
    random_state: int


@dataclass(frozen=True, slots=True)
class DbscanConfig:
    """DBSCAN parameters."""

    eps: float
    min_samples: int
    clustering_space: str = "feature"


@dataclass(frozen=True, slots=True)
class VisualizeConfig:
    """Visualization layout parameters."""

    xrd_min_panel_height_px: int = 240


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Full application configuration."""

    paths: PathsConfig
    grid: GridConfig
    preprocess: PreprocessConfig
    pca: PcaConfig
    tsne: TsneConfig
    dbscan: DbscanConfig
    visualize: VisualizeConfig


def _req_section(data: dict[str, object], name: str) -> dict[str, object]:
    raw = data.get(name)
    if raw is None:
        msg = f"Missing [{name}] section in config"
        raise ConfigError(msg)
    if not isinstance(raw, dict):
        msg = f"[{name}] must be a table"
        raise ConfigError(msg)
    return {str(k): v for k, v in raw.items()}


def _req_float(tbl: dict[str, object], key: str) -> float:
    if key not in tbl:
        msg = f"Missing key {key!r}"
        raise ConfigError(msg)
    v = tbl[key]
    if isinstance(v, bool) or not isinstance(v, int | float):
        msg = f"{key!r} must be a number"
        raise ConfigError(msg)
    return float(v)


def _req_int(tbl: dict[str, object], key: str) -> int:
    if key not in tbl:
        msg = f"Missing key {key!r}"
        raise ConfigError(msg)
    v = tbl[key]
    if not isinstance(v, int) or isinstance(v, bool):
        msg = f"{key!r} must be an integer"
        raise ConfigError(msg)
    return int(v)


def _opt_str(tbl: dict[str, object], key: str) -> str | None:
    if key not in tbl:
        return None
    v = tbl[key]
    if v is None:
        return None
    if not isinstance(v, str):
        msg = f"{key!r} must be a string or null"
        raise ConfigError(msg)
    return v


def _opt_normalized_str(tbl: dict[str, object], key: str) -> str | None:
    raw = _opt_str(tbl, key)
    if raw is None:
        return None
    return raw.strip().lower()


def _opt_float_pair_ranges(
    tbl: dict[str, object],
    key: str,
) -> tuple[tuple[float, float], ...]:
    if key not in tbl:
        return ()
    raw = tbl[key]
    if not isinstance(raw, list):
        msg = f"{key!r} must be an array of [start, end] pairs"
        raise ConfigError(msg)
    ranges: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            msg = f"{key!r} must contain only [start, end] pairs"
            raise ConfigError(msg)
        start_raw, end_raw = item
        if isinstance(start_raw, bool) or not isinstance(start_raw, int | float):
            msg = f"{key!r} range start must be a number"
            raise ConfigError(msg)
        if isinstance(end_raw, bool) or not isinstance(end_raw, int | float):
            msg = f"{key!r} range end must be a number"
            raise ConfigError(msg)
        start = float(start_raw)
        end = float(end_raw)
        if start >= end:
            msg = f"{key!r} ranges must satisfy start < end"
            raise ConfigError(msg)
        ranges.append((start, end))
    return tuple(ranges)


def _load_preprocess_config(data: dict[str, object]) -> PreprocessConfig:
    """Optional ``[preprocess]``; default intensity normalization is ``l2``."""
    raw = data.get("preprocess")
    if raw is None:
        return PreprocessConfig(intensity_normalization="l2")
    if not isinstance(raw, dict):
        msg = "[preprocess] must be a table"
        raise ConfigError(msg)
    tbl = {str(k): v for k, v in raw.items()}
    norm_raw = _opt_normalized_str(tbl, "intensity_normalization")
    if norm_raw is None:
        return PreprocessConfig(intensity_normalization="l2")
    if norm_raw not in {"none", "l2", "max"}:
        msg = "preprocess.intensity_normalization must be 'none', 'l2', or 'max'"
        raise ConfigError(msg)
    return PreprocessConfig(intensity_normalization=norm_raw)


def _load_visualize_config(data: dict[str, object]) -> VisualizeConfig:
    """Optional ``[visualize]`` section for layout tuning."""
    raw = data.get("visualize")
    if raw is None:
        return VisualizeConfig()
    if not isinstance(raw, dict):
        msg = "[visualize] must be a table"
        raise ConfigError(msg)
    tbl = {str(k): v for k, v in raw.items()}
    min_height = tbl.get("xrd_min_panel_height_px", 240)
    if not isinstance(min_height, int) or isinstance(min_height, bool):
        msg = "visualize.xrd_min_panel_height_px must be an integer"
        raise ConfigError(msg)
    if min_height < 120:
        msg = "visualize.xrd_min_panel_height_px must be >= 120"
        raise ConfigError(msg)
    return VisualizeConfig(xrd_min_panel_height_px=min_height)


def load_config(path: str | Path) -> AppConfig:
    """Parse TOML configuration from ``path``."""
    p = Path(path)
    try:
        raw_bytes = p.read_bytes()
    except OSError as e:
        msg = f"Cannot read config file: {p}"
        raise ConfigError(msg) from e
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except tomllib.TOMLDecodeError as e:
        msg = f"Invalid TOML in {p}"
        raise ConfigError(msg) from e
    if not isinstance(data, dict):
        msg = "Config root must be a table"
        raise ConfigError(msg)

    paths_tbl = _req_section(data, "paths")
    grid_tbl = _req_section(data, "grid")
    pca_tbl = _req_section(data, "pca")
    tsne_tbl = _req_section(data, "tsne")
    dbscan_tbl = _req_section(data, "dbscan")

    paths = PathsConfig(output_html=_opt_str(paths_tbl, "output_html"))

    grid = GridConfig(
        theta_min=_req_float(grid_tbl, "theta_min"),
        theta_max=_req_float(grid_tbl, "theta_max"),
        n_points=_req_int(grid_tbl, "n_points"),
        exclude_ranges=_opt_float_pair_ranges(grid_tbl, "exclude_ranges"),
    )
    if grid.theta_min >= grid.theta_max:
        msg = "grid.theta_min must be < grid.theta_max"
        raise ConfigError(msg)
    if grid.n_points < 2:
        msg = "grid.n_points must be >= 2"
        raise ConfigError(msg)

    preprocess = _load_preprocess_config(data)
    visualize = _load_visualize_config(data)

    pca = PcaConfig(
        n_components=_req_int(pca_tbl, "n_components"),
        random_state=_req_int(pca_tbl, "random_state"),
    )
    if pca.n_components < 2:
        msg = "pca.n_components must be >= 2"
        raise ConfigError(msg)

    tsne = TsneConfig(
        perplexity=_req_float(tsne_tbl, "perplexity"),
        max_iter=_req_int(tsne_tbl, "max_iter"),
        random_state=_req_int(tsne_tbl, "random_state"),
    )
    if tsne.perplexity <= 0:
        msg = "tsne.perplexity must be positive"
        raise ConfigError(msg)
    if tsne.max_iter < 250:
        msg = "tsne.max_iter must be >= 250 (scikit-learn TSNE constraint)"
        raise ConfigError(msg)

    dbscan = DbscanConfig(
        eps=_req_float(dbscan_tbl, "eps"),
        min_samples=_req_int(dbscan_tbl, "min_samples"),
        clustering_space=_opt_normalized_str(dbscan_tbl, "clustering_space") or "feature",
    )
    if dbscan.eps <= 0:
        msg = "dbscan.eps must be positive"
        raise ConfigError(msg)
    if dbscan.min_samples < 1:
        msg = "dbscan.min_samples must be >= 1"
        raise ConfigError(msg)
    if dbscan.clustering_space not in {"feature", "tsne", "pca2d"}:
        msg = "dbscan.clustering_space must be 'feature', 'tsne', or 'pca2d'"
        raise ConfigError(msg)

    return AppConfig(
        paths=paths,
        grid=grid,
        preprocess=preprocess,
        pca=pca,
        tsne=tsne,
        dbscan=dbscan,
        visualize=visualize,
    )
