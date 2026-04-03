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
    """PCA parameters (manifold への入力次元もここで決める)."""

    n_components: int
    random_state: int


@dataclass(frozen=True, slots=True)
class TsneEmbedParams:
    """t-SNE parameters under ``[embedding.tsne]``."""

    perplexity: float
    learning_rate: str | float  # "auto" or positive float
    random_state: int
    max_iter: int


@dataclass(frozen=True, slots=True)
class UmapEmbedParams:
    """UMAP parameters under ``[embedding.umap]``."""

    n_neighbors: int
    min_dist: float
    metric: str
    random_state: int


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """2D embedding for the right-hand scatter panel."""

    method: str  # "tsne" | "umap" | "pca2d"
    n_components: int
    tsne: TsneEmbedParams
    umap: UmapEmbedParams


@dataclass(frozen=True, slots=True)
class DbscanConfig:
    """DBSCAN parameters."""

    eps: float
    min_samples: int
    clustering_space: str = "scaled"


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
    embedding: EmbeddingConfig
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


def _load_tsne_learning_rate(tbl: dict[str, object], key: str) -> str | float:
    if key not in tbl:
        return "auto"
    v = tbl[key]
    if isinstance(v, str):
        if v.strip().lower() == "auto":
            return "auto"
        msg = f"{key!r} must be 'auto' or a number"
        raise ConfigError(msg)
    if isinstance(v, bool) or not isinstance(v, int | float):
        msg = f"{key!r} must be 'auto' or a number"
        raise ConfigError(msg)
    lr = float(v)
    if lr <= 0:
        msg = f"{key!r} must be positive when numeric"
        raise ConfigError(msg)
    return lr


def _req_str(tbl: dict[str, object], key: str) -> str:
    if key not in tbl:
        msg = f"Missing key {key!r}"
        raise ConfigError(msg)
    v = tbl[key]
    if not isinstance(v, str):
        msg = f"{key!r} must be a string"
        raise ConfigError(msg)
    return v


def _load_tsne_embed_params(raw: object | None) -> TsneEmbedParams:
    if raw is None:
        return TsneEmbedParams(
            perplexity=30.0,
            learning_rate="auto",
            random_state=0,
            max_iter=1000,
        )
    if not isinstance(raw, dict):
        msg = "[embedding.tsne] must be a table"
        raise ConfigError(msg)
    tbl = {str(k): v for k, v in raw.items()}
    perplexity = _req_float(tbl, "perplexity") if "perplexity" in tbl else 30.0
    learning_rate = _load_tsne_learning_rate(tbl, "learning_rate")
    random_state = _req_int(tbl, "random_state") if "random_state" in tbl else 0
    max_iter = _req_int(tbl, "max_iter") if "max_iter" in tbl else 1000
    if perplexity <= 0:
        msg = "embedding.tsne.perplexity must be positive"
        raise ConfigError(msg)
    if max_iter < 250:
        msg = "embedding.tsne.max_iter must be >= 250 (scikit-learn TSNE constraint)"
        raise ConfigError(msg)
    return TsneEmbedParams(
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
        max_iter=max_iter,
    )


def _load_umap_embed_params(raw: object | None) -> UmapEmbedParams:
    if raw is None:
        return UmapEmbedParams(
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=0,
        )
    if not isinstance(raw, dict):
        msg = "[embedding.umap] must be a table"
        raise ConfigError(msg)
    tbl = {str(k): v for k, v in raw.items()}
    n_neighbors = _req_int(tbl, "n_neighbors") if "n_neighbors" in tbl else 15
    min_dist = _req_float(tbl, "min_dist") if "min_dist" in tbl else 0.1
    metric = _req_str(tbl, "metric") if "metric" in tbl else "euclidean"
    random_state = _req_int(tbl, "random_state") if "random_state" in tbl else 0
    if n_neighbors < 2:
        msg = "embedding.umap.n_neighbors must be >= 2"
        raise ConfigError(msg)
    if min_dist < 0:
        msg = "embedding.umap.min_dist must be >= 0"
        raise ConfigError(msg)
    if not metric.strip():
        msg = "embedding.umap.metric must be a non-empty string"
        raise ConfigError(msg)
    return UmapEmbedParams(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )


def _load_embedding_config(emb_tbl: dict[str, object]) -> EmbeddingConfig:
    method_raw = emb_tbl.get("method")
    if not isinstance(method_raw, str):
        msg = "embedding.method must be a string"
        raise ConfigError(msg)
    method = method_raw.strip().lower()
    if method not in {"tsne", "umap", "pca2d"}:
        msg = "embedding.method must be 'tsne', 'umap', or 'pca2d'"
        raise ConfigError(msg)

    n_comp_raw = emb_tbl.get("n_components", 2)
    if not isinstance(n_comp_raw, int) or isinstance(n_comp_raw, bool):
        msg = "embedding.n_components must be an integer"
        raise ConfigError(msg)
    n_components = int(n_comp_raw)
    if n_components != 2:
        msg = "embedding.n_components must be 2 (current HTML layout is 2D-only)"
        raise ConfigError(msg)

    tsne = _load_tsne_embed_params(emb_tbl.get("tsne"))
    umap = _load_umap_embed_params(emb_tbl.get("umap"))

    return EmbeddingConfig(method=method, n_components=n_components, tsne=tsne, umap=umap)


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
    dbscan_tbl = _req_section(data, "dbscan")
    emb_tbl = _req_section(data, "embedding")

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

    embedding = _load_embedding_config(emb_tbl)

    dbscan = DbscanConfig(
        eps=_req_float(dbscan_tbl, "eps"),
        min_samples=_req_int(dbscan_tbl, "min_samples"),
        clustering_space=_opt_normalized_str(dbscan_tbl, "clustering_space") or "scaled",
    )
    if dbscan.eps <= 0:
        msg = "dbscan.eps must be positive"
        raise ConfigError(msg)
    if dbscan.min_samples < 1:
        msg = "dbscan.min_samples must be >= 1"
        raise ConfigError(msg)
    if dbscan.clustering_space not in {"scaled", "pca", "embedding", "pca2d"}:
        msg = "dbscan.clustering_space must be 'scaled', 'pca', 'embedding', or 'pca2d'"
        raise ConfigError(msg)

    return AppConfig(
        paths=paths,
        grid=grid,
        preprocess=preprocess,
        pca=pca,
        embedding=embedding,
        dbscan=dbscan,
        visualize=visualize,
    )
