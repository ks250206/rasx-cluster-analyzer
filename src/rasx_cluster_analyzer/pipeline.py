"""End-to-end rasX map clustering and HTML export."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from rasx_cluster_analyzer.clustering import run_dbscan
from rasx_cluster_analyzer.config import AppConfig
from rasx_cluster_analyzer.features import (
    apply_theta_exclude_ranges,
    build_feature_matrix,
    normalize_intensity_rows,
    theta_grid,
)
from rasx_cluster_analyzer.reduction import (
    pca_for_display_and_tsne,
    run_tsne_on_pca,
    run_umap_on_pca,
)
from rasx_cluster_analyzer.visualize import write_cluster_map_html

logger = logging.getLogger(__name__)


def resolve_output_path(rasx_dir: str | Path, cfg: AppConfig) -> Path:
    """HTML output path from config or default next to the rasX directory."""
    rd = Path(rasx_dir)
    if cfg.paths.output_html:
        p = Path(cfg.paths.output_html)
        return p if p.is_absolute() else Path.cwd() / p
    return rd / "cluster_map.html"


def resolve_clustering_input(
    feature_matrix: np.ndarray,
    pca_xy: np.ndarray,
    X_pca: np.ndarray,
    embed_xy: np.ndarray,
    cfg: AppConfig,
) -> np.ndarray:
    """Select the feature space used by DBSCAN."""
    space = cfg.dbscan.clustering_space
    if space == "embedding":
        return embed_xy
    if space == "pca2d":
        return pca_xy
    if space == "pca":
        return X_pca
    return feature_matrix


def resolve_secondary_embedding(
    pca_xy: np.ndarray,
    X_pca: np.ndarray,
    cfg: AppConfig,
) -> tuple[np.ndarray, str]:
    """Return the right-hand embedding panel coordinates and title."""
    m = cfg.embedding.method
    if m == "pca2d":
        return pca_xy, "PCA: PC1 vs PC2 (2D embedding)"
    if m == "umap":
        xy = run_umap_on_pca(
            X_pca,
            cfg.embedding.umap,
            n_components=cfg.embedding.n_components,
        )
        return xy, "UMAP (after PCA)"
    xy = run_tsne_on_pca(X_pca, cfg.embedding.tsne)
    return xy, "t-SNE (after PCA)"


def run_analysis(rasx_dir: str | Path, cfg: AppConfig) -> Path:
    """Build features, scale, cluster, embed, and write Plotly HTML."""
    rd = Path(rasx_dir)
    logger.info("Loading rasX files from %s", rd.resolve())
    X_full, meta = build_feature_matrix(rd, cfg.grid)
    logger.info("Built feature matrix with shape %s", X_full.shape)

    grid_full = theta_grid(cfg.grid)
    X_plot = normalize_intensity_rows(X_full, cfg.preprocess.intensity_normalization)
    logger.info("Intensity normalization: %s", cfg.preprocess.intensity_normalization)

    X_cluster, grid_cluster = apply_theta_exclude_ranges(
        X_full,
        grid_full,
        cfg.grid.exclude_ranges,
    )
    if cfg.grid.exclude_ranges:
        logger.info(
            "Applied theta exclude ranges %s (%d features kept)",
            cfg.grid.exclude_ranges,
            grid_cluster.size,
        )
    X_cluster = normalize_intensity_rows(X_cluster, cfg.preprocess.intensity_normalization)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_cluster)

    pca_xy, X_pca = pca_for_display_and_tsne(Xs, cfg.pca)
    secondary_xy, secondary_title = resolve_secondary_embedding(pca_xy, X_pca, cfg)
    cluster_input = resolve_clustering_input(Xs, pca_xy, X_pca, secondary_xy, cfg)
    logger.info("DBSCAN clustering space: %s", cfg.dbscan.clustering_space)

    labels = run_dbscan(cluster_input, cfg.dbscan)
    logger.info("DBSCAN finished (unique labels: %s)", sorted(set(labels.tolist())))

    out = resolve_output_path(rd, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_cluster_map_html(
        out,
        pca_xy,
        secondary_xy,
        labels,
        meta,
        grid_full,
        X_plot.astype(np.float64, copy=False),
        config=cfg,
        rasx_dir=rd,
        secondary_title=secondary_title,
    )
    logger.info("Wrote %s", out.resolve())
    return out
