"""Tests for PCA, t-SNE, UMAP, and DBSCAN wrappers."""

from __future__ import annotations

import numpy as np

from rasx_cluster_analyzer.clustering import run_dbscan
from rasx_cluster_analyzer.config import DbscanConfig, PcaConfig, TsneEmbedParams, UmapEmbedParams
from rasx_cluster_analyzer.reduction import (
    pca_for_display_and_tsne,
    run_tsne_on_pca,
    run_umap_on_pca,
)


def test_pca_for_display_and_tsne_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    pca_xy, X_pca = pca_for_display_and_tsne(X, PcaConfig(n_components=6, random_state=0))
    assert pca_xy.shape == (20, 2)
    assert X_pca.shape == (20, 6)


def test_tsne_after_pca_shape() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(15, 6))
    _, X_pca = pca_for_display_and_tsne(X, PcaConfig(n_components=5, random_state=0))
    y = run_tsne_on_pca(
        X_pca,
        TsneEmbedParams(
            perplexity=5.0,
            learning_rate="auto",
            random_state=0,
            max_iter=250,
        ),
    )
    assert y.shape == (15, 2)


def test_umap_after_pca_shape() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 12))
    _, X_pca = pca_for_display_and_tsne(X, PcaConfig(n_components=8, random_state=0))
    y = run_umap_on_pca(
        X_pca,
        UmapEmbedParams(n_neighbors=10, min_dist=0.1, metric="euclidean", random_state=0),
        n_components=2,
    )
    assert y.shape == (40, 2)


def test_umap_small_sample_count_falls_back_to_pca_projection() -> None:
    X_pca = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = run_umap_on_pca(
        X_pca,
        UmapEmbedParams(n_neighbors=10, min_dist=0.1, metric="euclidean", random_state=0),
        n_components=2,
    )
    assert y.shape == (2, 2)
    np.testing.assert_allclose(y, X_pca[:, :2])


def test_umap_single_sample_falls_back_to_zero_padded_projection() -> None:
    X_pca = np.array([[7.5]])
    y = run_umap_on_pca(
        X_pca,
        UmapEmbedParams(n_neighbors=10, min_dist=0.1, metric="euclidean", random_state=0),
        n_components=2,
    )
    assert y.shape == (1, 2)
    np.testing.assert_allclose(y, np.array([[7.5, 0.0]]))


def test_run_dbscan_labels() -> None:
    X = np.vstack(
        [
            np.zeros((5, 4)),
            np.ones((5, 4)) * 10.0,
        ]
    )
    labels = run_dbscan(X, DbscanConfig(eps=1.0, min_samples=2))
    assert labels.shape == (10,)
    assert set(np.unique(labels)) <= {-1, 0, 1}
