"""Dimensionality reduction (PCA, t-SNE, UMAP)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rasx_cluster_analyzer.config import PcaConfig, TsneEmbedParams, UmapEmbedParams


def pca_for_display_and_tsne(X: np.ndarray, cfg: PcaConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on ``X`` and return:

    - ``pca_xy``: first two principal components (左パネル用, shape ``(n, 2)``).
    - ``X_pca``: PCA 座標、t-SNE / UMAP への入力 (shape ``(n, k)``、``k`` はデータに応じて上限).
    """
    n_samples, n_features = X.shape
    k_max = min(n_features, n_samples)
    k = min(cfg.n_components, k_max)
    if k_max >= 2:
        k = max(k, 2)
    else:
        k = max(k, 1)

    model = PCA(n_components=k, random_state=cfg.random_state)
    X_pca = model.fit_transform(X)

    if X_pca.shape[1] >= 2:
        pca_xy = X_pca[:, :2].astype(np.float64, copy=False)
    else:
        second = np.zeros((n_samples,), dtype=np.float64)
        pca_xy = np.column_stack([X_pca[:, 0], second])

    return pca_xy, X_pca.astype(np.float64, copy=False)


def run_tsne_on_pca(X_pca: np.ndarray, cfg: TsneEmbedParams) -> np.ndarray:
    """
    ``n_components=2`` の t-SNE。入力は PCA 済みの低次元特徴。
    ``init='random'`` で、入力上への二重 PCA 初期化を避ける。
    """
    n = X_pca.shape[0]
    perplexity = float(cfg.perplexity)
    if n > 1:
        perplexity = min(perplexity, float(n - 1))
        perplexity = max(perplexity, 1e-6)
    lr: str | float = cfg.learning_rate
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        init="random",
        learning_rate=lr,
    )
    return model.fit_transform(X_pca)


def run_umap_on_pca(X_pca: np.ndarray, cfg: UmapEmbedParams, *, n_components: int) -> np.ndarray:
    """UMAP on PCA-reduced features (requires ``umap-learn``)."""
    from umap import UMAP  # lazy: clearer error if dependency missing

    if n_components != 2:
        msg = "Only n_components=2 is supported for the HTML embedding panel"
        raise ValueError(msg)
    n = X_pca.shape[0]
    if n <= 2:
        # UMAP becomes numerically unstable for extremely small sample sets.
        if X_pca.shape[1] >= 2:
            return X_pca[:, :2].astype(np.float64, copy=False)
        second = np.zeros((n,), dtype=np.float64)
        return np.column_stack([X_pca[:, 0], second])

    n_neighbors = min(cfg.n_neighbors, n - 1)
    model = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=cfg.random_state,
    )
    return model.fit_transform(X_pca).astype(np.float64, copy=False)
