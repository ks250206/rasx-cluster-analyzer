"""Spectral clustering with DBSCAN on feature vectors."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from rasx_cluster_analyzer.config import DbscanConfig


def run_dbscan(X: np.ndarray, cfg: DbscanConfig) -> np.ndarray:
    """Return cluster labels (``-1`` = noise) for each row of ``X``."""
    model = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples)
    return model.fit_predict(X)
