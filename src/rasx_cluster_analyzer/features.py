"""Build aligned intensity feature matrices from rasX map files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from rasx_cluster_analyzer.config import GridConfig
from rasx_cluster_analyzer.filename_parse import parse_rasx_filename
from rasx_cluster_analyzer.rasx_io import read_profile_arrays

logger = logging.getLogger(__name__)


def list_rasx_files(directory: str | Path) -> list[Path]:
    """Return sorted ``.rasx`` files directly under ``directory`` (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        msg = f"Not a directory: {d}"
        raise NotADirectoryError(msg)
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".rasx")
    return files


def theta_grid(cfg: GridConfig) -> np.ndarray:
    """Inclusive linear grid from ``theta_min`` to ``theta_max`` with ``n_points`` samples."""
    return np.linspace(cfg.theta_min, cfg.theta_max, cfg.n_points, dtype=np.float64)


def apply_theta_exclude_ranges(
    X: np.ndarray,
    grid: np.ndarray,
    exclude_ranges: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop feature columns whose theta positions fall inside any excluded range."""
    if X.shape[1] != grid.shape[0]:
        msg = "Feature matrix width must match theta grid length"
        raise ValueError(msg)
    if not exclude_ranges:
        return X, grid

    keep_mask = np.ones(grid.shape[0], dtype=bool)
    for start, end in exclude_ranges:
        keep_mask &= ~((grid >= start) & (grid <= end))

    if not np.any(keep_mask):
        msg = "All theta grid points were excluded by grid.exclude_ranges"
        raise ValueError(msg)

    return X[:, keep_mask], grid[keep_mask]


def _coverage_fraction(twotheta: np.ndarray, grid: np.ndarray) -> float:
    if twotheta.size == 0:
        return 0.0
    t_min = float(np.min(twotheta))
    t_max = float(np.max(twotheta))
    g_min = float(np.min(grid))
    g_max = float(np.max(grid))
    overlap_min = max(t_min, g_min)
    overlap_max = min(t_max, g_max)
    if overlap_max <= overlap_min:
        return 0.0
    span = g_max - g_min
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (overlap_max - overlap_min) / span))


def interpolate_profile(
    twotheta: np.ndarray, intensity: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Linearly interpolate intensity onto ``grid`` (``twotheta`` sorted internally)."""
    if twotheta.shape != intensity.shape:
        msg = "twotheta and intensity must have the same shape"
        raise ValueError(msg)
    order = np.argsort(twotheta)
    tt = twotheta[order].astype(np.float64, copy=False)
    ii = intensity[order].astype(np.float64, copy=False)
    return np.interp(grid, tt, ii).astype(np.float64, copy=False)


def build_feature_matrix(
    rasx_dir: str | Path,
    grid_cfg: GridConfig,
) -> tuple[np.ndarray, pl.DataFrame]:
    """
    Load all map rasX files, parse filenames, and build ``(n_files, n_theta)`` feature matrix.

    Returns:
        ``X``, ``meta`` where rows align; ``meta`` includes path, coordinates, and theta coverage.
    """
    paths = list_rasx_files(rasx_dir)
    if not paths:
        msg = f"No .rasx files in {Path(rasx_dir).resolve()}"
        raise ValueError(msg)

    grid = theta_grid(grid_cfg)
    rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    for path in paths:
        parsed = parse_rasx_filename(path)
        tt, intensity = read_profile_arrays(path)
        cov = _coverage_fraction(tt, grid)
        if cov < 1.0:
            logger.warning(
                "Theta grid only partially covered for %s (coverage ~%.3f)",
                path.name,
                cov,
            )
        row = interpolate_profile(tt, intensity, grid)
        rows.append(row)
        records.append(
            {
                "path": str(path.resolve()),
                "stem": parsed.stem,
                "sample": parsed.sample,
                "index": parsed.index,
                "x_mm": parsed.x_mm,
                "y_mm": parsed.y_mm,
                "theta_coverage": cov,
            }
        )

    X = np.vstack(rows)
    meta = pl.DataFrame(records)
    return X, meta


def normalize_intensity_rows(X: np.ndarray, mode: str) -> np.ndarray:
    """
    スペクトル（行）ごとに強度を規格化する。``StandardScaler`` より前に適用する。

    - ``none``: そのまま
    - ``l2``: 各行をユークリッドノルムで除算（形状比較向け）
    - ``max``: 各行を絶対値最大で除算（最大強度を 1 に）
    """
    if mode == "none":
        return X
    out = X.astype(np.float64, copy=True)
    if mode == "l2":
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.maximum(norms, np.finfo(np.float64).eps)
        out /= norms
        return out
    if mode == "max":
        m = np.max(np.abs(out), axis=1, keepdims=True)
        m = np.maximum(m, np.finfo(np.float64).eps)
        out /= m
        return out
    msg = f"Unknown intensity normalization mode: {mode!r}"
    raise ValueError(msg)
