"""Tests for feature matrix construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rasx_cluster_analyzer.config import GridConfig
from rasx_cluster_analyzer.features import (
    apply_theta_exclude_ranges,
    build_feature_matrix,
    interpolate_profile,
    list_rasx_files,
    theta_grid,
)
from tests.conftest import minimal_rasx_bytes


def test_theta_grid_endpoints() -> None:
    g = theta_grid(GridConfig(theta_min=0.0, theta_max=1.0, n_points=3))
    assert g.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_interpolate_profile_sorts_twotheta() -> None:
    grid = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    tt = np.array([30.0, 10.0, 20.0], dtype=np.float64)
    intensity = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    y = interpolate_profile(tt, intensity, grid)
    assert y.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_list_rasx_files_filters_suffix(tmp_path: Path) -> None:
    (tmp_path / "a.rasx").write_bytes(b"")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.rasx").write_bytes(b"")
    found = list_rasx_files(tmp_path)
    assert [p.name for p in found] == ["a.rasx"]


def test_build_feature_matrix_two_files(tmp_path: Path) -> None:
    rows_a = [(10.0, 1.0, 1.0), (20.0, 1.0, 1.0), (30.0, 1.0, 1.0)]
    rows_b = [(10.0, 2.0, 1.0), (20.0, 2.0, 1.0), (30.0, 2.0, 1.0)]
    (tmp_path / "s_01_0-0_0-0.rasx").write_bytes(minimal_rasx_bytes(rows_a))
    (tmp_path / "s_02_1-0_0-0.rasx").write_bytes(minimal_rasx_bytes(rows_b))

    grid = GridConfig(theta_min=10.0, theta_max=30.0, n_points=3)
    X, meta = build_feature_matrix(tmp_path, grid)
    assert X.shape == (2, 3)
    assert np.allclose(X[0], [1.0, 1.0, 1.0])
    assert np.allclose(X[1], [2.0, 2.0, 2.0])
    assert meta.height == 2
    assert meta["sample"].to_list() == ["s", "s"]


def test_apply_theta_exclude_ranges_masks_columns() -> None:
    grid = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    X = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=np.float64)
    masked_X, masked_grid = apply_theta_exclude_ranges(X, grid, ((19.0, 31.0),))
    np.testing.assert_allclose(masked_grid, [10.0, 40.0])
    np.testing.assert_allclose(masked_X, [[1.0, 4.0], [10.0, 40.0]])


def test_apply_theta_exclude_ranges_rejects_full_mask() -> None:
    grid = np.array([10.0, 20.0], dtype=np.float64)
    X = np.array([[1.0, 2.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="All theta grid points"):
        apply_theta_exclude_ranges(X, grid, ((9.0, 21.0),))
