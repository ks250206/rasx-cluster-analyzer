"""Intensity row normalization tests."""

from __future__ import annotations

import numpy as np
import pytest

from rasx_cluster_analyzer.features import normalize_intensity_rows


def test_normalize_none_is_identity() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = normalize_intensity_rows(X, "none")
    assert np.array_equal(out, X)


def test_normalize_l2_unit_rows() -> None:
    X = np.array([[3.0, 4.0], [5.0, 0.0]], dtype=np.float64)
    out = normalize_intensity_rows(X, "l2")
    assert out.shape == X.shape
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], rtol=1e-10)


def test_normalize_max_scales_peak() -> None:
    X = np.array([[2.0, -8.0, 4.0]], dtype=np.float64)
    out = normalize_intensity_rows(X, "max")
    assert np.max(np.abs(out)) == pytest.approx(1.0)
    np.testing.assert_allclose(out[0], [0.25, -1.0, 0.5])


def test_normalize_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        normalize_intensity_rows(np.ones((1, 2)), "bad")
