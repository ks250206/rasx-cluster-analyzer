"""Tests for wafer SVG map geometry and HTML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from rasx_cluster_analyzer.wafer_svg import (
    build_wafer_cluster_map_panel_html,
    get_wafer_flat_y,
    get_wafer_svg_path_d,
    is_inside_wafer,
)


def test_get_wafer_flat_y_matches_reference_geometry() -> None:
    """TS getWaferFlatY: R=50, halfFlat=16.25 → y ≈ -47.28."""
    y = get_wafer_flat_y(100.0, 32.5)
    assert y < -47.0 and y > -48.0


def test_is_inside_wafer_center_and_below_flat() -> None:
    assert is_inside_wafer(0.0, 0.0)
    assert is_inside_wafer(0.0, -47.0)
    assert not is_inside_wafer(0.0, -49.0)
    assert not is_inside_wafer(60.0, 0.0)


def test_get_wafer_svg_path_d_closed() -> None:
    d = get_wafer_svg_path_d(140.0, 140.0, 2.0, 2.0, 100.0, 32.5)
    assert d.startswith("M ")
    assert d.endswith(" Z")


def test_build_wafer_cluster_map_panel_html_smoke(tmp_path: Path) -> None:
    labels = np.array([0, 0, -1], dtype=np.int64)
    meta = pl.DataFrame(
        {
            "path": [str(tmp_path / f"p{i}.rasx") for i in range(3)],
            "x_mm": [0.0, 10.0, 0.0],
            "y_mm": [0.0, 5.0, -49.0],
        }
    )
    html = build_wafer_cluster_map_panel_html(labels, meta)
    assert "rasx-wafer-panel" in html
    assert "Wafer map (clusters)" in html
    assert "xmlns=" in html
    assert "circle" in html
    assert "aria-hidden" in html
    assert "Wafer map: measurement points" in html
    assert "outside wafer outline" in html
    assert "rasx-wafer-caption" not in html
