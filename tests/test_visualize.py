"""Tests for Plotly figure generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rasx_cluster_analyzer.config import (
    AppConfig,
    DbscanConfig,
    EmbeddingConfig,
    GridConfig,
    PathsConfig,
    PcaConfig,
    PreprocessConfig,
    TsneEmbedParams,
    UmapEmbedParams,
    VisualizeConfig,
)
from rasx_cluster_analyzer.visualize import (
    _secondary_embedding_axis_titles,
    _xrd_controls_html,
    _xrd_controls_script,
    _y_profile_for_plot,
    build_embedding_figure,
    build_xrd_profiles_figure,
    write_cluster_map_html,
)


def _minimal_cfg() -> AppConfig:
    return AppConfig(
        paths=PathsConfig(output_html=None),
        grid=GridConfig(
            theta_min=10.0,
            theta_max=30.0,
            n_points=5,
            exclude_ranges=((12.0, 14.0),),
        ),
        preprocess=PreprocessConfig(intensity_normalization="none"),
        pca=PcaConfig(n_components=2, random_state=0),
        embedding=EmbeddingConfig(
            method="tsne",
            n_components=2,
            tsne=TsneEmbedParams(
                perplexity=2.0,
                learning_rate="auto",
                random_state=0,
                max_iter=250,
            ),
            umap=UmapEmbedParams(
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                random_state=0,
            ),
        ),
        dbscan=DbscanConfig(eps=1.0, min_samples=2, clustering_space="scaled"),
        visualize=VisualizeConfig(xrd_min_panel_height_px=280),
    )


def test_secondary_embedding_axis_titles_inference() -> None:
    assert _secondary_embedding_axis_titles("PCA space used for DBSCAN") == ("PC1", "PC2")
    assert _secondary_embedding_axis_titles("t-SNE (after PCA)") == ("tSNE1", "tSNE2")
    assert _secondary_embedding_axis_titles("TSNE map") == ("tSNE1", "tSNE2")
    assert _secondary_embedding_axis_titles("UMAP (2D)") == (
        "UMAP dimension 1",
        "UMAP dimension 2",
    )
    assert _secondary_embedding_axis_titles("PCA: PC1 vs PC2 (2D embedding)") == ("PC1", "PC2")
    assert _secondary_embedding_axis_titles("Other embedding") == (
        "Embedding dimension 1",
        "Embedding dimension 2",
    )


def test_y_profile_for_plot_no_nan_gaps() -> None:
    y = np.array([0.0, 1.0, 0.0, 2.0, -0.5], dtype=np.float64)
    out = _y_profile_for_plot(y)
    assert not np.any(np.isnan(out))
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)


def test_xrd_controls_include_intensity_floor() -> None:
    html = _xrd_controls_html()
    assert 'name="y_floor"' in html
    assert "Intensity floor" in html
    assert 'placeholder="1e-6"' in html
    assert 'name="y_offset"' in html
    assert "Intensity offset" in html
    assert 'placeholder="0.001"' in html


def test_xrd_controls_script_applies_and_resets_floor_and_offset() -> None:
    script = _xrd_controls_script()
    assert "applyIntensityTransform" in script
    assert "plot.__rasxOriginalY" in script
    assert 'const yFloor = parsePositiveNumber(data.get("y_floor"));' in script
    assert 'const yOffset = parseNumber(data.get("y_offset"));' in script
    assert "const shifted = value + offsetValue;" in script
    assert "floorValue == null || shifted >= floorValue ? shifted : floorValue" in script
    assert "Plotly.restyle(plot, {y: [nextRow]}, [index]);" in script
    assert "applyIntensityTransform(plot, null, 0);" in script


def test_build_embedding_figure_smoke() -> None:
    n = 4
    pca_xy = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    tsne_xy = np.arange(n * 2, dtype=np.float64).reshape(n, 2) * 0.5
    labels = np.array([-1, 0, 0, 1], dtype=np.int64)
    meta = pl.DataFrame(
        {
            "path": [f"/tmp/f{i}.rasx" for i in range(n)],
            "stem": [f"f{i}" for i in range(n)],
            "sample": ["s"] * n,
            "index": list(range(n)),
            "x_mm": [0.0, 1.0, 2.0, 3.0],
            "y_mm": [0.0, 0.0, 1.0, 1.0],
            "theta_coverage": [1.0] * n,
        }
    )
    fig = build_embedding_figure(pca_xy, tsne_xy, labels, meta)
    assert fig.layout.title is not None
    assert fig.layout.xaxis.title.text == "PC1"
    assert fig.layout.yaxis.title.text == "PC2"
    assert fig.layout.xaxis2.title.text == "tSNE1"
    assert fig.layout.yaxis2.title.text == "tSNE2"
    n_lab = len({int(x) for x in np.unique(labels)})
    assert len(fig.data) == 2 * n_lab
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.yaxis2.scaleanchor == "x2"
    assert fig.layout.xaxis.constrain == "domain"
    assert fig.layout.yaxis.constrain == "domain"
    assert fig.layout.xaxis2.constrain == "domain"
    assert fig.layout.yaxis2.constrain == "domain"


def test_build_embedding_figure_pca2d_secondary_uses_pc_axes() -> None:
    n = 3
    pca_xy = np.arange(n * 2, dtype=np.float64).reshape(n, 2)
    labels = np.array([0, 0, 1], dtype=np.int64)
    meta = pl.DataFrame(
        {
            "path": [f"/tmp/f{i}.rasx" for i in range(n)],
            "stem": [f"f{i}" for i in range(n)],
            "sample": ["s"] * n,
            "index": list(range(n)),
            "x_mm": [0.0, 1.0, 2.0],
            "y_mm": [0.0, 0.0, 1.0],
            "theta_coverage": [1.0] * n,
        }
    )
    fig = build_embedding_figure(
        pca_xy,
        pca_xy,
        labels,
        meta,
        secondary_title="PCA space used for DBSCAN",
    )
    assert fig.layout.xaxis2.title.text == "PC1"
    assert fig.layout.yaxis2.title.text == "PC2"


def test_build_xrd_profiles_figure_smoke() -> None:
    n = 4
    n_theta = 5
    meta = pl.DataFrame(
        {
            "path": [f"/tmp/f{i}.rasx" for i in range(n)],
            "stem": [f"f{i}" for i in range(n)],
            "sample": ["s"] * n,
            "index": list(range(n)),
            "x_mm": [0.0, 1.0, 2.0, 3.0],
            "y_mm": [0.0, 0.0, 1.0, 1.0],
            "theta_coverage": [1.0] * n,
        }
    )
    twotheta = np.linspace(10.0, 30.0, n_theta, dtype=np.float64)
    intensity = np.arange(n * n_theta, dtype=np.float64).reshape(n, n_theta)
    fig = build_xrd_profiles_figure(
        "All profiles",
        meta,
        twotheta,
        intensity,
        exclude_ranges=((12.0, 14.0),),
        xrd_min_panel_height_px=280,
    )
    assert len(fig.data) == n
    assert len(fig.layout.shapes) == 1
    assert fig.layout.height >= 400
    assert all(getattr(tr, "showlegend", False) for tr in fig.data)
    assert len({tr.line.color for tr in fig.data}) == n


def test_build_xrd_profiles_figure_rejects_shape_mismatch() -> None:
    meta = pl.DataFrame(
        {
            "path": ["/tmp/a.rasx", "/tmp/b.rasx"],
            "stem": ["a", "b"],
            "sample": ["s", "s"],
            "index": [0, 1],
            "x_mm": [0.0, 1.0],
            "y_mm": [0.0, 0.0],
            "theta_coverage": [1.0, 1.0],
        }
    )
    twotheta = np.linspace(0.0, 1.0, 3)
    bad = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="match meta"):
        build_xrd_profiles_figure("Bad", meta, twotheta, bad)


def test_write_cluster_map_html_layout(tmp_path: Path) -> None:
    n = 3
    n_theta = 4
    pca_xy = np.zeros((n, 2), dtype=np.float64)
    tsne_xy = np.zeros((n, 2), dtype=np.float64)
    labels = np.array([0, 0, 1], dtype=np.int64)
    meta = pl.DataFrame(
        {
            "path": [str(tmp_path / f"f{i}.rasx") for i in range(n)],
            "stem": [f"f{i}" for i in range(n)],
            "sample": ["s"] * n,
            "index": list(range(n)),
            "x_mm": [0.0, 1.0, 2.0],
            "y_mm": [0.0, 0.0, 0.0],
            "theta_coverage": [1.0] * n,
        }
    )
    twotheta = np.linspace(10.0, 20.0, n_theta)
    intensity = np.ones((n, n_theta), dtype=np.float64)
    out = tmp_path / "out.html"
    write_cluster_map_html(
        out,
        pca_xy,
        tsne_xy,
        labels,
        meta,
        twotheta,
        intensity,
        config=_minimal_cfg(),
        rasx_dir=tmp_path,
    )
    text = out.read_text(encoding="utf-8")
    assert "Analysis metadata" in text
    assert "Files by cluster" in text
    assert "XRD Patterns" in text
    assert "All profiles" in text
    assert "cluster 0" in text
    assert "cluster 1" in text
    assert "rasx-embedding-plot" in text
    assert "rasx-xrd-plot-0" in text
    assert 'class="plotly-graph-div rasx-xrd-plot"' in text
    assert "__rasxOriginalY = [[" in text
    assert 'class="rasx-xrd-plot" class="plotly-graph-div"' not in text
    assert "plot-gap" in text
    assert "rasx-meta-sidebar" in text
    assert 'id="rasx-meta-sidebar"' in text
    assert "rasx-sidebar-open" in text
    assert "flex-wrap: nowrap" in text
    assert "justify-content: flex-start" in text
    assert "flex: 1 1 0%" in text
    assert "margin-top: 56px" in text
    assert "position: sticky" in text
    assert "height: 100vh" in text
    assert "justify-content: center" in text
    assert "width: fit-content" in text
    assert "rasx-wafer-panel" in text
    assert "Wafer map (clusters)" in text
    assert "font-size: 24px" in text
    assert ".rasx-xrd-controls h2" in text
    assert "font-size: 22px" in text
    assert "rasx-embedding-row" in text
