"""Pipeline and path resolution tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rasx_cluster_analyzer.config import (
    AppConfig,
    DbscanConfig,
    GridConfig,
    PathsConfig,
    PcaConfig,
    PreprocessConfig,
    TsneConfig,
    VisualizeConfig,
)
from rasx_cluster_analyzer.pipeline import (
    resolve_clustering_input,
    resolve_output_path,
    run_analysis,
)
from tests.conftest import minimal_rasx_bytes


def _cfg(*, output: str | None = None, clustering_space: str = "feature") -> AppConfig:
    return AppConfig(
        paths=PathsConfig(output_html=output),
        grid=GridConfig(theta_min=10.0, theta_max=30.0, n_points=5),
        preprocess=PreprocessConfig(intensity_normalization="l2"),
        pca=PcaConfig(n_components=5, random_state=0),
        tsne=TsneConfig(perplexity=2.0, max_iter=250, random_state=0),
        dbscan=DbscanConfig(eps=50.0, min_samples=2, clustering_space=clustering_space),
        visualize=VisualizeConfig(),
    )


def test_resolve_output_path_default(tmp_path: Path) -> None:
    assert resolve_output_path(tmp_path, _cfg()) == tmp_path / "cluster_map.html"


def test_resolve_output_path_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(output="rel.html")
    assert resolve_output_path(tmp_path, cfg) == tmp_path / "rel.html"


def test_run_analysis_writes_html(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    rows_a = [(10.0, 1.0, 1.0), (20.0, 1.0, 1.0), (30.0, 1.0, 1.0)]
    rows_b = [(10.0, 5.0, 1.0), (20.0, 5.0, 1.0), (30.0, 5.0, 1.0)]
    (d / "s_01_0-0_0-0.rasx").write_bytes(minimal_rasx_bytes(rows_a))
    (d / "s_02_1-0_0-0.rasx").write_bytes(minimal_rasx_bytes(rows_b))

    out = tmp_path / "map.html"
    cfg = _cfg(output=str(out))
    path = run_analysis(d, cfg)
    assert path == out.resolve()
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text.lower() or "Plotly" in text


def test_resolve_clustering_input_feature_space() -> None:
    xs = pytest.importorskip("numpy").array([[1.0, 2.0], [3.0, 4.0]])
    pca_xy = pytest.importorskip("numpy").array([[100.0, 200.0], [300.0, 400.0]])
    tsne_xy = pytest.importorskip("numpy").array([[10.0, 20.0], [30.0, 40.0]])
    selected = resolve_clustering_input(xs, pca_xy, tsne_xy, _cfg())
    assert selected is xs


def test_resolve_clustering_input_tsne_space() -> None:
    xs = pytest.importorskip("numpy").array([[1.0, 2.0], [3.0, 4.0]])
    pca_xy = pytest.importorskip("numpy").array([[100.0, 200.0], [300.0, 400.0]])
    tsne_xy = pytest.importorskip("numpy").array([[10.0, 20.0], [30.0, 40.0]])
    selected = resolve_clustering_input(xs, pca_xy, tsne_xy, _cfg(clustering_space="tsne"))
    assert selected is tsne_xy


def test_resolve_clustering_input_pca2d_space() -> None:
    xs = pytest.importorskip("numpy").array([[1.0, 2.0], [3.0, 4.0]])
    pca_xy = pytest.importorskip("numpy").array([[100.0, 200.0], [300.0, 400.0]])
    tsne_xy = pytest.importorskip("numpy").array([[10.0, 20.0], [30.0, 40.0]])
    selected = resolve_clustering_input(xs, pca_xy, tsne_xy, _cfg(clustering_space="pca2d"))
    assert selected is pca_xy
