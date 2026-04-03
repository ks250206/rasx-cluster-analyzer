"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from rasx_cluster_analyzer.config import AppConfig, ConfigError, load_config


def _minimal_valid_toml() -> str:
    return """
[paths]

[grid]
theta_min = 1.0
theta_max = 2.0
n_points = 3

[pca]
n_components = 50
random_state = 0

[embedding]
method = "tsne"
n_components = 2

[embedding.tsne]
perplexity = 2.0
learning_rate = "auto"
max_iter = 250
random_state = 0

[embedding.umap]
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"
random_state = 0

[dbscan]
eps = 1.0
min_samples = 2
"""


def test_load_config_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text(_minimal_valid_toml(), encoding="utf-8")
    cfg = load_config(p)
    assert isinstance(cfg, AppConfig)
    assert cfg.grid.theta_min == 1.0
    assert cfg.grid.n_points == 3
    assert cfg.grid.exclude_ranges == ()
    assert cfg.pca.n_components == 50
    assert cfg.preprocess.intensity_normalization == "l2"
    assert cfg.dbscan.min_samples == 2
    assert cfg.visualize.xrd_min_panel_height_px == 240
    assert cfg.embedding.method == "tsne"
    assert cfg.embedding.n_components == 2


def test_theta_order_validated(tmp_path: Path) -> None:
    text = _minimal_valid_toml().replace("theta_min = 1.0", "theta_min = 3.0")
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_section(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text("[grid]\ntheta_min=0\ntheta_max=1\nn_points=2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_tsne_max_iter_minimum(tmp_path: Path) -> None:
    text = _minimal_valid_toml().replace("max_iter = 250", "max_iter = 100")
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_pca_n_components_minimum(tmp_path: Path) -> None:
    text = _minimal_valid_toml().replace("n_components = 50", "n_components = 1")
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_embedding_n_components_must_be_two(tmp_path: Path) -> None:
    text = _minimal_valid_toml().replace("n_components = 2", "n_components = 3", 1)
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_preprocess_invalid_mode(tmp_path: Path) -> None:
    extra = '\n[preprocess]\nintensity_normalization = "invalid"\n'
    p = tmp_path / "cfg.toml"
    p.write_text(_minimal_valid_toml() + extra, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_preprocess_explicit_none(tmp_path: Path) -> None:
    extra = '\n[preprocess]\nintensity_normalization = "none"\n'
    p = tmp_path / "cfg.toml"
    p.write_text(_minimal_valid_toml() + extra, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.preprocess.intensity_normalization == "none"


def test_dbscan_clustering_space_defaults_to_scaled(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text(_minimal_valid_toml(), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.dbscan.clustering_space == "scaled"


def test_dbscan_clustering_space_embedding(tmp_path: Path) -> None:
    extra = '\n[dbscan]\neps = 1.0\nmin_samples = 2\nclustering_space = "embedding"\n'
    text = _minimal_valid_toml().replace("[dbscan]\neps = 1.0\nmin_samples = 2\n", extra)
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.dbscan.clustering_space == "embedding"


def test_dbscan_clustering_space_pca(tmp_path: Path) -> None:
    extra = '\n[dbscan]\neps = 1.0\nmin_samples = 2\nclustering_space = "pca"\n'
    text = _minimal_valid_toml().replace("[dbscan]\neps = 1.0\nmin_samples = 2\n", extra)
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.dbscan.clustering_space == "pca"


def test_dbscan_clustering_space_pca2d(tmp_path: Path) -> None:
    extra = '\n[dbscan]\neps = 1.0\nmin_samples = 2\nclustering_space = "pca2d"\n'
    text = _minimal_valid_toml().replace("[dbscan]\neps = 1.0\nmin_samples = 2\n", extra)
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.dbscan.clustering_space == "pca2d"


def test_dbscan_invalid_clustering_space(tmp_path: Path) -> None:
    extra = '\n[dbscan]\neps = 1.0\nmin_samples = 2\nclustering_space = "invalid"\n'
    text = _minimal_valid_toml().replace("[dbscan]\neps = 1.0\nmin_samples = 2\n", extra)
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_embedding_method_umap(tmp_path: Path) -> None:
    text = _minimal_valid_toml().replace('method = "tsne"', 'method = "umap"')
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.embedding.method == "umap"


def test_grid_exclude_ranges_roundtrip(tmp_path: Path) -> None:
    extra = (
        "\n[grid]\n"
        "theta_min = 1.0\n"
        "theta_max = 2.0\n"
        "n_points = 3\n"
        "exclude_ranges = [[1.1, 1.2], [1.5, 1.8]]\n"
    )
    text = _minimal_valid_toml().replace(
        "[grid]\ntheta_min = 1.0\ntheta_max = 2.0\nn_points = 3\n",
        extra,
    )
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.grid.exclude_ranges == ((1.1, 1.2), (1.5, 1.8))


def test_grid_exclude_ranges_invalid_order(tmp_path: Path) -> None:
    extra = (
        "\n[grid]\ntheta_min = 1.0\ntheta_max = 2.0\nn_points = 3\nexclude_ranges = [[1.3, 1.2]]\n"
    )
    text = _minimal_valid_toml().replace(
        "[grid]\ntheta_min = 1.0\ntheta_max = 2.0\nn_points = 3\n",
        extra,
    )
    p = tmp_path / "cfg.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_visualize_xrd_min_panel_height_roundtrip(tmp_path: Path) -> None:
    extra = "\n[visualize]\nxrd_min_panel_height_px = 320\n"
    p = tmp_path / "cfg.toml"
    p.write_text(_minimal_valid_toml() + extra, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.visualize.xrd_min_panel_height_px == 320
