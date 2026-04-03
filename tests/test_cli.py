"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rasx_cluster_analyzer.cli import main
from tests.conftest import minimal_rasx_bytes


def _write_minimal_config(path: Path) -> None:
    path.write_text(
        """
[paths]

[grid]
theta_min = 10.0
theta_max = 30.0
n_points = 5

[pca]
n_components = 5
random_state = 0

[tsne]
perplexity = 2.0
max_iter = 250
random_state = 0

[dbscan]
eps = 50.0
min_samples = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_cli_exits_on_missing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as ei:
        main([str(tmp_path / "data")])
    assert ei.value.code == 1


def test_cli_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    _write_minimal_config(cfg)
    d = tmp_path / "data"
    d.mkdir()
    rows = [(10.0, 1.0, 1.0), (20.0, 2.0, 1.0), (30.0, 3.0, 1.0)]
    (d / "s_01_0-0_0-0.rasx").write_bytes(minimal_rasx_bytes(rows))
    (d / "s_02_1-0_0-0.rasx").write_bytes(minimal_rasx_bytes([(t, v * 2, 1.0) for t, v, _ in rows]))

    main(["--config", str(cfg), str(d)])
    out = d / "cluster_map.html"
    assert out.is_file()
    assert "html" in out.read_text(encoding="utf-8").lower()
