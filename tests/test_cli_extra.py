"""CLI error-path coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from rasx_cluster_analyzer.cli import main


def test_cli_invalid_toml_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text("not toml [[[", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        main(["--config", str(cfg), str(tmp_path)])
    assert ei.value.code == 1
