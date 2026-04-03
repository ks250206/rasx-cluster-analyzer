"""Extra rasx_io coverage (frame helper and parse failures)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rasx_cluster_analyzer.rasx_io import RasxReadError, read_profile_frame
from tests.conftest import minimal_rasx_bytes


def test_read_profile_frame_columns(tmp_path: Path) -> None:
    data = minimal_rasx_bytes([(1.0, 2.0, 0.5)])
    path = tmp_path / "s_01_0-0_0-0.rasx"
    path.write_bytes(data)
    df = read_profile_frame(path)
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["twotheta", "intensity"]
    assert df.height == 1
    assert df["intensity"][0] == pytest.approx(1.0)


def test_empty_profile_raises(tmp_path: Path) -> None:
    buf = minimal_rasx_bytes([])
    path = tmp_path / "s_01_0-0_0-0.rasx"
    path.write_bytes(buf)
    with pytest.raises(RasxReadError):
        read_profile_frame(path)
