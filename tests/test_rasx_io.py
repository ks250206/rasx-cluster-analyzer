"""Tests for rasX zip profile loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from rasx_cluster_analyzer.rasx_io import PROFILE_ZIP_PATH, RasxReadError, read_profile_arrays
from tests.conftest import minimal_rasx_bytes


def test_read_profile_arrays_roundtrip(tmp_path: Path) -> None:
    data = minimal_rasx_bytes([(10.0, 2.0, 3.0), (20.0, 4.0, 0.5)])
    path = tmp_path / "s_01_0-0_0-0.rasx"
    path.write_bytes(data)

    tt, intensity = read_profile_arrays(path)
    assert tt.shape == (2,)
    assert intensity.shape == (2,)
    assert tt[0] == pytest.approx(10.0)
    assert tt[1] == pytest.approx(20.0)
    assert intensity[0] == pytest.approx(6.0)
    assert intensity[1] == pytest.approx(2.0)


def test_missing_profile_raises(tmp_path: Path) -> None:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("empty.txt", b"")
    path = tmp_path / "s_01_0-0_0-0.rasx"
    path.write_bytes(buf.getvalue())

    with pytest.raises(RasxReadError) as ei:
        read_profile_arrays(path)
    assert PROFILE_ZIP_PATH in str(ei.value)


def test_not_zip_raises(tmp_path: Path) -> None:
    path = tmp_path / "s_01_0-0_0-0.rasx"
    path.write_text("not a zip", encoding="utf-8")
    with pytest.raises(RasxReadError):
        read_profile_arrays(path)


def test_reader_preserves_file_order(tmp_path: Path) -> None:
    """Reader returns arrays in file order; feature interpolation sorts by 2θ."""
    data = minimal_rasx_bytes([(30.0, 1.0, 1.0), (10.0, 2.0, 1.0)])
    path = tmp_path / "s_01_0-0_0-0.rasx"
    path.write_bytes(data)
    tt, intensity = read_profile_arrays(path)
    assert tt.tolist() == [30.0, 10.0]
    assert intensity.tolist() == [1.0, 2.0]
