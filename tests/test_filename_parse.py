"""Tests for rasX map filename parsing."""

from __future__ import annotations

import pytest

from rasx_cluster_analyzer.filename_parse import FilenameParseError, parse_rasx_filename


def test_parse_example_from_agents() -> None:
    p = parse_rasx_filename("sample1_01_-42-000_5-000.rasx")
    assert p.sample == "sample1"
    assert p.index == 1
    assert p.x_mm == pytest.approx(-42.0)
    assert p.y_mm == pytest.approx(5.0)


def test_parse_sample_with_underscores() -> None:
    p = parse_rasx_filename("my_batch_a_02_1-500_0-0.rasx")
    assert p.sample == "my_batch_a"
    assert p.index == 2
    assert p.x_mm == pytest.approx(1.5)
    assert p.y_mm == pytest.approx(0.0)


def test_parse_path_object() -> None:
    p = parse_rasx_filename(__import__("pathlib").Path("x_03_0-0_0-0.rasx"))
    assert p.sample == "x"
    assert p.index == 3


@pytest.mark.parametrize(
    "name",
    [
        "bad.rasx",
        "only_three_parts.rasx",
        "no_index_01_x_y.rasx",
    ],
)
def test_parse_invalid_raises(name: str) -> None:
    with pytest.raises(FilenameParseError):
        parse_rasx_filename(name)
