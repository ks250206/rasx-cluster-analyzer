"""Parse Rigaku map rasX filenames into sample metadata and coordinates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class FilenameParseError(ValueError):
    """Raised when a rasX filename does not match the expected pattern."""


@dataclass(frozen=True, slots=True)
class ParsedFilename:
    """Components of a map-scan rasX filename."""

    stem: str
    sample: str
    index: int
    x_mm: float
    y_mm: float


_RASX_NAME_RE = re.compile(
    r"^(.+)_(\d{2})_(-?\d+(?:-\d+)*)_(-?\d+(?:-\d+)*)$",
    re.IGNORECASE,
)


def _coord_token_to_float(token: str) -> float:
    """Rigaku-style float: minus sign only at start; other ``-`` are decimal points."""
    if token.startswith("-"):
        sign = -1.0
        body = token[1:]
    else:
        sign = 1.0
        body = token
    normalized = body.replace("-", ".")
    return sign * float(normalized)


def parse_rasx_filename(path: str | Path) -> ParsedFilename:
    """
    Parse ``<sample>_<index%02d>_<X>_<Y>.rasx`` where decimal points in X/Y are ``-``.

    Example: ``sample1_01_-42-000_5-000.rasx`` → x=-42.0, y=5.0 (per Rigaku-style tokens).
    """
    p = Path(path)
    stem = p.stem
    m = _RASX_NAME_RE.match(stem)
    if m is None:
        msg = f"Filename does not match map pattern: {p.name}"
        raise FilenameParseError(msg)
    sample, idx_s, x_tok, y_tok = m.groups()
    try:
        index = int(idx_s)
    except ValueError as e:
        msg = f"Invalid index in filename: {p.name}"
        raise FilenameParseError(msg) from e
    try:
        x_mm = _coord_token_to_float(x_tok)
        y_mm = _coord_token_to_float(y_tok)
    except ValueError as e:
        msg = f"Invalid coordinate token in filename: {p.name}"
        raise FilenameParseError(msg) from e
    return ParsedFilename(stem=stem, sample=sample, index=index, x_mm=x_mm, y_mm=y_mm)
