"""Shared test helpers."""

from __future__ import annotations

import io
import zipfile


def minimal_rasx_bytes(rows: list[tuple[float, float, float]]) -> bytes:
    """Build a minimal .rasx zip with ``Data0/Profile0.txt`` TSV rows (2θ, raw, att)."""
    lines = "\n".join(f"{a}\t{b}\t{c}" for a, b, c in rows) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("Data0/Profile0.txt", lines.encode("utf-8"))
    return buf.getvalue()
