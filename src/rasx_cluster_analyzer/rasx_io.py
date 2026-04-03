"""Read Rigaku rasX (zip) profile data."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

PROFILE_ZIP_PATH = "Data0/Profile0.txt"


class RasxReadError(RuntimeError):
    """Raised when a rasX archive is missing expected entries or has invalid data."""


def read_profile_arrays(rasx_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load two-theta and intensity (raw * attenuator) from ``Data0/Profile0.txt`` inside the zip.

    Returns:
        twotheta_deg, intensity as 1-D float64 arrays of equal length.
    """
    path = Path(rasx_path)
    try:
        zf = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as e:
        msg = f"Not a valid zip/rasX file: {path}"
        raise RasxReadError(msg) from e
    with zf:
        try:
            raw = zf.read(PROFILE_ZIP_PATH)
        except KeyError as e:
            msg = f"Missing {PROFILE_ZIP_PATH} in {path.name}"
            raise RasxReadError(msg) from e
    try:
        df = pl.read_csv(
            io.BytesIO(raw),
            separator="\t",
            has_header=False,
            new_columns=["twotheta", "intensity_raw", "attenuator"],
        )
    except Exception as e:
        msg = f"Failed to parse profile TSV in {path.name}"
        raise RasxReadError(msg) from e
    if df.height == 0:
        msg = f"Empty profile in {path.name}"
        raise RasxReadError(msg)
    if df.width != 3:
        msg = f"Expected 3 columns in profile, got {df.width}: {path.name}"
        raise RasxReadError(msg)
    tt = df["twotheta"].cast(pl.Float64).to_numpy()
    ir = df["intensity_raw"].cast(pl.Float64).to_numpy()
    att = df["attenuator"].cast(pl.Float64).to_numpy()
    intensity = ir * att
    return tt, intensity.astype(np.float64, copy=False)


def read_profile_frame(rasx_path: str | Path) -> pl.DataFrame:
    """Polars view of the profile (twotheta, intensity)."""
    tt, intensity = read_profile_arrays(rasx_path)
    return pl.DataFrame({"twotheta": tt, "intensity": intensity})
