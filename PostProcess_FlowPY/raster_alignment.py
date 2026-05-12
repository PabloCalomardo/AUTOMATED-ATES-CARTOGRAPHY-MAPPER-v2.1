from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def read_single_band(path: str | Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    raster_path = Path(path).expanduser().resolve()
    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True)
        data = np.asarray(band.data)
        valid = ~np.asarray(band.mask)
        profile = src.profile.copy()
    return data, valid, profile


def read_single_band_on_ref_grid(path: str | Path, ref_profile: dict, resampling: Resampling = Resampling.bilinear) -> Tuple[np.ndarray, np.ndarray, dict]:
    raster_path = Path(path).expanduser().resolve()
    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True)
        data = np.asarray(band.data)
        valid = ~np.asarray(band.mask)
        profile = src.profile.copy()

        same_grid = (
            src.width == ref_profile.get("width")
            and src.height == ref_profile.get("height")
            and src.transform == ref_profile.get("transform")
            and src.crs == ref_profile.get("crs")
        )
        if same_grid:
            return data, valid, profile

        aligned = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32)
        # Cast masked array to float before filling so integer dtypes accept NaN
        source = band.astype(np.float32).filled(np.nan)
        reproject(
            source=source,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )

        aligned_profile = profile.copy()
        aligned_profile.update(
            width=ref_profile["width"],
            height=ref_profile["height"],
            transform=ref_profile["transform"],
            crs=ref_profile["crs"],
        )
        return aligned.astype(np.float32, copy=False), np.isfinite(aligned), aligned_profile
