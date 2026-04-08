#!/usr/bin/env python3
"""Compute curvature-based landforms for multiple neighborhood sizes.

The output classes follow the 9-class layout from profile (rows) and plan/tangential
(columns) curvature signs:

    plan:      convex   even   concave
    profile
    convex        7       4       1
    even          8       5       2
    concave       9       6       3

Class 0 is reserved for nodata / invalid cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
from scipy.ndimage import convolve, correlate, minimum_filter


LANDFORM_LUT = np.array(
    [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3],
    ],
    dtype=np.uint8,
)


LANDFORM_QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.40.13-Bratislava">
    <pipe>
        <rasterrenderer type="paletted" band="1" alphaBand="-1" nodataColor="" opacity="1">
            <rasterTransparency/>
            <colorPalette>
                <paletteEntry value="0" color="#000000" alpha="0" label="nodata"/>
                <paletteEntry value="1" color="#8ecad3" alpha="255" label="1 convex profile + concave plan"/>
                <paletteEntry value="2" color="#5178b8" alpha="255" label="2 even profile + concave plan"/>
                <paletteEntry value="3" color="#cc74b1" alpha="255" label="3 concave profile + concave plan"/>
                <paletteEntry value="4" color="#8fbe4f" alpha="255" label="4 convex profile + even plan"/>
                <paletteEntry value="5" color="#69bb70" alpha="255" label="5 even profile + even plan"/>
                <paletteEntry value="6" color="#a9d6cc" alpha="255" label="6 concave profile + even plan"/>
                <paletteEntry value="7" color="#ec6544" alpha="255" label="7 convex profile + convex plan"/>
                <paletteEntry value="8" color="#f29a43" alpha="255" label="8 even profile + convex plan"/>
                <paletteEntry value="9" color="#e7d24c" alpha="255" label="9 concave profile + convex plan"/>
            </colorPalette>
        </rasterrenderer>
        <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
        <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" grayscaleMode="0" saturation="0" invertColors="0"/>
        <rasterresampler maxOversampling="2"/>
    </pipe>
</qgis>
"""


def _valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(arr)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isfinite(arr)
    return np.logical_and(np.isfinite(arr), arr != nodata)


def _read_dem(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict, float, float]:
    dem_path = Path(path).expanduser().resolve()
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32, copy=False)
        nodata = src.nodata
        profile = src.profile.copy()
        transform = src.transform

    dx = float(abs(transform.a))
    dy = float(abs(transform.e))
    if dx == 0.0 or dy == 0.0:
        raise ValueError("Invalid DEM transform: pixel size cannot be zero")

    valid = _valid_mask(dem, nodata)
    return dem, valid, profile, dx, dy


def _core_valid_mask(valid: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be >= 1")
    return minimum_filter(valid.astype(np.uint8), size=window, mode="constant", cval=0) == 1


def _derivatives_3x3_closed_form(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, ...]:
    # Horn/Zevenbergen-style finite differences on a 3x3 neighborhood.
    kx = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ) / (8.0 * dx)
    ky = np.array(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    ) / (8.0 * dy)

    kxx = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ) / (dx * dx)
    kyy = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ) / (dy * dy)
    kxy = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ) / (4.0 * dx * dy)

    p = convolve(dem, kx, mode="nearest")
    q = convolve(dem, ky, mode="nearest")
    r = convolve(dem, kxx, mode="nearest")
    t = convolve(dem, kyy, mode="nearest")
    s = convolve(dem, kxy, mode="nearest")
    return p, q, r, s, t


def _quadric_fit_kernels(window: int, dx: float, dy: float) -> tuple[np.ndarray, ...]:
    if window < 3:
        raise ValueError("window must be >= 3")

    # For even windows (6, 12), coordinates are centered between the two middle cells.
    coords = np.arange(window, dtype=np.float64) - (window - 1) / 2.0
    yy, xx = np.meshgrid(coords * dy, coords * dx, indexing="ij")

    x = xx.reshape(-1)
    y = yy.reshape(-1)
    A = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])

    # beta = [a, b, c, d, e, f] with z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    pinv = np.linalg.pinv(A)

    # Convert each row of pinv into a spatial kernel so beta is computed by correlation.
    ka = pinv[0, :].reshape(window, window).astype(np.float32)
    kb = pinv[1, :].reshape(window, window).astype(np.float32)
    kc = pinv[2, :].reshape(window, window).astype(np.float32)
    kd = pinv[3, :].reshape(window, window).astype(np.float32)
    ke = pinv[4, :].reshape(window, window).astype(np.float32)
    return ka, kb, kc, kd, ke


def _derivatives_from_quadric(
    dem: np.ndarray,
    window: int,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, ...]:
    ka, kb, kc, kd, ke = _quadric_fit_kernels(window=window, dx=dx, dy=dy)

    a = correlate(dem, ka, mode="nearest")
    b = correlate(dem, kb, mode="nearest")
    c = correlate(dem, kc, mode="nearest")
    d = correlate(dem, kd, mode="nearest")
    e = correlate(dem, ke, mode="nearest")

    p = d
    q = e
    r = 2.0 * a
    s = c
    t = 2.0 * b
    return p, q, r, s, t


def _compute_profile_and_plan_curvature(
    p: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    valid: np.ndarray,
    flat_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    g2 = p * p + q * q

    profile = np.full(p.shape, np.nan, dtype=np.float32)
    plan = np.full(p.shape, np.nan, dtype=np.float32)

    curv_valid = np.logical_and(valid, g2 > flat_eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        den_profile = g2 * np.power(1.0 + g2, 1.5)
        den_plan = np.power(g2, 1.5)

        num_profile = -(r * p * p + 2.0 * s * p * q + t * q * q)
        num_plan = (r * q * q - 2.0 * s * p * q + t * p * p)

        np.divide(num_profile, den_profile, out=profile, where=curv_valid)
        np.divide(num_plan, den_plan, out=plan, where=curv_valid)

    return profile, plan


def _to_sign_class(curv: np.ndarray, threshold: float) -> np.ndarray:
    # 0=convex, 1=even, 2=concave according to thresholded sign.
    out = np.full(curv.shape, 1, dtype=np.uint8)
    out[curv > threshold] = 0
    out[curv < -threshold] = 2
    return out


def _classify_landforms(
    profile_curv: np.ndarray,
    plan_curv: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> np.ndarray:
    out = np.zeros(profile_curv.shape, dtype=np.uint8)

    pcls = _to_sign_class(profile_curv, threshold)
    ccls = _to_sign_class(plan_curv, threshold)

    out[valid] = LANDFORM_LUT[pcls[valid], ccls[valid]]
    return out


def _write_landform_raster(path: Path, data: np.ndarray, profile_ref: dict) -> Path:
    out_profile = profile_ref.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data, 1)
    return path


def _write_landform_qml_for_raster(raster_path: Path) -> Path:
    qml_path = raster_path.with_suffix(".qml")
    qml_path.write_text(LANDFORM_QML, encoding="utf-8")
    return qml_path


def _parse_windows(values: Sequence[int] | str) -> List[int]:
    if isinstance(values, str):
        parts = [p.strip() for p in values.split(",") if p.strip()]
        parsed = [int(p) for p in parts]
    else:
        parsed = [int(v) for v in values]

    if not parsed:
        raise ValueError("At least one neighborhood window size is required")
    for w in parsed:
        if w < 3:
            raise ValueError(f"Invalid window size {w}: must be >= 3")
    return parsed


def run_landforms_multiscale(
    dem_path: str | Path,
    out_dir: str | Path,
    windows: Sequence[int] = (3, 6, 12),
    curvature_threshold: float = 1e-4,
    flat_gradient_eps: float = 1e-10,
) -> List[Path]:
    dem, valid, dem_profile, dx, dy = _read_dem(dem_path)
    out_root = Path(out_dir).expanduser().resolve()

    # Fill invalid cells for convolution stability; invalids are masked out later.
    if np.any(valid):
        fill_value = float(np.nanmean(dem[valid]))
    else:
        fill_value = 0.0
    dem_filled = np.where(valid, dem, fill_value).astype(np.float32, copy=False)

    written: List[Path] = []
    for window in _parse_windows(windows):
        core_valid = _core_valid_mask(valid, window=window)

        if window == 3:
            p, q, r, s, t = _derivatives_3x3_closed_form(dem_filled, dx=dx, dy=dy)
        else:
            p, q, r, s, t = _derivatives_from_quadric(dem_filled, window=window, dx=dx, dy=dy)

        profile_curv, plan_curv = _compute_profile_and_plan_curvature(
            p=p,
            q=q,
            r=r,
            s=s,
            t=t,
            valid=core_valid,
            flat_eps=flat_gradient_eps,
        )

        landforms = _classify_landforms(
            profile_curv=profile_curv,
            plan_curv=plan_curv,
            valid=core_valid,
            threshold=curvature_threshold,
        )

        out_path = out_root / f"2_Landforms_curvature_{window}x{window}.tif"
        raster_path = _write_landform_raster(path=out_path, data=landforms, profile_ref=dem_profile)
        _write_landform_qml_for_raster(raster_path)
        written.append(raster_path)

    return written


def parse_args() -> argparse.Namespace:
    app_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compute multiscale (3x3, 6x6, 12x12 by default) landforms from DEM "
            "using profile and plan/tangential curvature"
        )
    )
    parser.add_argument(
        "--dem",
        default=str(app_root / "outputs" / "Preprocess" / "dem_filled_simple.tif"),
        help="Input DEM GeoTIFF (default: outputs/Preprocess/dem_filled_simple.tif)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(app_root / "outputs" / "Definitive_Layers"),
        help="Output directory for landform rasters (default: outputs/Definitive_Layers)",
    )
    parser.add_argument(
        "--windows",
        default="3,6,12",
        help="Comma-separated neighborhood sizes (default: 3,6,12)",
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=1e-4,
        help="Absolute threshold to separate convex/even/concave curvature signs",
    )
    parser.add_argument(
        "--flat-gradient-eps",
        type=float,
        default=1e-10,
        help="Minimum gradient magnitude squared to compute curvature",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_landforms_multiscale(
        dem_path=args.dem,
        out_dir=args.out_dir,
        windows=args.windows,
        curvature_threshold=args.curvature_threshold,
        flat_gradient_eps=args.flat_gradient_eps,
    )
    print("Done. Landforms written:")
    for p in outputs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
