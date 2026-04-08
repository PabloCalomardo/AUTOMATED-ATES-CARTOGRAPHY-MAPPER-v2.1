#!/usr/bin/env python3
"""Slope and forest ATES 2.0-style classification from DEM and PCC density."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter, uniform_filter


def _valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(arr)
    if isinstance(nodata, float) and np.isnan(nodata):
        return np.isfinite(arr)
    return np.logical_and(np.isfinite(arr), arr != nodata)


def _window_mean(values: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    values_zeros = np.where(valid, values, 0.0).astype(np.float32, copy=False)
    weight = valid.astype(np.float32, copy=False)

    summed = uniform_filter(values_zeros, size=size, mode="nearest")
    count = uniform_filter(weight, size=size, mode="nearest")

    mean = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(summed, count, out=mean, where=count > 0.0)
    return mean


def _masked_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        out = np.where(valid, values, np.nan).astype(np.float32, copy=False)
        return out

    values_zeros = np.where(valid, values, 0.0).astype(np.float32, copy=False)
    weight = valid.astype(np.float32, copy=False)

    smoothed_sum = gaussian_filter(values_zeros, sigma=sigma, mode="nearest")
    smoothed_weight = gaussian_filter(weight, sigma=sigma, mode="nearest")

    out = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(smoothed_sum, smoothed_weight, out=out, where=smoothed_weight > 1e-6)
    return out


def classify_slope_and_forest(
    dem: np.ndarray,
    dem_valid: np.ndarray,
    transform: rasterio.Affine,
    pcc_percent: np.ndarray,
    pcc_valid: np.ndarray,
    forest_window: int = 7,
    slope_sigma: float = 1.0,
    forest_adjustment: str = "paper_pra",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if forest_window < 1 or forest_window % 2 == 0:
        raise ValueError("forest_window must be an odd integer >= 1")

    res_x = float(abs(transform.a))
    res_y = float(abs(transform.e))
    if res_x == 0.0 or res_y == 0.0:
        raise ValueError("Invalid raster transform: cell size cannot be zero")

    dem_float = dem.astype(np.float32, copy=False)
    if np.any(dem_valid):
        dem_fill = np.where(dem_valid, dem_float, np.nanmean(dem_float[dem_valid]))
    else:
        dem_fill = np.zeros_like(dem_float, dtype=np.float32)

    grad_y, grad_x = np.gradient(dem_fill, res_y, res_x)
    slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32, copy=False)
    slope_deg_smooth = _masked_gaussian(slope_deg, dem_valid, sigma=slope_sigma)

    forest_density = np.zeros_like(pcc_percent, dtype=np.float32)
    forest_density[pcc_valid] = np.clip(pcc_percent[pcc_valid], 0.0, 100.0) / 100.0
    forest_density_mean = _window_mean(forest_density, pcc_valid, size=forest_window)
    # If forest data is unavailable in a DEM-valid cell, treat it as open terrain.
    forest_density_for_rules = np.where(
        np.isfinite(forest_density_mean),
        forest_density_mean,
        0.0,
    ).astype(np.float32, copy=False)

    valid = dem_valid

    # Forest categories for PCC (percent) as in AutoATES v2.0 defaults for pcc:
    # open 0-10, sparse 10-50, moderate 50-65, dense >65.
    pcc_for_rules = forest_density_for_rules * 100.0
    forest_class = np.full(forest_density_mean.shape, -1, dtype=np.int8)
    forest_class[np.logical_and(valid, pcc_for_rules <= 10.0)] = 0
    forest_class[np.logical_and(valid, np.logical_and(pcc_for_rules > 10.0, pcc_for_rules <= 50.0))] = 1
    forest_class[np.logical_and(valid, np.logical_and(pcc_for_rules > 50.0, pcc_for_rules <= 65.0))] = 2
    forest_class[np.logical_and(valid, pcc_for_rules > 65.0)] = 3

    # Preliminary ATES class from slope using project thresholds:
    # SAT01=10, SAT12=18, SAT23=28, SAT34=39.
    # Classes are encoded as: 0 null, 1 simple, 2 challenging, 3 complex, 4 extreme.
    ates_pre = np.zeros(forest_density_mean.shape, dtype=np.uint8)
    ates_pre[np.logical_and(valid, slope_deg > 10.0)] = 1
    ates_pre[np.logical_and(valid, slope_deg > 18.0)] = 2
    ates_pre[np.logical_and(valid, slope_deg > 28.0)] = 3
    ates_pre[np.logical_and(valid, slope_deg_smooth > 39.0)] = 4

    if forest_adjustment not in {"legacy", "conservative", "paper_pra", "paper_runout"}:
        raise ValueError(
            "forest_adjustment must be one of: "
            "'legacy', 'conservative', 'paper_pra', 'paper_runout'"
        )

    # Post-forest criteria.
    # - legacy: original aggressive downgrading rules.
    # - conservative: backward-compatible alias for paper_pra.
    # - paper_pra: Table 3 mapping for PRA cells (AutoATES v2.0).
    # - paper_runout: Table 3 mapping for runout cells (AutoATES v2.0).
    ates = ates_pre.copy()
    sparse = forest_class == 1
    moderate = forest_class == 2
    dense = forest_class == 3

    if forest_adjustment == "legacy":
        ates[np.logical_and(sparse, ates_pre == 2)] = 1
        ates[np.logical_and(sparse, ates_pre == 3)] = 2
        ates[np.logical_and(sparse, ates_pre == 4)] = 3

        ates[np.logical_and(moderate, ates_pre == 2)] = 1
        ates[np.logical_and(moderate, ates_pre == 3)] = 1
        ates[np.logical_and(moderate, ates_pre == 4)] = 3

        ates[np.logical_and(dense, ates_pre == 2)] = 1
        ates[np.logical_and(dense, ates_pre == 3)] = 1
        ates[np.logical_and(dense, ates_pre == 4)] = 2
    elif forest_adjustment in {"conservative", "paper_pra"}:
        ates[np.logical_and(sparse, ates_pre == 2)] = 2
        ates[np.logical_and(sparse, ates_pre == 3)] = 3
        ates[np.logical_and(sparse, ates_pre == 4)] = 3

        ates[np.logical_and(moderate, ates_pre == 2)] = 1
        ates[np.logical_and(moderate, ates_pre == 3)] = 2
        ates[np.logical_and(moderate, ates_pre == 4)] = 3

        ates[np.logical_and(dense, ates_pre == 2)] = 1
        ates[np.logical_and(dense, ates_pre == 3)] = 2
        ates[np.logical_and(dense, ates_pre == 4)] = 2
    else:
        # Table 3 runout mapping: stronger downgrade for moderate/dense runout terrain.
        ates[np.logical_and(sparse, ates_pre == 2)] = 1
        ates[np.logical_and(sparse, ates_pre == 3)] = 2
        ates[np.logical_and(sparse, ates_pre == 4)] = 3

        ates[np.logical_and(moderate, ates_pre == 2)] = 1
        ates[np.logical_and(moderate, ates_pre == 3)] = 1
        ates[np.logical_and(moderate, ates_pre == 4)] = 3

        ates[np.logical_and(dense, ates_pre == 2)] = 1
        ates[np.logical_and(dense, ates_pre == 3)] = 1
        ates[np.logical_and(dense, ates_pre == 4)] = 2

    ates[~valid] = 0
    return ates, slope_deg_smooth, forest_density_mean, forest_class


def classify_slope_only(
    dem: np.ndarray,
    dem_valid: np.ndarray,
    transform: rasterio.Affine,
    slope_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    res_x = float(abs(transform.a))
    res_y = float(abs(transform.e))
    if res_x == 0.0 or res_y == 0.0:
        raise ValueError("Invalid raster transform: cell size cannot be zero")

    dem_float = dem.astype(np.float32, copy=False)
    if np.any(dem_valid):
        dem_fill = np.where(dem_valid, dem_float, np.nanmean(dem_float[dem_valid]))
    else:
        dem_fill = np.zeros_like(dem_float, dtype=np.float32)

    grad_y, grad_x = np.gradient(dem_fill, res_y, res_x)
    slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y))).astype(np.float32, copy=False)
    slope_deg_smooth = _masked_gaussian(slope_deg, dem_valid, sigma=slope_sigma)

    valid = dem_valid

    ates = np.zeros(dem.shape, dtype=np.uint8)
    ates[np.logical_and(valid, slope_deg > 10.0)] = 1
    ates[np.logical_and(valid, slope_deg > 18.0)] = 2
    ates[np.logical_and(valid, slope_deg > 28.0)] = 3
    ates[np.logical_and(valid, slope_deg_smooth > 39.0)] = 4
    ates[~valid] = 0
    return ates, slope_deg_smooth


def _read_dem_and_pcc(
    dem_path: Path,
    pcc_path: Path,
) -> tuple[np.ndarray, dict, float | int | None, rasterio.Affine, np.ndarray, float | int | None]:
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_profile = dem_src.profile.copy()
        dem_nodata = dem_src.nodata
        transform = dem_src.transform

    with rasterio.open(pcc_path) as pcc_src:
        pcc = pcc_src.read(1)
        pcc_nodata = pcc_src.nodata

    if dem.shape != pcc.shape:
        raise ValueError("DEM and PCC raster must have the same shape. Use aligned forest raster.")

    return dem, dem_profile, dem_nodata, transform, pcc, pcc_nodata


def run_slope_and_forest_classification(
    dem_path: str | Path,
    pcc_path: str | Path,
    out_path: str | Path,
    forest_window: int = 7,
    slope_sigma: float = 1.0,
    forest_adjustment: str = "paper_pra",
) -> Path:
    dem_path = Path(dem_path).expanduser().resolve()
    pcc_path = Path(pcc_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dem, dem_profile, dem_nodata, transform, pcc, pcc_nodata = _read_dem_and_pcc(
        dem_path=dem_path,
        pcc_path=pcc_path,
    )

    dem_valid = _valid_mask(dem, dem_nodata)
    pcc_valid = _valid_mask(pcc, pcc_nodata)

    ates, _, _, _ = classify_slope_and_forest(
        dem=dem,
        dem_valid=dem_valid,
        transform=transform,
        pcc_percent=pcc.astype(np.float32, copy=False),
        pcc_valid=pcc_valid,
        forest_window=forest_window,
        slope_sigma=slope_sigma,
        forest_adjustment=forest_adjustment,
    )

    out_profile = dem_profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(ates, 1)

    return out_path


def run_slope_only_classification(
    dem_path: str | Path,
    pcc_path: str | Path,
    out_path: str | Path,
    slope_sigma: float = 1.0,
) -> Path:
    """Compute ATES classes from slope only (ignores forest effect)."""
    dem_path = Path(dem_path).expanduser().resolve()
    pcc_path = Path(pcc_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dem, dem_profile, dem_nodata, transform, pcc, pcc_nodata = _read_dem_and_pcc(
        dem_path=dem_path,
        pcc_path=pcc_path,
    )

    dem_valid = _valid_mask(dem, dem_nodata)
    _ = _valid_mask(pcc, pcc_nodata)

    ates_slope_only, _ = classify_slope_only(
        dem=dem,
        dem_valid=dem_valid,
        transform=transform,
        slope_sigma=slope_sigma,
    )

    out_profile = dem_profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(ates_slope_only, 1)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute slope+forest ATES 2.0 classes from DEM + PCC raster")
    parser.add_argument("--dem", required=True, help="Input DEM GeoTIFF")
    parser.add_argument("--pcc", required=True, help="Input forest PCC GeoTIFF (0..100)")
    parser.add_argument("--out", required=True, help="Output ATES GeoTIFF")
    parser.add_argument("--window", type=int, default=7, help="Forest density moving window size (odd, default: 7)")
    parser.add_argument("--slope-sigma", type=float, default=1.0, help="Gaussian sigma for slope smoothing")
    parser.add_argument(
        "--forest-adjustment",
        choices=("legacy", "conservative", "paper_pra", "paper_runout"),
        default="paper_pra",
        help=(
            "Forest class downgrading profile: "
            "paper_pra (default), paper_runout, legacy, or conservative (alias of paper_pra)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = run_slope_and_forest_classification(
        dem_path=args.dem,
        pcc_path=args.pcc,
        out_path=args.out,
        forest_window=args.window,
        slope_sigma=args.slope_sigma,
        forest_adjustment=args.forest_adjustment,
    )
    print(f"Done. Slope and forest classification raster written to: {out_path}")


if __name__ == "__main__":
    main()
