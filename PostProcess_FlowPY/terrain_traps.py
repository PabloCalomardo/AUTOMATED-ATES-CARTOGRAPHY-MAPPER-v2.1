#!/usr/bin/env python3
"""Detect avalanche terrain traps from available APP_ATES layers.

Outputs:
- Five independent binary rasters (uint8 0/1): Trees, Cliffs, Gullies, RoadCuts, Lakes.
- Two additional bitmask rasters:
  - Trauma amplifiers bitmask: Trees=1, Cliffs=2
  - Burial amplifiers bitmask: Gullies=1, RoadCuts=2, Lakes=4

Terrain-trap pixels are only produced where avalanche impact exists (z_delta > 0).
Gullies use an SPI-based approach parameterized as:
    SPI = A^m * S^n
where A is drainage area (D8 flow accumulation * pixel area) and
S is local slope gradient (tan(slope)).
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from scipy.ndimage import maximum_filter, uniform_filter

from PostProcess_FlowPY.raster_alignment import read_single_band, read_single_band_on_ref_grid


TRAUMA_TREE = np.uint8(1)
TRAUMA_CLIFF = np.uint8(2)

BURIAL_GULLY = np.uint8(1)
BURIAL_ROADCUT = np.uint8(2)
BURIAL_LAKE = np.uint8(4)


GULLY_PRESETS = {
    "conservative": {
        "energy_threshold": 0.30,
        "spi_percentile": 93.0,
        "min_drainage_area_m2": 8000.0,
        "min_slope_deg": 15.0,
        "max_slope_deg": 42.0,
    },
    "balanced": {
        "energy_threshold": 0.22,
        "spi_percentile": 88.0,
        "min_drainage_area_m2": 4000.0,
        "min_slope_deg": 13.0,
        "max_slope_deg": 48.0,
    },
    "aggressive": {
        "energy_threshold": 0.15,
        "spi_percentile": 82.0,
        "min_drainage_area_m2": 2000.0,
        "min_slope_deg": 10.0,
        "max_slope_deg": 52.0,
    },
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _check_alignment(profile_a: dict, profile_b: dict, label_a: str, label_b: str) -> None:
    checks = [
        ("width", profile_a.get("width"), profile_b.get("width")),
        ("height", profile_a.get("height"), profile_b.get("height")),
        ("transform", profile_a.get("transform"), profile_b.get("transform")),
        ("crs", profile_a.get("crs"), profile_b.get("crs")),
    ]
    mismatches = [name for name, a, b in checks if a != b]
    if mismatches:
        raise ValueError(
            f"Rasters are not aligned ({label_a} vs {label_b}): {', '.join(mismatches)}"
        )


def _dem_slope_deg(dem: np.ndarray, valid: np.ndarray, transform) -> np.ndarray:
    dx = float(abs(transform.a))
    dy = float(abs(transform.e))
    if dx == 0.0 or dy == 0.0:
        raise ValueError("Invalid DEM transform: pixel size cannot be zero")

    fill_value = float(np.nanmean(dem[valid])) if np.any(valid) else 0.0
    dem_filled = np.where(valid, dem, fill_value).astype(np.float32, copy=False)

    gy, gx = np.gradient(dem_filled, dy, dx)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    return np.degrees(slope_rad).astype(np.float32, copy=False)


def _local_relief(dem: np.ndarray, valid: np.ndarray, window: int = 7) -> np.ndarray:
    fill_value = float(np.nanmean(dem[valid])) if np.any(valid) else 0.0
    dem_filled = np.where(valid, dem, fill_value).astype(np.float32, copy=False)

    mean = uniform_filter(dem_filled, size=window, mode="nearest")
    mean_sq = uniform_filter(dem_filled * dem_filled, size=window, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32, copy=False)


def _topographic_position_index(dem: np.ndarray, valid: np.ndarray, window: int = 21) -> np.ndarray:
    fill_value = float(np.nanmean(dem[valid])) if np.any(valid) else 0.0
    dem_filled = np.where(valid, dem, fill_value).astype(np.float32, copy=False)
    neighborhood_mean = uniform_filter(dem_filled, size=window, mode="nearest")
    return (dem_filled - neighborhood_mean).astype(np.float32, copy=False)


def _latest_result_dir(basin_dir: Path) -> Optional[Path]:
    res_dirs = [p for p in basin_dir.glob("res_*") if p.is_dir()]
    if not res_dirs:
        return None
    return max(res_dirs, key=lambda p: p.stat().st_mtime)


def _collect_latest_zdelta_max(flowpy_root: Path, ref_profile: dict) -> tuple[Optional[np.ndarray], np.ndarray]:
    zdelta_max: Optional[np.ndarray] = None
    zdelta_valid = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=bool)

    basin_dirs = sorted([p for p in flowpy_root.iterdir() if p.is_dir() and p.name.lower().startswith("pra_basin_")])
    for basin_dir in basin_dirs:
        latest = _latest_result_dir(basin_dir)
        if latest is None:
            continue

        z_delta_path = latest / "z_delta.tif"
        if not z_delta_path.exists():
            continue

        z_delta, valid, profile = read_single_band_on_ref_grid(z_delta_path, ref_profile)
        _check_alignment(ref_profile, profile, "dem", str(z_delta_path))

        z_delta = z_delta.astype(np.float32, copy=False)
        if zdelta_max is None:
            zdelta_max = np.full(z_delta.shape, np.nan, dtype=np.float32)

        valid_pos = np.logical_and(valid, np.isfinite(z_delta))
        if np.any(valid_pos):
            current = zdelta_max[valid_pos]
            new_vals = z_delta[valid_pos]
            both_finite = np.isfinite(current)
            merged = new_vals.copy()
            merged[both_finite] = np.maximum(current[both_finite], new_vals[both_finite])
            zdelta_max[valid_pos] = merged
            zdelta_valid[valid_pos] = True

    return zdelta_max, zdelta_valid


def _normalize_energy_0_1(zdelta: Optional[np.ndarray], valid: np.ndarray) -> np.ndarray:
    energy = np.zeros(valid.shape, dtype=np.float32)
    if zdelta is None:
        return energy

    valid_energy = np.logical_and(valid, np.isfinite(zdelta))
    if not np.any(valid_energy):
        return energy

    vals = zdelta[valid_energy]
    p05 = float(np.percentile(vals, 5))
    p95 = float(np.percentile(vals, 95))

    if np.isclose(p95, p05):
        energy[valid_energy] = 0.0
        return energy

    scaled = (zdelta[valid_energy] - p05) / (p95 - p05)
    energy[valid_energy] = np.clip(scaled, 0.0, 1.0)
    return energy


def _load_landforms(path: Optional[Path], ref_profile: dict, shape: Tuple[int, int]) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(shape, dtype=np.uint8)

    arr, valid, profile = read_single_band(path)
    _check_alignment(ref_profile, profile, "dem", str(path))
    out = np.zeros(shape, dtype=np.uint8)
    out[valid] = arr[valid].astype(np.uint8, copy=False)
    return out


def _select_landform_path(definitive_layers_dir: Path, preferred: Sequence[str]) -> Optional[Path]:
    for name in preferred:
        candidate = definitive_layers_dir / name
        if candidate.exists():
            return candidate
    return None


def _d8_receivers(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return flattened receiver index per cell for D8 steepest descent, -1 if sink/invalid."""
    h, w = dem.shape
    size = h * w

    dem_f = dem.astype(np.float32, copy=False)
    valid_f = valid.astype(bool, copy=False)

    row_idx, col_idx = np.indices((h, w), dtype=np.int32)
    flat_idx = (row_idx * w + col_idx).astype(np.int64)

    max_drop = np.full((h, w), -np.inf, dtype=np.float32)
    receiver = np.full((h, w), -1, dtype=np.int64)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for dy, dx in neighbors:
        src_r0 = max(0, -dy)
        src_r1 = h - max(0, dy)
        src_c0 = max(0, -dx)
        src_c1 = w - max(0, dx)

        nbr_r0 = max(0, dy)
        nbr_r1 = h - max(0, -dy)
        nbr_c0 = max(0, dx)
        nbr_c1 = w - max(0, -dx)

        drop = np.full((h, w), -np.inf, dtype=np.float32)
        nbr_valid = np.zeros((h, w), dtype=bool)
        nbr_flat = np.full((h, w), -1, dtype=np.int64)

        src_slice = (slice(src_r0, src_r1), slice(src_c0, src_c1))
        nbr_slice = (slice(nbr_r0, nbr_r1), slice(nbr_c0, nbr_c1))

        dem_src = dem_f[src_slice]
        dem_nbr = dem_f[nbr_slice]
        valid_src = valid_f[src_slice]
        valid_nbr = valid_f[nbr_slice]

        drop[src_slice] = dem_src - dem_nbr
        nbr_valid[src_slice] = valid_nbr
        nbr_flat[src_slice] = flat_idx[nbr_slice]

        better = valid_f & nbr_valid & (drop > 0.0) & (drop > max_drop)
        max_drop[better] = drop[better]
        receiver[better] = nbr_flat[better]

    receiver_flat = receiver.reshape(size)
    valid_flat = valid_f.reshape(size)
    receiver_flat[~valid_flat] = -1
    return receiver_flat


def _d8_flow_accumulation_cells(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute D8 flow accumulation in number of cells (including self)."""
    receiver = _d8_receivers(dem=dem, valid=valid)
    size = receiver.size

    valid_flat = valid.reshape(size)
    indegree = np.zeros(size, dtype=np.int32)

    has_receiver = receiver >= 0
    rec_targets = receiver[has_receiver]
    np.add.at(indegree, rec_targets, 1)

    accum = np.zeros(size, dtype=np.float64)
    accum[valid_flat] = 1.0

    queue = deque(np.where(valid_flat & (indegree == 0))[0].tolist())
    while queue:
        cell = queue.popleft()
        rcv = int(receiver[cell])
        if rcv >= 0:
            accum[rcv] += accum[cell]
            indegree[rcv] -= 1
            if indegree[rcv] == 0:
                queue.append(rcv)

    return accum.reshape(dem.shape).astype(np.float32, copy=False)


def _stream_power_index(
    dem: np.ndarray,
    valid: np.ndarray,
    transform,
    slope_deg: np.ndarray,
    m_exp: float,
    n_exp: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return SPI and drainage area using SPI = A^m * S^n."""
    pixel_width = float(abs(transform.a))
    pixel_height = float(abs(transform.e))
    pixel_area = pixel_width * pixel_height

    if pixel_width == 0.0 or pixel_height == 0.0:
        raise ValueError("Invalid DEM transform: pixel size cannot be zero")

    flow_acc_cells = _d8_flow_accumulation_cells(dem=dem, valid=valid)
    drainage_area = flow_acc_cells * np.float32(pixel_area)

    slope_rad = np.deg2rad(slope_deg)
    slope_grad = np.tan(slope_rad)
    slope_grad = np.clip(slope_grad, 0.0, None).astype(np.float32, copy=False)

    spi = np.zeros(dem.shape, dtype=np.float32)
    safe_a = np.maximum(drainage_area, 1e-6)
    safe_s = np.maximum(slope_grad, 1e-6)

    spi[valid] = np.power(safe_a[valid], m_exp) * np.power(safe_s[valid], n_exp)
    return spi, drainage_area


def _threshold_from_absolute_or_percentile(
    values: np.ndarray,
    valid_mask: np.ndarray,
    absolute_threshold: Optional[float],
    percentile: float,
) -> float:
    if absolute_threshold is not None and absolute_threshold > 0.0:
        return float(absolute_threshold)

    sample = values[valid_mask]
    if sample.size == 0:
        return 0.0
    return float(np.percentile(sample, percentile))


def _compute_gully_mask(
    terrain_valid: np.ndarray,
    slope: np.ndarray,
    drainage_area_m2: np.ndarray,
    spi: np.ndarray,
    concave_plan_like: np.ndarray,
    energy: np.ndarray,
    relief: np.ndarray,
    min_slope_deg: float,
    max_slope_deg: float,
    min_drainage_area_m2: float,
    spi_threshold: Optional[float],
    spi_percentile: float,
    energy_threshold: float,
) -> Tuple[np.ndarray, float]:
    gully_domain = np.logical_and.reduce(
        [
            terrain_valid,
            slope >= min_slope_deg,
            slope <= max_slope_deg,
            drainage_area_m2 >= min_drainage_area_m2,
        ]
    )

    gully_spi_cut = _threshold_from_absolute_or_percentile(
        values=spi,
        valid_mask=gully_domain,
        absolute_threshold=spi_threshold,
        percentile=spi_percentile,
    )

    gullies = np.logical_and.reduce(
        [
            gully_domain,
            concave_plan_like,
            spi >= gully_spi_cut,
            np.logical_or(energy >= energy_threshold, relief >= 1.0),
        ]
    )
    return gullies, gully_spi_cut


def detect_terrain_traps(
    dem_path: str | Path,
    forest_path: str | Path,
    definitive_layers_dir: str | Path,
    flowpy_root: str | Path,
    out_dir: str | Path,
    forest_tree_threshold: float = 35.0,
    energy_trauma_threshold: float = 0.35,
    gully_energy_threshold: float = 0.22,
    gully_spi_m: float = 1.0,
    gully_spi_n: float = 1.0,
    gully_spi_threshold: Optional[float] = None,
    gully_spi_percentile: float = 88.0,
    gully_min_drainage_area_m2: float = 4000.0,
    gully_min_slope_deg: float = 13.0,
    gully_max_slope_deg: float = 48.0,
    lake_max_slope_deg: float = 6.0,
    lake_tpi_threshold: float = -1.8,
    lake_max_spi_threshold: Optional[float] = None,
    lake_max_spi_percentile: float = 35.0,
) -> List[Path]:
    dem, dem_valid, dem_profile = read_single_band(dem_path)
    dem = dem.astype(np.float32, copy=False)

    forest, forest_valid, forest_profile = read_single_band(forest_path)
    forest = forest.astype(np.float32, copy=False)
    _check_alignment(dem_profile, forest_profile, "dem", "forest")

    definitive = Path(definitive_layers_dir).expanduser().resolve()
    flowpy = Path(flowpy_root).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    _ensure_dir(out_root)
    out_individual = out_root / "3_TerrainTraps"
    _ensure_dir(out_individual)

    landform_path = _select_landform_path(
        definitive_layers_dir=definitive,
        preferred=(
            "2_Landforms/2_Landforms_curvature_15x15.tif",
            "2_Landforms/2_Landforms_curvature_10x10.tif",
            "2_Landforms/2_Landforms_curvature_20x20.tif",
            "2_Landforms/2_Landforms_curvature_5x5.tif",
            "2_Landforms/2_Landforms_curvature_25x25.tif",
            "2_Landforms/2_Landforms_curvature_30x30.tif",
            "2_Landforms_curvature_15x15.tif",
            "2_Landforms_curvature_10x10.tif",
            "2_Landforms_curvature_20x20.tif",
            "2_Landforms_curvature_5x5.tif",
            "2_Landforms_curvature_25x25.tif",
            "2_Landforms_curvature_30x30.tif",
            "Landforms_curvature_15x15.tif",
            "Landforms_curvature_10x10.tif",
            "Landforms_curvature_20x20.tif",
            "Landforms_curvature_5x5.tif",
            "Landforms_curvature_25x25.tif",
            "Landforms_curvature_30x30.tif",
            "2_Landforms_curvature_12x12.tif",
            "2_Landforms_curvature_6x6.tif",
            "2_Landforms_curvature_3x3.tif",
            "Landforms_curvature_3x3.tif",
            "Landforms_curvature_6x6.tif",
            "Landforms_curvature_12x12.tif",
        ),
    )
    landforms = _load_landforms(landform_path, dem_profile, dem.shape)

    slope = _dem_slope_deg(dem=dem, valid=dem_valid, transform=dem_profile["transform"])
    relief = _local_relief(dem=dem, valid=dem_valid, window=7)
    tpi = _topographic_position_index(dem=dem, valid=dem_valid, window=21)

    zdelta_max, zdelta_valid = _collect_latest_zdelta_max(flowpy_root=flowpy, ref_profile=dem_profile)
    energy = _normalize_energy_0_1(zdelta=zdelta_max, valid=np.logical_and(dem_valid, zdelta_valid))

    if zdelta_max is None:
        avalanche_zone = np.zeros(dem.shape, dtype=bool)
    else:
        avalanche_zone = np.logical_and.reduce(
            [
                dem_valid,
                zdelta_valid,
                np.isfinite(zdelta_max),
                zdelta_max > 0.0,
            ]
        )

    terrain_valid = np.logical_and(dem_valid, avalanche_zone)

    spi, drainage_area_m2 = _stream_power_index(
        dem=dem,
        valid=dem_valid,
        transform=dem_profile["transform"],
        slope_deg=slope,
        m_exp=gully_spi_m,
        n_exp=gully_spi_n,
    )

    concave_plan_like = np.isin(landforms, np.array([1, 2, 3], dtype=np.uint8))
    benches_like = np.isin(landforms, np.array([2, 3, 5, 6], dtype=np.uint8))

    trees = np.logical_and.reduce(
        [
            terrain_valid,
            forest_valid,
            forest >= forest_tree_threshold,
            energy >= energy_trauma_threshold,
            slope >= 18.0,
        ]
    )

    cliffs = np.logical_and.reduce(
        [
            terrain_valid,
            slope >= 40.0,
            relief >= 2.0,
            energy >= energy_trauma_threshold,
        ]
    )

    gullies, gully_spi_cut = _compute_gully_mask(
        terrain_valid=terrain_valid,
        slope=slope,
        drainage_area_m2=drainage_area_m2,
        spi=spi,
        concave_plan_like=concave_plan_like,
        energy=energy,
        relief=relief,
        min_slope_deg=gully_min_slope_deg,
        max_slope_deg=gully_max_slope_deg,
        min_drainage_area_m2=gully_min_drainage_area_m2,
        spi_threshold=gully_spi_threshold,
        spi_percentile=gully_spi_percentile,
        energy_threshold=gully_energy_threshold,
    )

    gully_masks_by_preset: dict[str, np.ndarray] = {}
    gully_spi_cut_by_preset: dict[str, float] = {}
    for preset_name, cfg in GULLY_PRESETS.items():
        preset_mask, preset_spi_cut = _compute_gully_mask(
            terrain_valid=terrain_valid,
            slope=slope,
            drainage_area_m2=drainage_area_m2,
            spi=spi,
            concave_plan_like=concave_plan_like,
            energy=energy,
            relief=relief,
            min_slope_deg=float(cfg["min_slope_deg"]),
            max_slope_deg=float(cfg["max_slope_deg"]),
            min_drainage_area_m2=float(cfg["min_drainage_area_m2"]),
            spi_threshold=None,
            spi_percentile=float(cfg["spi_percentile"]),
            energy_threshold=float(cfg["energy_threshold"]),
        )
        gully_masks_by_preset[preset_name] = preset_mask
        gully_spi_cut_by_preset[preset_name] = preset_spi_cut

    local_max_slope = maximum_filter(slope, size=7, mode="nearest")
    slope_break = local_max_slope - slope
    roadcuts = np.logical_and.reduce(
        [
            terrain_valid,
            benches_like,
            slope <= 20.0,
            slope_break >= 15.0,
        ]
    )

    lake_domain = np.logical_and.reduce(
        [
            terrain_valid,
            slope <= lake_max_slope_deg,
            tpi <= lake_tpi_threshold,
            concave_plan_like,
        ]
    )

    lake_spi_cut = _threshold_from_absolute_or_percentile(
        values=spi,
        valid_mask=terrain_valid,
        absolute_threshold=lake_max_spi_threshold,
        percentile=lake_max_spi_percentile,
    )

    lakes = np.logical_and(lake_domain, spi <= lake_spi_cut)

    trauma = np.zeros(dem.shape, dtype=np.uint8)
    trauma[trees] |= TRAUMA_TREE
    trauma[cliffs] |= TRAUMA_CLIFF

    burial = np.zeros(dem.shape, dtype=np.uint8)
    burial[gullies] |= BURIAL_GULLY
    burial[roadcuts] |= BURIAL_ROADCUT
    burial[lakes] |= BURIAL_LAKE

    profile_uint8 = dem_profile.copy()
    profile_uint8.update(dtype="uint8", count=1, nodata=0, compress="deflate")

    out_trees = out_individual / "3_Terrain_Traps_Trees.tif"
    out_cliffs = out_individual / "3_Terrain_Traps_Cliffs.tif"
    out_gullies = out_individual / "3_Terrain_Traps_Gullies.tif"
    out_gullies_conservative = out_individual / "3_Terrain_Traps_Gullies_conservative.tif"
    out_gullies_balanced = out_individual / "3_Terrain_Traps_Gullies_balanced.tif"
    out_gullies_aggressive = out_individual / "3_Terrain_Traps_Gullies_aggressive.tif"
    out_roadcuts = out_individual / "3_Terrain_Traps_RoadCuts.tif"
    out_lakes = out_individual / "3_Terrain_Traps_Lakes.tif"
    out_trauma = out_root / "3_Terrain_Traps_trauma_bitmask.tif"
    out_burial = out_root / "3_Terrain_Traps_burial_bitmask.tif"
    out_energy = out_root / "3_Terrain_Traps_energy_proxy.tif"
    out_spi = out_root / "3_Terrain_Traps_SPI_gullies.tif"

    trees_u8 = np.zeros(dem.shape, dtype=np.uint8)
    trees_u8[trees] = 1
    cliffs_u8 = np.zeros(dem.shape, dtype=np.uint8)
    cliffs_u8[cliffs] = 1
    gullies_u8 = np.zeros(dem.shape, dtype=np.uint8)
    gullies_u8[gullies] = 1
    gullies_cons_u8 = np.zeros(dem.shape, dtype=np.uint8)
    gullies_cons_u8[gully_masks_by_preset["conservative"]] = 1
    gullies_bal_u8 = np.zeros(dem.shape, dtype=np.uint8)
    gullies_bal_u8[gully_masks_by_preset["balanced"]] = 1
    gullies_agg_u8 = np.zeros(dem.shape, dtype=np.uint8)
    gullies_agg_u8[gully_masks_by_preset["aggressive"]] = 1
    roadcuts_u8 = np.zeros(dem.shape, dtype=np.uint8)
    roadcuts_u8[roadcuts] = 1
    lakes_u8 = np.zeros(dem.shape, dtype=np.uint8)
    lakes_u8[lakes] = 1

    with rasterio.open(out_trees, "w", **profile_uint8) as dst:
        dst.write(trees_u8, 1)
    with rasterio.open(out_cliffs, "w", **profile_uint8) as dst:
        dst.write(cliffs_u8, 1)
    with rasterio.open(out_gullies, "w", **profile_uint8) as dst:
        dst.write(gullies_u8, 1)
    with rasterio.open(out_gullies_conservative, "w", **profile_uint8) as dst:
        dst.write(gullies_cons_u8, 1)
    with rasterio.open(out_gullies_balanced, "w", **profile_uint8) as dst:
        dst.write(gullies_bal_u8, 1)
    with rasterio.open(out_gullies_aggressive, "w", **profile_uint8) as dst:
        dst.write(gullies_agg_u8, 1)
    with rasterio.open(out_roadcuts, "w", **profile_uint8) as dst:
        dst.write(roadcuts_u8, 1)
    with rasterio.open(out_lakes, "w", **profile_uint8) as dst:
        dst.write(lakes_u8, 1)
    with rasterio.open(out_trauma, "w", **profile_uint8) as dst:
        dst.write(trauma, 1)
    with rasterio.open(out_burial, "w", **profile_uint8) as dst:
        dst.write(burial, 1)

    profile_float = dem_profile.copy()
    profile_float.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")

    energy_out = np.full(dem.shape, -9999.0, dtype=np.float32)
    energy_out[terrain_valid] = energy[terrain_valid]
    with rasterio.open(out_energy, "w", **profile_float) as dst:
        dst.write(energy_out, 1)

    spi_out = np.full(dem.shape, -9999.0, dtype=np.float32)
    spi_out[terrain_valid] = spi[terrain_valid]
    with rasterio.open(out_spi, "w", **profile_float) as dst:
        dst.write(spi_out, 1)

    legend_csv = out_root / "3_Terrain_Traps_legend.csv"
    with legend_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "value", "label", "notes"])
        writer.writerow(["Trees", 1, "Tree trap", "Independent binary raster"]) 
        writer.writerow(["Cliffs", 1, "Cliff/rock trap", "Independent binary raster"])
        writer.writerow(["Gullies", 1, "Gully trap", "SPI-based with A^m*S^n"])
        writer.writerow(["RoadCuts", 1, "Road cut/bench trap", "Independent binary raster"])
        writer.writerow(["Lakes", 1, "Lake trap", "Creeks excluded via low-SPI criterion"])
        writer.writerow(["TraumaBitmask", 1, "Trees", "Trees bit"])
        writer.writerow(["TraumaBitmask", 2, "Cliffs", "Cliffs bit"])
        writer.writerow(["BurialBitmask", 1, "Gullies", "Gullies bit"])
        writer.writerow(["BurialBitmask", 2, "RoadCuts", "RoadCuts bit"])
        writer.writerow(["BurialBitmask", 4, "Lakes", "Lakes bit"])

    stats_csv = out_root / "3_Terrain_Traps_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["Trees_pixels", int(np.count_nonzero(trees))])
        writer.writerow(["Cliffs_pixels", int(np.count_nonzero(cliffs))])
        writer.writerow(["Gullies_pixels", int(np.count_nonzero(gullies))])
        writer.writerow(["Gullies_conservative_pixels", int(np.count_nonzero(gully_masks_by_preset["conservative"]))])
        writer.writerow(["Gullies_balanced_pixels", int(np.count_nonzero(gully_masks_by_preset["balanced"]))])
        writer.writerow(["Gullies_aggressive_pixels", int(np.count_nonzero(gully_masks_by_preset["aggressive"]))])
        writer.writerow(["RoadCuts_pixels", int(np.count_nonzero(roadcuts))])
        writer.writerow(["Lakes_pixels", int(np.count_nonzero(lakes))])
        writer.writerow(["Gully_SPI_threshold_used", gully_spi_cut])
        writer.writerow(["Gully_SPI_threshold_conservative", gully_spi_cut_by_preset["conservative"]])
        writer.writerow(["Gully_SPI_threshold_balanced", gully_spi_cut_by_preset["balanced"]])
        writer.writerow(["Gully_SPI_threshold_aggressive", gully_spi_cut_by_preset["aggressive"]])
        writer.writerow(["Lake_SPI_max_threshold_used", lake_spi_cut])

    return [
        out_trees,
        out_cliffs,
        out_gullies,
        out_gullies_conservative,
        out_gullies_balanced,
        out_gullies_aggressive,
        out_roadcuts,
        out_lakes,
        out_trauma,
        out_burial,
        out_energy,
        out_spi,
        legend_csv,
        stats_csv,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect avalanche terrain traps (trees, cliffs, gullies, road cuts, lakes)."
    )
    parser.add_argument("--dem", required=True, help="Path to DEM raster aligned to pipeline outputs")
    parser.add_argument("--forest", required=True, help="Path to forest raster aligned to DEM")
    parser.add_argument(
        "--definitive-layers-dir",
        default="outputs/Definitive_Layers",
        help="Folder containing landforms rasters and where outputs are written",
    )
    parser.add_argument(
        "--flowpy-root",
        default="outputs/Flow-Py",
        help="Flow-Py root folder containing pra_basin_X/res_* with z_delta.tif",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: same as --definitive-layers-dir)",
    )

    parser.add_argument("--forest-tree-threshold", type=float, default=35.0)
    parser.add_argument("--energy-trauma-threshold", type=float, default=0.35)
    parser.add_argument("--gully-energy-threshold", type=float, default=0.22)

    parser.add_argument("--gully-spi-m", type=float, default=1.0)
    parser.add_argument("--gully-spi-n", type=float, default=1.0)
    parser.add_argument("--gully-spi-threshold", type=float, default=0.0)
    parser.add_argument("--gully-spi-percentile", type=float, default=88.0)
    parser.add_argument("--gully-min-drainage-area-m2", type=float, default=4000.0)
    parser.add_argument("--gully-min-slope-deg", type=float, default=13.0)
    parser.add_argument("--gully-max-slope-deg", type=float, default=48.0)

    parser.add_argument("--lake-max-slope-deg", type=float, default=6.0)
    parser.add_argument("--lake-tpi-threshold", type=float, default=-1.8)
    parser.add_argument("--lake-max-spi-threshold", type=float, default=0.0)
    parser.add_argument("--lake-max-spi-percentile", type=float, default=35.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_root = Path(__file__).resolve().parents[1]

    dem_path = Path(args.dem).expanduser()
    if not dem_path.is_absolute():
        dem_path = (app_root / dem_path).resolve()

    forest_path = Path(args.forest).expanduser()
    if not forest_path.is_absolute():
        forest_path = (app_root / forest_path).resolve()

    definitive = Path(args.definitive_layers_dir).expanduser()
    if not definitive.is_absolute():
        definitive = (app_root / definitive).resolve()

    flowpy_root = Path(args.flowpy_root).expanduser()
    if not flowpy_root.is_absolute():
        flowpy_root = (app_root / flowpy_root).resolve()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir is not None else definitive
    if not out_dir.is_absolute():
        out_dir = (app_root / out_dir).resolve()

    gully_spi_threshold = args.gully_spi_threshold if args.gully_spi_threshold > 0 else None
    lake_spi_threshold = args.lake_max_spi_threshold if args.lake_max_spi_threshold > 0 else None

    outputs = detect_terrain_traps(
        dem_path=dem_path,
        forest_path=forest_path,
        definitive_layers_dir=definitive,
        flowpy_root=flowpy_root,
        out_dir=out_dir,
        forest_tree_threshold=args.forest_tree_threshold,
        energy_trauma_threshold=args.energy_trauma_threshold,
        gully_energy_threshold=args.gully_energy_threshold,
        gully_spi_m=args.gully_spi_m,
        gully_spi_n=args.gully_spi_n,
        gully_spi_threshold=gully_spi_threshold,
        gully_spi_percentile=args.gully_spi_percentile,
        gully_min_drainage_area_m2=args.gully_min_drainage_area_m2,
        gully_min_slope_deg=args.gully_min_slope_deg,
        gully_max_slope_deg=args.gully_max_slope_deg,
        lake_max_slope_deg=args.lake_max_slope_deg,
        lake_tpi_threshold=args.lake_tpi_threshold,
        lake_max_spi_threshold=lake_spi_threshold,
        lake_max_spi_percentile=args.lake_max_spi_percentile,
    )

    print("Done. Terrain traps outputs:")
    for p in outputs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
