#!/usr/bin/env python3
"""Detect avalanche terrain traps from available APP_ATES layers.

This module builds a terrain-trap raster aligned to the DEM and encodes trap types
with a bitmask so overlapping traps can coexist in the same pixel:

- bit 1  (value 1): Trees (trauma + deeper burial)
- bit 2  (value 2): Cliffs / Rocks (trauma)
- bit 3  (value 4): Gullies (burial depth)
- bit 4  (value 8): Road cuts / Benches (burial depth)
- bit 5  (value 16): Lakes / Creeks (burial + drowning hazard)

Conceptual grouping used by avalanche.org terrain-trap guidance:
- Trauma amplifiers: trees, cliffs/rocks
- Burial amplifiers: gullies, benches, lakes/creeks
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from scipy.ndimage import maximum_filter, uniform_filter


TRAP_TREES = np.uint8(1)
TRAP_CLIFFS_ROCKS = np.uint8(2)
TRAP_GULLIES = np.uint8(4)
TRAP_ROADCUTS_BENCHES = np.uint8(8)
TRAP_LAKES_CREEKS = np.uint8(16)


BIT_COMPONENTS = [
    (1, "Trees", (34, 139, 34)),
    (2, "Cliffs_Rocks", (170, 85, 0)),
    (4, "Gullies", (30, 144, 255)),
    (8, "RoadCuts_Benches", (255, 140, 0)),
    (16, "Lakes_Creeks", (0, 206, 209)),
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_single_band(path: str | Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    raster_path = Path(path).expanduser().resolve()
    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True)
        data = np.asarray(band.data)
        valid = ~np.asarray(band.mask)
        profile = src.profile.copy()
    return data, valid, profile


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

        z_delta, valid, profile = _read_single_band(z_delta_path)
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

    arr, valid, profile = _read_single_band(path)
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


def _bit_label(value: int) -> str:
    if value == 0:
        return "0 none"
    names: List[str] = []
    for bit_value, name, _ in BIT_COMPONENTS:
        if value & bit_value:
            names.append(name)
    joined = " + ".join(names)
    return f"{value} {joined}"


def _bit_color_hex(value: int) -> str:
    if value == 0:
        return "#000000"

    rgbs: List[Tuple[int, int, int]] = []
    for bit_value, _name, rgb in BIT_COMPONENTS:
        if value & bit_value:
            rgbs.append(rgb)

    if not rgbs:
        return "#000000"

    arr = np.asarray(rgbs, dtype=np.float32)
    rgb = np.mean(arr, axis=0)
    r = int(np.clip(np.round(rgb[0]), 0, 255))
    g = int(np.clip(np.round(rgb[1]), 0, 255))
    b = int(np.clip(np.round(rgb[2]), 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _write_bitmask_qml(qml_path: Path) -> Path:
    entries: List[str] = [
        '                <paletteEntry value="0" color="#000000" alpha="0" label="0 none / nodata"/>'
    ]

    for value in range(1, 32):
        color = _bit_color_hex(value)
        label = _bit_label(value)
        entries.append(
            f'                <paletteEntry value="{value}" color="{color}" alpha="255" label="{label}"/>'
        )

    palette_xml = "\n".join(entries)

    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" version="3.40.13-Bratislava">
    <pipe>
        <rasterrenderer type="paletted" band="1" alphaBand="-1" nodataColor="" opacity="1">
            <rasterTransparency/>
            <colorPalette>
{palette_xml}
            </colorPalette>
        </rasterrenderer>
        <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
        <huesaturation colorizeOn="0" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" colorizeStrength="100" grayscaleMode="0" saturation="0" invertColors="0"/>
        <rasterresampler maxOversampling="2"/>
    </pipe>
</qgis>
"""

    qml_path.write_text(qml, encoding="utf-8")
    return qml_path


def detect_terrain_traps(
    dem_path: str | Path,
    forest_path: str | Path,
    definitive_layers_dir: str | Path,
    flowpy_root: str | Path,
    out_dir: str | Path,
    forest_tree_threshold: float = 35.0,
    energy_trauma_threshold: float = 0.35,
    gully_energy_threshold: float = 0.2,
) -> List[Path]:
    dem, dem_valid, dem_profile = _read_single_band(dem_path)
    dem = dem.astype(np.float32, copy=False)

    forest, forest_valid, forest_profile = _read_single_band(forest_path)
    forest = forest.astype(np.float32, copy=False)
    _check_alignment(dem_profile, forest_profile, "dem", "forest")

    definitive = Path(definitive_layers_dir).expanduser().resolve()
    flowpy = Path(flowpy_root).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    _ensure_dir(out_root)

    landform_path = _select_landform_path(
        definitive_layers_dir=definitive,
        preferred=(
            "2_Landforms_curvature_3x3.tif",
            "2_Landforms_curvature_6x6.tif",
            "2_Landforms_curvature_12x12.tif",
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

    # Terrain traps must be located inside avalanche-affected terrain.
    # We enforce this by requiring z_delta > 0.
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

    # website guidance: trees/cliffs are high-consequence trauma features, so we
    # keep a stronger energy gate for trauma amplifiers.
    trees = np.logical_and.reduce(
        [
            terrain_valid,
            forest_valid,
            forest >= forest_tree_threshold,
            energy >= energy_trauma_threshold,
            slope >= 18.0,
        ]
    )

    cliffs_rocks = np.logical_and.reduce(
        [
            terrain_valid,
            slope >= 40.0,
            relief >= 2.0,
            energy >= energy_trauma_threshold,
        ]
    )

    concave_plan_like = np.isin(landforms, np.array([1, 2, 3], dtype=np.uint8))
    gullies = np.logical_and.reduce(
        [
            terrain_valid,
            concave_plan_like,
            slope >= 12.0,
            slope <= 50.0,
            np.logical_or(energy >= gully_energy_threshold, relief >= 1.0),
        ]
    )

    local_max_slope = maximum_filter(slope, size=7, mode="nearest")
    slope_break = local_max_slope - slope
    benches_like = np.isin(landforms, np.array([2, 3, 5, 6], dtype=np.uint8))
    roadcuts_benches = np.logical_and.reduce(
        [
            terrain_valid,
            benches_like,
            slope <= 20.0,
            slope_break >= 15.0,
        ]
    )

    lakes_creeks = np.logical_and.reduce(
        [
            terrain_valid,
            slope <= 8.0,
            tpi <= -1.5,
            np.logical_or(concave_plan_like, energy >= 0.1),
        ]
    )

    bitmask = np.zeros(dem.shape, dtype=np.uint8)
    bitmask[trees] |= TRAP_TREES
    bitmask[cliffs_rocks] |= TRAP_CLIFFS_ROCKS
    bitmask[gullies] |= TRAP_GULLIES
    bitmask[roadcuts_benches] |= TRAP_ROADCUTS_BENCHES
    bitmask[lakes_creeks] |= TRAP_LAKES_CREEKS

    trauma = np.zeros(dem.shape, dtype=np.uint8)
    trauma[np.logical_or(trees, cliffs_rocks)] = 1

    burial = np.zeros(dem.shape, dtype=np.uint8)
    burial[np.logical_or.reduce([gullies, roadcuts_benches, lakes_creeks])] = 1

    profile_uint8 = dem_profile.copy()
    profile_uint8.update(dtype="uint8", count=1, nodata=0, compress="deflate")

    out_bitmask = out_root / "3_Terrain_Traps_bitmask.tif"
    out_trauma = out_root / "3_Terrain_Traps_trauma_amplifiers.tif"
    out_burial = out_root / "3_Terrain_Traps_burial_amplifiers.tif"
    out_energy = out_root / "3_Terrain_Traps_energy_proxy.tif"

    with rasterio.open(out_bitmask, "w", **profile_uint8) as dst:
        dst.write(bitmask, 1)
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

    qml_path = _write_bitmask_qml(out_root / "3_Terrain_Traps_bitmask.qml")

    legend_csv = out_root / "3_Terrain_Traps_legend.csv"
    with legend_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bit_value", "trap_type", "group"])
        writer.writerow([1, "Trees", "Trauma (also deeper burial risk)"])
        writer.writerow([2, "Cliffs_Rocks", "Trauma"])
        writer.writerow([4, "Gullies", "Burial"])
        writer.writerow([8, "RoadCuts_Benches", "Burial"])
        writer.writerow([16, "Lakes_Creeks", "Burial + drowning hazard"])

    stats_csv = out_root / "3_Terrain_Traps_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trap_type", "pixels"])
        writer.writerow(["Trees", int(np.count_nonzero(trees))])
        writer.writerow(["Cliffs_Rocks", int(np.count_nonzero(cliffs_rocks))])
        writer.writerow(["Gullies", int(np.count_nonzero(gullies))])
        writer.writerow(["RoadCuts_Benches", int(np.count_nonzero(roadcuts_benches))])
        writer.writerow(["Lakes_Creeks", int(np.count_nonzero(lakes_creeks))])

    return [out_bitmask, qml_path, out_trauma, out_burial, out_energy, legend_csv, stats_csv]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect avalanche terrain traps (trees, cliffs/rocks, gullies, benches, lakes/creeks)."
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
    parser.add_argument("--gully-energy-threshold", type=float, default=0.2)
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

    outputs = detect_terrain_traps(
        dem_path=dem_path,
        forest_path=forest_path,
        definitive_layers_dir=definitive,
        flowpy_root=flowpy_root,
        out_dir=out_dir,
        forest_tree_threshold=args.forest_tree_threshold,
        energy_trauma_threshold=args.energy_trauma_threshold,
        gully_energy_threshold=args.gully_energy_threshold,
    )

    print("Done. Terrain traps outputs:")
    for p in outputs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
