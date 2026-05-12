#!/usr/bin/env python3
"""Compute continuous runout-zone difficulty characteristics (0..1).

Output layer:
- 6_Runout_Zone_Characteristics.tif

Interpretation axis (approximate bands):
- 0.00: No known runout zones
- (0.00, 0.25]: Gentle/smooth runouts, weak connection
- (0.25, 0.50]: Confined runouts, longer connection
- (0.50, 0.75]: Multiple converging/confined runouts
- (0.75, 1.00]: Steep fans/gullies/cliffs, overhead severe runout

The score is computed from Flow-Py outputs (latest res_* per basin):
- flux.tif
- z_delta.tif
- cell_counts.tif
- FP_travel_angle.tif (fallback: SL_travel_angle.tif)
- source_ids_bitmask.tif (overlap/convergence from PRA contributions)

Domain restriction:
- Only ending zones (value=3) from Definitive_Layers/BasinX/Star_propagating_Ending_Zones/Ava_*.tif

Optional context layers from Definitive_Layers (if present):
- 2_Landforms/2_Landforms_curvature_10x10.tif (preferred, with fallbacks)
- 3_Terrain_Traps_burial_bitmask.tif
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio

from PostProcess_FlowPY.raster_alignment import read_single_band, read_single_band_on_ref_grid


DEFAULT_OUTPUT_NODATA = 0.0


@dataclass
class BasinRunoutData:
    basin_id: int
    res_dir: Path
    profile: dict
    bitmask: np.ndarray
    bitmask_valid: np.ndarray
    flux: np.ndarray
    flux_valid: np.ndarray
    z_delta: np.ndarray
    z_valid: np.ndarray
    cell_counts: np.ndarray
    cc_valid: np.ndarray
    travel_angle: np.ndarray
    angle_valid: np.ndarray


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


def _latest_result_dir(basin_dir: Path) -> Optional[Path]:
    res_dirs = [p for p in basin_dir.glob("res_*") if p.is_dir()]
    if not res_dirs:
        return None
    return max(res_dirs, key=lambda p: p.stat().st_mtime)


def _extract_basin_id_from_flowpy(name: str) -> int:
    match = re.match(r"^pra_basin_(\d+)$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid Flow-Py basin folder name: {name}")
    return int(match.group(1))


def _select_first_existing(base_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def _global_percentiles(values_list: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> Tuple[float, float]:
    chunks: List[np.ndarray] = []
    for arr, msk in zip(values_list, masks):
        valid = np.logical_and(msk, np.isfinite(arr))
        if np.any(valid):
            chunks.append(arr[valid].astype(np.float32, copy=False))

    if not chunks:
        return 0.0, 1.0

    sample = np.concatenate(chunks)
    lo = float(np.percentile(sample, 5.0))
    hi = float(np.percentile(sample, 95.0))
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return lo, hi


def _scale_0_1(arr: np.ndarray, lo: float, hi: float, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.float32)
    m = np.logical_and(valid, np.isfinite(arr))
    if not np.any(m):
        return out
    scaled = (arr[m] - lo) / (hi - lo)
    out[m] = np.clip(scaled, 0.0, 1.0)
    return out


def _popcount_uint64(bitmask: np.ndarray) -> np.ndarray:
    if bitmask.dtype != np.uint64:
        bitmask = bitmask.astype(np.uint64, copy=False)

    h, w = bitmask.shape
    bytes_view = bitmask.view(np.uint8).reshape(h, w, 8)
    bits = np.unpackbits(bytes_view, axis=2)
    return bits.sum(axis=2).astype(np.uint8, copy=False)


def _load_ending_zone_masks(definitive_layers_dir: Path, basin_id: int, ref_profile: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return union and overlap count of ending zones (value 3) for one basin."""
    zones_dir = definitive_layers_dir / f"Basin{basin_id}" / "Star_propagating_Ending_Zones"
    union = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=bool)
    overlap = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.float32)

    if not zones_dir.exists():
        return union, overlap

    for ava_path in sorted(zones_dir.glob("Ava_*.tif")):
        arr, valid, profile = read_single_band(ava_path)
        _check_alignment(ref_profile, profile, "reference", str(ava_path))
        ending = np.logical_and(valid, arr == 3)
        if np.any(ending):
            union[ending] = True
            overlap[ending] += 1.0

    return union, overlap


def _load_basins(flowpy_root: Path) -> List[BasinRunoutData]:
    basin_dirs = sorted([p for p in flowpy_root.iterdir() if p.is_dir() and p.name.lower().startswith("pra_basin_")])
    loaded: List[BasinRunoutData] = []

    for basin_dir in basin_dirs:
        basin_id = _extract_basin_id_from_flowpy(basin_dir.name)
        res_dir = _latest_result_dir(basin_dir)
        if res_dir is None:
            continue

        bitmask_path = res_dir / "source_ids_bitmask.tif"
        flux_path = res_dir / "flux.tif"
        z_delta_path = res_dir / "z_delta.tif"
        cell_counts_path = res_dir / "cell_counts.tif"
        fp_angle_path = res_dir / "FP_travel_angle.tif"
        sl_angle_path = res_dir / "SL_travel_angle.tif"

        if not bitmask_path.exists() or not flux_path.exists() or not z_delta_path.exists() or not cell_counts_path.exists():
            continue

        travel_path = fp_angle_path if fp_angle_path.exists() else sl_angle_path
        if not travel_path.exists():
            continue

        bitmask_raw, bitmask_valid, profile = read_single_band(bitmask_path)
        bitmask = bitmask_raw.astype(np.uint64, copy=False)

        flux, flux_valid, flux_profile = read_single_band(flux_path)
        z_delta, z_valid, z_profile = read_single_band(z_delta_path)
        cell_counts, cc_valid, cc_profile = read_single_band(cell_counts_path)
        angle, angle_valid, angle_profile = read_single_band(travel_path)

        _check_alignment(profile, flux_profile, "bitmask", "flux")
        _check_alignment(profile, z_profile, "bitmask", "z_delta")
        _check_alignment(profile, cc_profile, "bitmask", "cell_counts")
        _check_alignment(profile, angle_profile, "bitmask", travel_path.name)

        loaded.append(
            BasinRunoutData(
                basin_id=basin_id,
                res_dir=res_dir,
                profile=profile,
                bitmask=bitmask,
                bitmask_valid=bitmask_valid,
                flux=flux.astype(np.float32, copy=False),
                flux_valid=flux_valid,
                z_delta=z_delta.astype(np.float32, copy=False),
                z_valid=z_valid,
                cell_counts=cell_counts.astype(np.float32, copy=False),
                cc_valid=cc_valid,
                travel_angle=angle.astype(np.float32, copy=False),
                angle_valid=angle_valid,
            )
        )

    return loaded


def run_runout_zone_characteristics(
    definitive_layers_dir: str | Path,
    flowpy_root: str | Path,
    out_raster_path: str | Path,
    out_stats_csv: str | Path,
    out_legend_csv: str | Path,
    flux_min_threshold: float = 0.01,
    min_evidence_threshold: float = 0.03,
) -> List[Path]:
    definitive = Path(definitive_layers_dir).expanduser().resolve()
    flowpy = Path(flowpy_root).expanduser().resolve()

    out_raster = Path(out_raster_path).expanduser().resolve()
    out_stats = Path(out_stats_csv).expanduser().resolve()
    out_legend = Path(out_legend_csv).expanduser().resolve()

    basins = _load_basins(flowpy)
    if not basins:
        raise RuntimeError(f"No valid Flow-Py basin results found under: {flowpy}")

    ref_profile = basins[0].profile.copy()
    for basin in basins[1:]:
        _check_alignment(ref_profile, basin.profile, "reference", f"Basin{basin.basin_id}")

    ending_union_by_basin: dict[int, np.ndarray] = {}
    ending_overlap_by_basin: dict[int, np.ndarray] = {}
    for basin in basins:
        ending_union, ending_overlap = _load_ending_zone_masks(definitive, basin.basin_id, basin.profile)
        ending_union_by_basin[basin.basin_id] = ending_union
        ending_overlap_by_basin[basin.basin_id] = ending_overlap

    # Optional context layers (if available, they refine confinement/trap characteristics).
    landform_path = _select_first_existing(
        definitive,
        (
            "2_Landforms/2_Landforms_curvature_10x10.tif",
            "2_Landforms/2_Landforms_curvature_15x15.tif",
            "2_Landforms/2_Landforms_curvature_20x20.tif",
            "2_Landforms_curvature_10x10.tif",
            "2_Landforms_curvature_15x15.tif",
            "2_Landforms_curvature_20x20.tif",
        ),
    )
    burial_path = definitive / "3_Terrain_Traps_burial_bitmask.tif"

    landforms = None
    landforms_valid = None
    if landform_path is not None and landform_path.exists():
        lf, lf_valid, lf_profile = read_single_band_on_ref_grid(landform_path, ref_profile)
        _check_alignment(ref_profile, lf_profile, "reference", str(landform_path))
        landforms = lf.astype(np.uint8, copy=False)
        landforms_valid = lf_valid

    burial = None
    burial_valid = None
    if burial_path.exists():
        bt, bt_valid, bt_profile = read_single_band_on_ref_grid(burial_path, ref_profile)
        _check_alignment(ref_profile, bt_profile, "reference", str(burial_path))
        burial = bt.astype(np.uint8, copy=False)
        burial_valid = bt_valid

    flux_lo, flux_hi = _global_percentiles(
        [b.flux for b in basins],
        [np.logical_and(np.logical_and(b.flux_valid, b.flux > flux_min_threshold), ending_union_by_basin[b.basin_id]) for b in basins],
    )
    z_lo, z_hi = _global_percentiles(
        [b.z_delta for b in basins],
        [np.logical_and(np.logical_and(b.z_valid, b.z_delta > 0.0), ending_union_by_basin[b.basin_id]) for b in basins],
    )
    cc_lo, cc_hi = _global_percentiles(
        [b.cell_counts for b in basins],
        [np.logical_and(np.logical_and(b.cc_valid, b.cell_counts > 0.0), ending_union_by_basin[b.basin_id]) for b in basins],
    )
    angle_lo, angle_hi = _global_percentiles(
        [b.travel_angle for b in basins],
        [np.logical_and(np.logical_and(b.angle_valid, b.travel_angle > 0.0), ending_union_by_basin[b.basin_id]) for b in basins],
    )

    # Global output stores max score across basins in overlapping cells.
    out_score = np.full((ref_profile["height"], ref_profile["width"]), DEFAULT_OUTPUT_NODATA, dtype=np.float32)
    stats_rows: List[List[object]] = []

    for basin in basins:
        ending_union = ending_union_by_basin[basin.basin_id]
        ending_overlap_count = ending_overlap_by_basin[basin.basin_id]
        overlap_mask = np.logical_and(basin.bitmask_valid, ending_union)

        flux_n = _scale_0_1(basin.flux, flux_lo, flux_hi, np.logical_and(basin.flux_valid, basin.flux > flux_min_threshold))
        z_n = _scale_0_1(basin.z_delta, z_lo, z_hi, np.logical_and(basin.z_valid, basin.z_delta > 0.0))
        cc_n = _scale_0_1(basin.cell_counts, cc_lo, cc_hi, np.logical_and(basin.cc_valid, basin.cell_counts > 0.0))
        angle_n = _scale_0_1(basin.travel_angle, angle_lo, angle_hi, np.logical_and(basin.angle_valid, basin.travel_angle > 0.0))

        ov_n = np.zeros_like(ending_overlap_count, dtype=np.float32)
        if np.any(overlap_mask):
            # 1 overlapping ending zone -> 0.0, many overlaps -> closer to 1.0
            safe_overlap = np.maximum(ending_overlap_count[overlap_mask], 1.0)
            ov_n[overlap_mask] = 1.0 - (1.0 / safe_overlap)

        context_n = np.zeros_like(flux_n, dtype=np.float32)
        if landforms is not None and landforms_valid is not None:
            # Confined and convergent landform families.
            confined = np.isin(landforms, np.array([1, 2, 3, 5, 6], dtype=np.uint8))
            context_n[np.logical_and(landforms_valid, confined)] = np.maximum(
                context_n[np.logical_and(landforms_valid, confined)],
                0.7,
            )
        if burial is not None and burial_valid is not None:
            trap_mask = np.logical_and(burial_valid, burial > 0)
            context_n[trap_mask] = np.maximum(context_n[trap_mask], 1.0)

        evidence = 0.45 * flux_n + 0.35 * z_n + 0.20 * angle_n
        runout_mask = np.logical_and.reduce(
            [
                overlap_mask,
                np.logical_or(evidence >= min_evidence_threshold, ending_union),
            ]
        )

        score = np.zeros_like(flux_n, dtype=np.float32)
        score[runout_mask] = (
            0.30 * flux_n[runout_mask]
            + 0.25 * z_n[runout_mask]
            + 0.15 * angle_n[runout_mask]
            + 0.15 * cc_n[runout_mask]
            + 0.10 * ov_n[runout_mask]
            + 0.05 * context_n[runout_mask]
        )
        score = np.clip(score, 0.0, 1.0)

        out_score = np.maximum(out_score, score)

        if np.any(runout_mask):
            vals = score[runout_mask]
            stats_rows.append(
                [
                    f"Basin{basin.basin_id}",
                    int(np.count_nonzero(runout_mask)),
                    float(np.min(vals)),
                    float(np.percentile(vals, 25)),
                    float(np.percentile(vals, 50)),
                    float(np.percentile(vals, 75)),
                    float(np.max(vals)),
                ]
            )
        else:
            stats_rows.append([f"Basin{basin.basin_id}", 0, 0.0, 0.0, 0.0, 0.0, 0.0])

    _ensure_dir(out_raster.parent)
    out_profile = ref_profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=DEFAULT_OUTPUT_NODATA, compress="deflate")
    with rasterio.open(out_raster, "w", **out_profile) as dst:
        dst.write(out_score.astype(np.float32, copy=False), 1)

    _ensure_dir(out_stats.parent)
    with out_stats.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["basin", "runout_cells", "min", "p25", "p50", "p75", "max"])
        for row in stats_rows:
            writer.writerow(row)

        global_mask = out_score > 0.0
        if np.any(global_mask):
            g = out_score[global_mask]
            writer.writerow(
                [
                    "GLOBAL",
                    int(np.count_nonzero(global_mask)),
                    float(np.min(g)),
                    float(np.percentile(g, 25)),
                    float(np.percentile(g, 50)),
                    float(np.percentile(g, 75)),
                    float(np.max(g)),
                ]
            )
        else:
            writer.writerow(["GLOBAL", 0, 0.0, 0.0, 0.0, 0.0, 0.0])

    _ensure_dir(out_legend.parent)
    with out_legend.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["range", "characteristic_axis"])
        writer.writerow(["0.00", "No known runout zones"])
        writer.writerow(["(0.00, 0.25]", "Clear boundaries, gentle transitions, smooth runouts"])
        writer.writerow(["(0.25, 0.50]", "Abrupt transitions, confined runouts, longer connection"])
        writer.writerow(["(0.50, 0.75]", "Multiple converging/confined runouts connected to starts"])
        writer.writerow(["(0.75, 1.00]", "Steep fans/gullies/cliffs, severe overhead runout"])

    return [out_raster, out_stats, out_legend]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute continuous 0-1 runout zone characteristics from Flow-Py outputs.",
    )
    parser.add_argument(
        "--definitive-layers-dir",
        default="outputs/Definitive_Layers",
        help="Definitive layers output directory",
    )
    parser.add_argument(
        "--flowpy-root",
        default="outputs/Flow-Py",
        help="Flow-Py root directory with pra_basin_X/res_*",
    )
    parser.add_argument(
        "--out-raster",
        default=None,
        help="Output raster path (default: <definitive>/6_Runout_Zone_Characteristics.tif)",
    )
    parser.add_argument(
        "--out-stats",
        default=None,
        help="Output stats CSV path (default: <definitive>/6_Runout_Zone_Characteristics_stats.csv)",
    )
    parser.add_argument(
        "--out-legend",
        default=None,
        help="Output legend CSV path (default: <definitive>/6_Runout_Zone_Characteristics_legend.csv)",
    )
    parser.add_argument(
        "--flux-min-threshold",
        type=float,
        default=0.01,
        help="Minimum flux value to be considered as active runout evidence (default: 0.01)",
    )
    parser.add_argument(
        "--min-evidence-threshold",
        type=float,
        default=0.03,
        help="Minimum combined evidence score to keep runout cells (default: 0.03)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_root = Path(__file__).resolve().parents[1]

    definitive = Path(args.definitive_layers_dir).expanduser()
    if not definitive.is_absolute():
        definitive = (app_root / definitive).resolve()

    flowpy = Path(args.flowpy_root).expanduser()
    if not flowpy.is_absolute():
        flowpy = (app_root / flowpy).resolve()

    if args.out_raster is None:
        out_raster = definitive / "6_Runout_Zone_Characteristics.tif"
    else:
        out_raster = Path(args.out_raster).expanduser()
        if not out_raster.is_absolute():
            out_raster = (app_root / out_raster).resolve()

    if args.out_stats is None:
        out_stats = definitive / "6_Runout_Zone_Characteristics_stats.csv"
    else:
        out_stats = Path(args.out_stats).expanduser()
        if not out_stats.is_absolute():
            out_stats = (app_root / out_stats).resolve()

    if args.out_legend is None:
        out_legend = definitive / "6_Runout_Zone_Characteristics_legend.csv"
    else:
        out_legend = Path(args.out_legend).expanduser()
        if not out_legend.is_absolute():
            out_legend = (app_root / out_legend).resolve()

    outputs = run_runout_zone_characteristics(
        definitive_layers_dir=definitive,
        flowpy_root=flowpy,
        out_raster_path=out_raster,
        out_stats_csv=out_stats,
        out_legend_csv=out_legend,
        flux_min_threshold=args.flux_min_threshold,
        min_evidence_threshold=args.min_evidence_threshold,
    )

    print("Done. Runout zone characteristics outputs:")
    for p in outputs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
