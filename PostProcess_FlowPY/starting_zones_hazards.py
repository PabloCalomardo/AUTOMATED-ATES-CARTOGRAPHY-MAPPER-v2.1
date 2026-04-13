#!/usr/bin/env python3
"""Compute PRA starting-zone hazard classes from landforms and PRA geometry.

Inputs expected under a results folder:
- Definitive_Layers/2_Landforms/2_Landforms_curvature_XXxXX.tif (preferred 15x15)
- Definitive_Layers/Basin*/Star_propagating_Ending_Zones/Ava_*.tif

Landform-based base hazard follows the project mapping:
- Very dangerous (3): landform classes 5, 3, 2
- Dangerous (2): landform classes 6, 1, 8, 7
- Low dangerous (1): landform classes 9, 4

Then each PRA component is adjusted by:
- Size (very small decreases, large increases)
- Isolation (far from other PRAs decreases)
- Landform coherence (single dominant landform increases)

Outputs:
- 4_StartingZones_Hazards/4_StartingZones_Hazard_BaseByLandform.tif
- 4_StartingZones_Hazards/4_StartingZones_Hazard_Adjusted.tif
- 4_StartingZones_Hazards/4_StartingZones_Hazard_legend.csv
- 4_StartingZones_Hazards/4_StartingZones_Hazard_components.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from scipy.ndimage import label


VERY_DANGEROUS_LANDFORMS = {5, 3, 2}
DANGEROUS_LANDFORMS = {6, 1, 8, 7}
LOW_DANGEROUS_LANDFORMS = {9, 4}


@dataclass
class ComponentInfo:
    basin: str
    avalanche: str
    component_id: int
    rows: np.ndarray
    cols: np.ndarray
    area_cells: int
    dominant_landform: int
    dominant_ratio: float
    centroid_row: float
    centroid_col: float
    base_score: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_single_band(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
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


def _select_landform_path(definitive_layers_dir: Path, preferred: Sequence[str]) -> Optional[Path]:
    for name in preferred:
        candidate = definitive_layers_dir / name
        if candidate.exists():
            return candidate
    return None


def _base_score_from_landform(landform_class: int) -> int:
    if landform_class in VERY_DANGEROUS_LANDFORMS:
        return 3
    if landform_class in DANGEROUS_LANDFORMS:
        return 2
    if landform_class in LOW_DANGEROUS_LANDFORMS:
        return 1
    return 0


def _collect_components(
    definitive_layers_dir: Path,
    landforms: np.ndarray,
    landform_valid: np.ndarray,
    landform_profile: dict,
) -> List[ComponentInfo]:
    components: List[ComponentInfo] = []
    structure = np.ones((3, 3), dtype=np.uint8)

    basin_dirs = sorted([p for p in definitive_layers_dir.glob("Basin*") if p.is_dir()])
    for basin_dir in basin_dirs:
        zones_dir = basin_dir / "Star_propagating_Ending_Zones"
        if not zones_dir.exists():
            continue

        for ava_path in sorted(zones_dir.glob("Ava_*.tif")):
            zones, zones_valid, zones_profile = _read_single_band(ava_path)
            _check_alignment(landform_profile, zones_profile, "landforms", str(ava_path))

            start_mask = np.logical_and.reduce(
                [
                    zones_valid,
                    zones == 1,
                    landform_valid,
                    landforms > 0,
                ]
            )
            if not np.any(start_mask):
                continue

            labels, n_components = label(start_mask.astype(np.uint8), structure=structure)
            for comp_idx in range(1, int(n_components) + 1):
                rr, cc = np.where(labels == comp_idx)
                if rr.size == 0:
                    continue

                lf_values = landforms[rr, cc]
                counts = np.bincount(lf_values.astype(np.int64), minlength=10)
                if np.sum(counts[1:]) == 0:
                    continue

                dominant_landform = int(np.argmax(counts[1:]) + 1)
                dominant_count = int(counts[dominant_landform])
                dominant_ratio = float(dominant_count) / float(rr.size)

                base_score = _base_score_from_landform(dominant_landform)
                if base_score == 0:
                    continue

                components.append(
                    ComponentInfo(
                        basin=basin_dir.name,
                        avalanche=ava_path.stem,
                        component_id=comp_idx,
                        rows=rr,
                        cols=cc,
                        area_cells=int(rr.size),
                        dominant_landform=dominant_landform,
                        dominant_ratio=dominant_ratio,
                        centroid_row=float(np.mean(rr)),
                        centroid_col=float(np.mean(cc)),
                        base_score=base_score,
                    )
                )

    return components


def _component_nearest_distances_cells(components: Sequence[ComponentInfo]) -> np.ndarray:
    n = len(components)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([np.inf], dtype=np.float32)

    centroids = np.array([(c.centroid_row, c.centroid_col) for c in components], dtype=np.float32)
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1).astype(np.float32, copy=False)


def _write_raster_uint8(path: Path, data: np.ndarray, profile_ref: dict) -> Path:
    out_profile = profile_ref.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")

    _ensure_dir(path.parent)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(np.uint8, copy=False), 1)
    return path


def compute_starting_zones_hazards(
    definitive_layers_dir: str | Path,
    out_dir: str | Path,
    landform_path: str | Path | None = None,
) -> List[Path]:
    definitive = Path(definitive_layers_dir).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    _ensure_dir(out_root)

    if landform_path is None:
        landform_selected = _select_landform_path(
            definitive_layers_dir=definitive,
            preferred=(
                "2_Landforms/2_Landforms_curvature_10x10.tif",
                "2_Landforms/2_Landforms_curvature_15x15.tif",
                "2_Landforms/2_Landforms_curvature_20x20.tif",
                "2_Landforms/2_Landforms_curvature_5x5.tif",
                "2_Landforms/2_Landforms_curvature_25x25.tif",
                "2_Landforms/2_Landforms_curvature_30x30.tif",
                "2_Landforms_curvature_10x10.tif",
                "2_Landforms_curvature_15x15.tif",
                "2_Landforms_curvature_20x20.tif",
                "2_Landforms_curvature_5x5.tif",
                "2_Landforms_curvature_25x25.tif",
                "2_Landforms_curvature_30x30.tif",
            ),
        )
    else:
        landform_selected = Path(landform_path).expanduser().resolve()

    if landform_selected is None or not landform_selected.exists():
        raise FileNotFoundError("Landform raster not found in Definitive_Layers")

    print(f"Using landform raster: {landform_selected}")

    landforms, landform_valid, landform_profile = _read_single_band(landform_selected)
    landforms = landforms.astype(np.uint8, copy=False)

    components = _collect_components(
        definitive_layers_dir=definitive,
        landforms=landforms,
        landform_valid=landform_valid,
        landform_profile=landform_profile,
    )
    if not components:
        raise RuntimeError("No starting-zone PRA components found (Ava_*.tif with value 1)")

    areas = np.array([c.area_cells for c in components], dtype=np.float32)
    area_q25 = float(np.percentile(areas, 25))
    area_q50 = float(np.percentile(areas, 50))
    area_q75 = float(np.percentile(areas, 75))

    nearest_dist = _component_nearest_distances_cells(components)
    finite_dist = nearest_dist[np.isfinite(nearest_dist)]
    if finite_dist.size > 0:
        iso_q25 = float(np.percentile(finite_dist, 25))
        iso_q75 = float(np.percentile(finite_dist, 75))
    else:
        iso_q25 = np.inf
        iso_q75 = np.inf

    base_raster = np.zeros(landforms.shape, dtype=np.uint8)
    adjusted_raster = np.zeros(landforms.shape, dtype=np.uint8)

    components_csv = out_root / "4_StartingZones_Hazard_components.csv"
    with components_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "basin",
                "avalanche",
                "component_id",
                "area_cells",
                "dominant_landform",
                "dominant_ratio",
                "base_score",
                "size_adjust",
                "isolation_adjust",
                "coherence_adjust",
                "final_score",
                "nearest_component_dist_cells",
            ]
        )

        for idx, comp in enumerate(components):
            size_adjust = 0
            if comp.area_cells <= area_q25:
                size_adjust = -1
            elif comp.area_cells >= area_q75:
                size_adjust = 1

            isolation_adjust = 0
            nearest = float(nearest_dist[idx])
            if not np.isfinite(nearest) or nearest >= iso_q75:
                isolation_adjust = -1
            elif nearest <= iso_q25:
                isolation_adjust = 1

            coherence_adjust = 0
            if comp.dominant_ratio >= 0.85 and comp.area_cells >= area_q50:
                coherence_adjust = 1
            elif comp.dominant_ratio < 0.55:
                coherence_adjust = -1

            final_score = int(np.clip(comp.base_score + size_adjust + isolation_adjust + coherence_adjust, 1, 3))

            base_raster[comp.rows, comp.cols] = np.maximum(base_raster[comp.rows, comp.cols], comp.base_score)
            adjusted_raster[comp.rows, comp.cols] = np.maximum(adjusted_raster[comp.rows, comp.cols], final_score)

            writer.writerow(
                [
                    comp.basin,
                    comp.avalanche,
                    comp.component_id,
                    comp.area_cells,
                    comp.dominant_landform,
                    round(comp.dominant_ratio, 4),
                    comp.base_score,
                    size_adjust,
                    isolation_adjust,
                    coherence_adjust,
                    final_score,
                    "inf" if not np.isfinite(nearest) else round(nearest, 3),
                ]
            )

    legend_csv = out_root / "4_StartingZones_Hazard_legend.csv"
    with legend_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "label", "description"])
        writer.writerow([0, "NoData/Outside start zones", "No start zone PRA pixels"])
        writer.writerow([1, "Poc perillos", "Landform low risk or downgraded by size/isolation"])
        writer.writerow([2, "Perillos", "Intermediate hazard"])
        writer.writerow([3, "Molt perillos", "High hazard due to landform and PRA geometry"])
        writer.writerow([],)
        writer.writerow(["Base landform groups", "", ""])
        writer.writerow(["Very dangerous", "5,3,2", "Mapped to base score 3"])
        writer.writerow(["Dangerous", "6,1,8,7", "Mapped to base score 2"])
        writer.writerow(["Low dangerous", "9,4", "Mapped to base score 1"])

    out_base = out_root / "4_StartingZones_Hazard_BaseByLandform.tif"
    out_adjusted = out_root / "4_StartingZones_Hazard_Adjusted.tif"

    _write_raster_uint8(out_base, base_raster, landform_profile)
    _write_raster_uint8(out_adjusted, adjusted_raster, landform_profile)

    return [out_base, out_adjusted, legend_csv, components_csv]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify PRA starting-zone hazards from landforms + PRA geometry.",
    )
    parser.add_argument(
        "--definitive-layers-dir",
        default="outputs/Definitive_Layers",
        help="Definitive layers directory containing Basin*/Star_propagating_Ending_Zones",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <definitive>)",
    )
    parser.add_argument(
        "--landform",
        default=None,
        help="Optional explicit landform raster path. If omitted, auto-select from Definitive_Layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_root = Path(__file__).resolve().parents[1]

    definitive = Path(args.definitive_layers_dir).expanduser()
    if not definitive.is_absolute():
        definitive = (app_root / definitive).resolve()

    if args.out_dir is None:
        out_dir = definitive
    else:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = (app_root / out_dir).resolve()

    landform = None
    if args.landform is not None:
        landform = Path(args.landform).expanduser()
        if not landform.is_absolute():
            landform = (app_root / landform).resolve()

    outputs = compute_starting_zones_hazards(
        definitive_layers_dir=definitive,
        out_dir=out_dir,
        landform_path=landform,
    )

    print("Done. Starting-zone hazard outputs:")
    for p in outputs:
        print(f" - {p}")


if __name__ == "__main__":
    main()
