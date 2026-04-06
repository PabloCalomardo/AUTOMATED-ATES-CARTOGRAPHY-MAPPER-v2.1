#!/usr/bin/env python3
"""Build per-basin avalanche phase zones from Flow-Py flux and source IDs.

For each basin latest `res_*` folder:
- Reads `flux.tif` (0..1 energy proxy)
- Reads `source_ids_bitmask.tif` (up to 64 avalanche IDs)
- Writes one raster per avalanche ID present in that basin:
    values: 0=no avalanche, 1=start, 2=propagating, 3=ending

Classification thresholds (modifiable):
- start: flux >= start_threshold
- ending: flux < ending_threshold
- propagating: otherwise
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import rasterio


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _latest_result_dir(basin_dir: Path) -> Optional[Path]:
    res_dirs = [p for p in basin_dir.glob("res_*") if p.is_dir()]
    if not res_dirs:
        return None
    return max(res_dirs, key=lambda p: p.stat().st_mtime)


def _extract_basin_id(name: str) -> int:
    match = re.match(r"^pra_basin_(\d+)$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid basin folder name: {name}")
    return int(match.group(1))


def _avalanche_ids_present(bitmask: np.ndarray) -> List[int]:
    ids: List[int] = []
    for avalanche_id in range(1, 65):
        bit = np.uint64(1) << np.uint64(avalanche_id - 1)
        if np.any((bitmask & bit) > 0):
            ids.append(avalanche_id)
    return ids


def _zones_for_avalanche(
    flux: np.ndarray,
    avalanche_mask: np.ndarray,
    start_threshold: float,
    ending_threshold: float,
) -> np.ndarray:
    band = np.zeros(flux.shape, dtype=np.uint8)
    if not np.any(avalanche_mask):
        return band

    start = np.logical_and(avalanche_mask, flux >= start_threshold)
    ending = np.logical_and(avalanche_mask, flux < ending_threshold)
    propagating = np.logical_and(avalanche_mask, np.logical_not(np.logical_or(start, ending)))

    band[start] = 1
    band[propagating] = 2
    band[ending] = 3
    return band


def _write_zones_index_csv(path: Path, avalanche_ids: Sequence[int]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["avalanche_id", "file", "1_start", "2_propagating", "3_ending"])
        for avalanche_id in avalanche_ids:
            file_name = f"Ava_{avalanche_id}.tif"
            writer.writerow([avalanche_id, file_name, 1, 2, 3])
    return path


def _write_single_zone_raster(path: Path, arr: np.ndarray, profile_ref: dict) -> Path:
    profile = profile_ref.copy()
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def _cleanup_previous_outputs(out_dir: Path, basin_id: int) -> None:
    # Remove legacy multiband outputs from previous implementation.
    legacy = [
        out_dir / f"Star_propagating_Ending_Zones_Basin{basin_id}.tif",
        out_dir / f"Star_propagating_Ending_Zones_Basin{basin_id}.tif.aux.xml",
        out_dir / f"Star_propagating_Ending_Zones_Basin{basin_id}_bands.csv",
        out_dir / f"Star_propagating_Ending_Zones_Basin{basin_id}_index.csv",
        out_dir / "index.csv",
    ]
    for p in legacy:
        if p.exists():
            p.unlink()

    # Remove previous per-avalanche rasters to avoid stale files.
    for p in out_dir.glob("Star_propagating_Ending_Zones_BasinAva_*.tif"):
        p.unlink()
    for p in out_dir.glob("Star_propagating_Ending_Zones_BasinAva_*.tif.aux.xml"):
        p.unlink()
    for p in out_dir.glob("Ava_*.tif"):
        p.unlink()
    for p in out_dir.glob("Ava_*.tif.aux.xml"):
        p.unlink()


def build_start_propagating_ending_zones_for_basin(
    basin_dir: str | Path,
    definitive_layers_dir: str | Path,
    start_threshold: float = 0.99,
    ending_threshold: float = 0.075,
) -> List[Path]:
    basin_path = Path(basin_dir).expanduser().resolve()
    definitive_root = Path(definitive_layers_dir).expanduser().resolve()

    if start_threshold <= ending_threshold:
        raise ValueError("start_threshold must be greater than ending_threshold")

    if not basin_path.exists() or not basin_path.is_dir():
        return []

    basin_id = _extract_basin_id(basin_path.name)
    latest_res = _latest_result_dir(basin_path)
    if latest_res is None:
        return []

    flux_path = latest_res / "flux.tif"
    bitmask_path = latest_res / "source_ids_bitmask.tif"
    if not flux_path.exists() or not bitmask_path.exists():
        return []

    with rasterio.open(flux_path) as src_flux:
        flux = src_flux.read(1).astype(np.float32, copy=False)
        flux_profile = src_flux.profile.copy()

    with rasterio.open(bitmask_path) as src_mask:
        bitmask = src_mask.read(1).astype(np.uint64, copy=False)
        mask_profile = src_mask.profile.copy()

    checks = [
        ("width", flux_profile.get("width"), mask_profile.get("width")),
        ("height", flux_profile.get("height"), mask_profile.get("height")),
        ("transform", flux_profile.get("transform"), mask_profile.get("transform")),
        ("crs", flux_profile.get("crs"), mask_profile.get("crs")),
    ]
    mismatches = [name for name, a, b in checks if a != b]
    if mismatches:
        raise ValueError(
            "flux.tif and source_ids_bitmask.tif are not aligned "
            f"for basin {basin_id}: {', '.join(mismatches)}"
        )

    avalanche_ids = _avalanche_ids_present(bitmask)
    out_dir = definitive_root / f"Basin{basin_id}" / "Star_propagating_Ending_Zones"
    _ensure_dir(out_dir)
    _cleanup_previous_outputs(out_dir=out_dir, basin_id=basin_id)

    out_csv = out_dir / "index.csv"

    if not avalanche_ids:
        _write_zones_index_csv(out_csv, [])
        return []

    written: List[Path] = []
    for avalanche_id in avalanche_ids:
        bit = np.uint64(1) << np.uint64(avalanche_id - 1)
        avalanche_mask = (bitmask & bit) > 0
        zones = _zones_for_avalanche(
            flux=flux,
            avalanche_mask=avalanche_mask,
            start_threshold=start_threshold,
            ending_threshold=ending_threshold,
        )

        out_tif = out_dir / f"Ava_{avalanche_id}.tif"
        written.append(_write_single_zone_raster(out_tif, zones, flux_profile))

    _write_zones_index_csv(out_csv, avalanche_ids)
    return written


def run_for_all_basins(
    flowpy_root: str | Path,
    definitive_layers_dir: str | Path,
    start_threshold: float = 0.99,
    ending_threshold: float = 0.075,
) -> List[Path]:
    flowpy_root_path = Path(flowpy_root).expanduser().resolve()
    if not flowpy_root_path.exists():
        raise FileNotFoundError(f"Flow-Py root not found: {flowpy_root_path}")

    written: List[Path] = []
    basin_dirs = sorted([p for p in flowpy_root_path.iterdir() if p.is_dir() and p.name.lower().startswith("pra_basin_")])

    for basin_dir in basin_dirs:
        out = build_start_propagating_ending_zones_for_basin(
            basin_dir=basin_dir,
            definitive_layers_dir=definitive_layers_dir,
            start_threshold=start_threshold,
            ending_threshold=ending_threshold,
        )
        written.extend(out)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create start/propagating/ending zones per avalanche and basin from Flow-Py flux"
    )
    parser.add_argument(
        "--flowpy-root",
        default="outputs/Flow-Py",
        help="Flow-Py root folder containing pra_basin_X/res_*",
    )
    parser.add_argument(
        "--definitive-layers-dir",
        default="outputs/Definitive_Layers",
        help="Definitive layers root where basin folders will be written",
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.99,
        help="Flux threshold for start zone (flux >= threshold)",
    )
    parser.add_argument(
        "--ending-threshold",
        type=float,
        default=0.075,
        help="Flux threshold for ending zone (flux < threshold)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_root = Path(__file__).resolve().parents[1]

    flowpy_root = Path(args.flowpy_root).expanduser()
    if not flowpy_root.is_absolute():
        flowpy_root = (app_root / flowpy_root).resolve()

    definitive_layers_dir = Path(args.definitive_layers_dir).expanduser()
    if not definitive_layers_dir.is_absolute():
        definitive_layers_dir = (app_root / definitive_layers_dir).resolve()

    outputs = run_for_all_basins(
        flowpy_root=flowpy_root,
        definitive_layers_dir=definitive_layers_dir,
        start_threshold=args.start_threshold,
        ending_threshold=args.ending_threshold,
    )

    print("Done. Start/propagating/ending zones written:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
