#!/usr/bin/env python3
"""Post-process Flow-Py bitmasks to one GeoJSON with avalanche contours.

Reads all source_ids_bitmask.tif files under outputs/Flow-Py/*/res_* and writes
one single GeoJSON containing:

1) avalanches: one (multi)polygon feature per avalanche id and run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom


def _find_flowpy_result_dirs(flowpy_root: Path) -> List[Path]:
	"""Return all result folders matching outputs/Flow-Py/*/res_*."""
	result_dirs: List[Path] = []
	if not flowpy_root.exists():
		return result_dirs

	for basin_dir in sorted([p for p in flowpy_root.iterdir() if p.is_dir()]):
		for res_dir in sorted([p for p in basin_dir.iterdir() if p.is_dir() and p.name.startswith("res_")]):
			result_dirs.append(res_dir)
	return result_dirs


def _run_context(res_dir: Path) -> Dict[str, str]:
	basin_dir = res_dir.parent.name
	run_id = res_dir.name
	return {
		"basin_id": basin_dir,
		"run_id": run_id,
	}


def _dem_crs_wkt(dem_path: Path) -> str | None:
	if not dem_path.exists():
		raise FileNotFoundError(f"DEM not found for CRS extraction: {dem_path}")
	with rasterio.open(dem_path) as src:
		if src.crs is None:
			return None
		return src.crs.to_wkt()


def _write_geojson(flowpy_root: Path, output_geojson: Path, target_crs_wkt: str | None = None) -> None:
	output_geojson.parent.mkdir(parents=True, exist_ok=True)

	result_dirs = _find_flowpy_result_dirs(flowpy_root)
	if not result_dirs:
		raise RuntimeError(f"No Flow-Py result folders found under: {flowpy_root}")

	# Use first available CRS as geometry source CRS.
	first_bitmask = result_dirs[0] / "source_ids_bitmask.tif"
	if not first_bitmask.exists():
		raise RuntimeError(f"Missing source_ids_bitmask.tif in: {result_dirs[0]}")
	with rasterio.open(first_bitmask) as src_ref:
		source_crs_wkt = None if src_ref.crs is None else src_ref.crs.to_wkt()

	# If no explicit target CRS is provided, preserve source geometry CRS.
	crs_wkt = target_crs_wkt if target_crs_wkt is not None else source_crs_wkt

	features: List[Dict] = []

	for res_dir in result_dirs:
		bitmask_path = res_dir / "source_ids_bitmask.tif"
		if not bitmask_path.exists():
			print(f"[skip] No source_ids_bitmask.tif in {res_dir}")
			continue

		with rasterio.open(bitmask_path) as src:
			bitmask = src.read(1).astype(np.uint64, copy=False)
			transform = src.transform
			bitmask_crs_wkt = None if src.crs is None else src.crs.to_wkt()

		ctx = _run_context(res_dir)

		# One feature group: avalanche polygons.
		n_avalanches = 0
		for avalanche_id in range(1, 65):
			bit = np.uint64(1) << np.uint64(avalanche_id - 1)
			mask = (bitmask & bit) > 0
			if not np.any(mask):
				continue
			n_avalanches += 1

			for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
				if int(value) != 1:
					continue

				if (
					crs_wkt is not None
					and bitmask_crs_wkt is not None
					and bitmask_crs_wkt != crs_wkt
				):
					geom = transform_geom(
						src_crs=bitmask_crs_wkt,
						dst_crs=crs_wkt,
						geom=geom,
						antimeridian_cutting=False,
						precision=-1,
					)

				features.append(
					{
						"type": "Feature",
						"geometry": geom,
						"properties": {
							"feature_type": "avalanche",
							"basin_id": ctx["basin_id"],
							"run_id": ctx["run_id"],
							"avalanche_id": avalanche_id,
						},
					}
				)

		print(f"[ok] {res_dir}: {n_avalanches} allaus processades")

	collection = {
		"type": "FeatureCollection",
		"name": "avalanche_shapes",
		"crs_wkt": crs_wkt,
		"features": features,
	}
	if crs_wkt is not None:
		collection["crs"] = {
			"type": "name",
			"properties": {
				"name": crs_wkt,
			},
		}
	output_geojson.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export Flow-Py avalanche polygons to one GeoJSON"
	)
	parser.add_argument(
		"--flowpy-root",
		default="outputs/Flow-Py",
		help="Root folder containing Flow-Py basin folders (default: outputs/Flow-Py)",
	)
	parser.add_argument(
		"--output-geojson",
		default="outputs/Avalanche_Shapes/avalanche_shapes.geojson",
		help="Single GeoJSON output path (default: outputs/Avalanche_Shapes/avalanche_shapes.geojson)",
	)
	parser.add_argument(
		"--dem-crs-source",
		default=None,
		help="Optional DEM path to force output CRS from DEM original metadata.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	app_root = Path(__file__).resolve().parents[1]
	flowpy_root = Path(args.flowpy_root).expanduser()
	if not flowpy_root.is_absolute():
		flowpy_root = (app_root / flowpy_root).resolve()

	output_geojson = Path(args.output_geojson).expanduser()
	if not output_geojson.is_absolute():
		output_geojson = (app_root / output_geojson).resolve()

	target_crs_wkt = None
	if args.dem_crs_source is not None:
		dem_path = Path(args.dem_crs_source).expanduser()
		if not dem_path.is_absolute():
			dem_path = (app_root / dem_path).resolve()
		target_crs_wkt = _dem_crs_wkt(dem_path)

	print(f"Found {len(_find_flowpy_result_dirs(flowpy_root))} Flow-Py result folders")
	_write_geojson(
		flowpy_root=flowpy_root,
		output_geojson=output_geojson,
		target_crs_wkt=target_crs_wkt,
	)
	print(f"Done. Output: {output_geojson}")


if __name__ == "__main__":
	main()

