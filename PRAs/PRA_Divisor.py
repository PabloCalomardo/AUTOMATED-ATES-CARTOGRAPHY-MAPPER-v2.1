#!/usr/bin/env python3
"""
Pipeline to subdivide PRA raster using drainage basins derived from a DEM.

The workflow follows four explicit stages:
1) DEM validation + hydrologic preprocessing + Strahler extraction (Whitebox).
2) Junction-cell detection on the stream network (8-neighbour D8 routing).
3) Drainage area delineation for every junction cell A(p).
4) Assignment of PRA cells to the smallest possible Strahler-order basin.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from whitebox.whitebox_tools import WhiteboxTools


# Whitebox D8 pointer (ESRI style):
# 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
POINTER_TO_OFFSET: Dict[int, Tuple[int, int]] = {
	1: (0, 1),
	2: (1, 1),
	4: (1, 0),
	8: (1, -1),
	16: (0, -1),
	32: (-1, -1),
	64: (-1, 0),
	128: (-1, 1),
}


@dataclass
class RasterBundle:
	"""Container for key rasters and shared metadata."""

	dem_filled: Path
	d8_pointer: Path
	streams: Path
	strahler: Path
	profile: dict
	transform: Affine


def abs_path(path: str | Path) -> Path:
	return Path(path).expanduser().resolve()


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def validate_dem(dem_path: Path) -> None:
	"""Basic DEM sanity checks before hydrologic processing."""
	with rasterio.open(dem_path) as src:
		data = src.read(1, masked=True)
		if data.count() == 0:
			raise ValueError("DEM contains no valid cells.")
		if np.nanmin(data) == np.nanmax(data):
			raise ValueError("DEM has no elevation variability (flat constant raster).")
		if src.transform == Affine.identity():
			raise ValueError("DEM transform is identity; georeferencing looks invalid.")


def build_whitebox(work_dir: Path, verbose: bool = True) -> WhiteboxTools:
	wbt = WhiteboxTools()
	wbt.set_verbose_mode(verbose)
	wbt.set_working_dir(str(work_dir))
	return wbt


def stage_1_preprocess_and_strahler(
	wbt: WhiteboxTools,
	dem_path: Path,
	out_dir: Path,
	stream_threshold: float,
	channel_init_exponent: float,
	channel_min_slope: float,
) -> RasterBundle:
	"""Stage 1: DEM filling + D8 + streams + Strahler order."""
	validate_dem(dem_path)

	filled_dem = out_dir / "dem_filled.tif"
	d8_pointer = out_dir / "d8_pointer.tif"
	d8_accum = out_dir / "d8_accum.tif"
	slope_deg = out_dir / "slope_degrees.tif"
	channel_metric = out_dir / "channel_initiation_metric.tif"
	streams = out_dir / "streams.tif"
	strahler = out_dir / "strahler_order.tif"

	# Fill depressions first so flow routing and Strahler are stable.
	wbt.fill_depressions(
		dem=str(dem_path),
		output=str(filled_dem),
		fix_flats=True,
	)

	wbt.d8_pointer(
		dem=str(filled_dem),
		output=str(d8_pointer),
		esri_pntr=True,
	)

	wbt.d8_flow_accumulation(
		i=str(d8_pointer),
		output=str(d8_accum),
		out_type="cells",
		pntr=True,
		esri_pntr=True,
	)

	wbt.slope(
		dem=str(filled_dem),
		output=str(slope_deg),
		units="degrees",
	)

	with rasterio.open(d8_accum) as src_accum:
		accum = src_accum.read(1, masked=True).astype(np.float32)
		accum_profile = src_accum.profile.copy()

	with rasterio.open(slope_deg) as src_slope:
		slope = src_slope.read(1, masked=True).astype(np.float32)

	valid = (~accum.mask) & (~slope.mask)
	slope_tan = np.tan(np.deg2rad(np.clip(slope.filled(0.0), 0.0, None)))
	slope_tan = np.maximum(slope_tan, channel_min_slope)

	metric_arr = np.full(accum.shape, -9999.0, dtype=np.float32)
	metric_values = accum.filled(0.0) * np.power(slope_tan, channel_init_exponent)
	metric_arr[valid] = metric_values[valid]

	stream_arr = np.zeros(accum.shape, dtype=np.uint8)
	stream_arr[valid & (metric_arr > stream_threshold)] = 1

	write_raster(channel_metric, metric_arr, accum_profile, nodata=-9999.0)
	write_raster(streams, stream_arr, accum_profile, nodata=0)

	wbt.strahler_stream_order(
		d8_pntr=str(d8_pointer),
		streams=str(streams),
		output=str(strahler),
		esri_pntr=True,
	)

	with rasterio.open(filled_dem) as src:
		profile = src.profile.copy()
		transform = src.transform

	return RasterBundle(
		dem_filled=filled_dem,
		d8_pointer=d8_pointer,
		streams=streams,
		strahler=strahler,
		profile=profile,
		transform=transform,
	)


def load_raster_array(path: Path) -> Tuple[np.ndarray, dict]:
	with rasterio.open(path) as src:
		arr = src.read(1)
		profile = src.profile.copy()
	return arr, profile


def write_raster(path: Path, arr: np.ndarray, profile: dict, nodata=None) -> None:
	out_profile = profile.copy()
	out_profile.update(dtype=str(arr.dtype), count=1, compress="deflate")
	if nodata is not None:
		out_profile["nodata"] = nodata
	with rasterio.open(path, "w", **out_profile) as dst:
		dst.write(arr, 1)


def build_downstream_index(
	d8_pointer: np.ndarray,
	valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	rows, cols = d8_pointer.shape
	down_r = np.full(d8_pointer.shape, -1, dtype=np.int32)
	down_c = np.full(d8_pointer.shape, -1, dtype=np.int32)

	rr, cc = np.where(valid_mask)
	for r, c in zip(rr, cc):
		code = int(d8_pointer[r, c])
		if code not in POINTER_TO_OFFSET:
			continue
		dr, dc = POINTER_TO_OFFSET[code]
		nr, nc = r + dr, c + dc
		if 0 <= nr < rows and 0 <= nc < cols and valid_mask[nr, nc]:
			down_r[r, c] = nr
			down_c[r, c] = nc

	return down_r, down_c


def stage_2_detect_junction_cells(
	d8_pointer_path: Path,
	streams_path: Path,
	strahler_path: Path,
	out_dir: Path,
	base_profile: dict,
) -> List[dict]:
	"""Stage 2: detect junctions on 8-neighbour D8 streams.

	Junctions are created for:
	- confluences (classic tributary merge behavior), and
	- Strahler order transitions along a stream (e.g. 1->2) even without bifurcation.
	"""
	d8, _ = load_raster_array(d8_pointer_path)
	streams, _ = load_raster_array(streams_path)
	strahler, _ = load_raster_array(strahler_path)
	rows, cols = streams.shape

	stream_mask = streams > 0
	down_r, down_c = build_downstream_index(d8, stream_mask)

	upstream_count = np.zeros(streams.shape, dtype=np.int16)
	upstream_contributors: List[List[List[Tuple[int, int]]]] = [
		[[] for _ in range(streams.shape[1])] for _ in range(streams.shape[0])
	]
	rr, cc = np.where(stream_mask)
	for r, c in zip(rr, cc):
		nr, nc = down_r[r, c], down_c[r, c]
		if nr >= 0:
			upstream_count[nr, nc] += 1
			upstream_contributors[nr][nc].append((int(r), int(c)))

	# Confluence cell: receives 2+ stream contributors.
	confluence_mask = stream_mask & (upstream_count >= 2)
	c_rows, c_cols = np.where(confluence_mask)

	# Requested behavior: mark each last upstream tributary cell before confluence.
	base_junction_cell_set: set[Tuple[int, int]] = set()
	for r, c in zip(c_rows, c_cols):
		for ur, uc in upstream_contributors[r][c]:
			base_junction_cell_set.add((ur, uc))

	# Prune duplicated junction chains: when several adjacent cells of the same tributary
	# are flagged, keep only the most-upstream cell for that tributary segment.
	junction_cell_set: set[Tuple[int, int]] = set()
	for r, c in base_junction_cell_set:
		has_upstream_candidate = False
		for ur, uc in upstream_contributors[r][c]:
			if (ur, uc) in base_junction_cell_set:
				has_upstream_candidate = True
				break
		if not has_upstream_candidate:
			junction_cell_set.add((r, c))

	# Additional rule requested by user:
	# create a junction where a stream cell flows into a downstream cell with
	# strictly higher Strahler order, even if there is no geometric bifurcation.
	for r, c in zip(rr, cc):
		nr, nc = down_r[r, c], down_c[r, c]
		if nr < 0:
			continue
		up_order = int(strahler[r, c])
		down_order = int(strahler[nr, nc])
		if up_order > 0 and down_order > up_order:
			junction_cell_set.add((int(nr), int(nc)))

	# Edge fallback requested by user:
	# add junctions on border cells that are stream heads or stream outlets.
	border_mask = np.zeros(streams.shape, dtype=bool)
	border_mask[0, :] = True
	border_mask[rows - 1, :] = True
	border_mask[:, 0] = True
	border_mask[:, cols - 1] = True

	head_mask = stream_mask & (upstream_count == 0)
	outlet_mask = stream_mask & (down_r < 0)
	border_fallback_mask = border_mask & (head_mask | outlet_mask)
	b_rows, b_cols = np.where(border_fallback_mask)
	for r, c in zip(b_rows, b_cols):
		junction_cell_set.add((int(r), int(c)))

	junction_cells = sorted(junction_cell_set)

	junctions: List[dict] = []
	for idx, (r, c) in enumerate(junction_cells, start=1):
		order_value = int(strahler[r, c])
		if order_value <= 0:
			raise RuntimeError(
				f"Invalid Strahler value ({order_value}) at junction cell ({r}, {c}). "
				"Strahler must come from Whitebox output and be >= 1 on stream cells."
			)

		junctions.append(
			{
				"junction_id": idx,
				"row": int(r),
				"col": int(c),
				"strahler_order": order_value,
			}
		)

	junction_tif = out_dir / "junctions.tif"
	junction_arr = np.zeros(streams.shape, dtype=np.int32)
	for j in junctions:
		junction_arr[j["row"], j["col"]] = j["junction_id"]

	write_raster(junction_tif, junction_arr, base_profile, nodata=0)

	junction_csv = out_dir / "junctions.csv"
	with open(junction_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=["junction_id", "row", "col", "strahler_order"],
		)
		writer.writeheader()
		writer.writerows(junctions)

	return junctions


def build_upstream_adjacency(
	down_r: np.ndarray,
	down_c: np.ndarray,
	valid_mask: np.ndarray,
) -> List[List[List[Tuple[int, int]]]]:
	rows, cols = down_r.shape
	upstream: List[List[List[Tuple[int, int]]]] = [
		[[] for _ in range(cols)] for _ in range(rows)
	]
	rr, cc = np.where(valid_mask)
	for r, c in zip(rr, cc):
		nr, nc = down_r[r, c], down_c[r, c]
		if nr >= 0:
			upstream[nr][nc].append((int(r), int(c)))
	return upstream


def collect_upstream_cells(
	start_r: int,
	start_c: int,
	upstream: List[List[List[Tuple[int, int]]]],
	valid_mask: np.ndarray,
) -> List[Tuple[int, int]]:
	visited = np.zeros(valid_mask.shape, dtype=bool)
	q: deque[Tuple[int, int]] = deque([(start_r, start_c)])
	out: List[Tuple[int, int]] = []
	while q:
		r, c = q.popleft()
		if not valid_mask[r, c] or visited[r, c]:
			continue
		visited[r, c] = True
		out.append((r, c))
		for ur, uc in upstream[r][c]:
			if not visited[ur, uc]:
				q.append((ur, uc))
	return out


def stage_3_drainage_areas(
	dem_filled_path: Path,
	d8_pointer_path: Path,
	junctions: Sequence[dict],
	out_dir: Path,
	base_profile: dict,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
	"""
	Stage 3: compute A(p) for each junction p and build overlap/assignment helper rasters.

	Returns:
		best_junction_id: each cell assigned to preferred (smallest Strahler order) junction basin.
		min_order: minimum Strahler order among junction basins covering each cell.
		drainage_area_size: area (cell count) for each junction_id.
	"""
	with rasterio.open(dem_filled_path) as src:
		dem = src.read(1, masked=True)

	d8, _ = load_raster_array(d8_pointer_path)
	valid_mask = ~dem.mask

	down_r, down_c = build_downstream_index(d8, valid_mask)
	upstream = build_upstream_adjacency(down_r, down_c, valid_mask)

	rows, cols = d8.shape
	overlap_count = np.zeros((rows, cols), dtype=np.uint16)
	min_order = np.full((rows, cols), 32767, dtype=np.int16)
	best_junction_id = np.zeros((rows, cols), dtype=np.int32)
	best_area = np.full((rows, cols), np.iinfo(np.int32).max, dtype=np.int32)
	drainage_area_size: Dict[int, int] = {}

	for j in junctions:
		jid = int(j["junction_id"])
		jr, jc = int(j["row"]), int(j["col"])
		order = int(j["strahler_order"])

		cells = collect_upstream_cells(jr, jc, upstream, valid_mask)
		drainage_area_size[jid] = len(cells)

		for r, c in cells:
			overlap_count[r, c] = min(65535, overlap_count[r, c] + 1)
			current_order = min_order[r, c]
			current_area = best_area[r, c]
			current_jid = best_junction_id[r, c]

			better = False
			if order < current_order:
				better = True
			elif order == current_order and len(cells) < current_area:
				better = True
			elif order == current_order and len(cells) == current_area and jid < current_jid:
				better = True

			if better:
				min_order[r, c] = order
				best_junction_id[r, c] = jid
				best_area[r, c] = len(cells)

	# Where no junction basin covers a cell, force 0 order for convenience.
	min_order[min_order == 32767] = 0

	write_raster(out_dir / "junction_basin_overlap_count.tif", overlap_count, base_profile, nodata=0)
	write_raster(out_dir / "junction_basin_min_order.tif", min_order.astype(np.int16), base_profile, nodata=0)
	write_raster(out_dir / "junction_basin_best_junction_id.tif", best_junction_id, base_profile, nodata=0)

	return best_junction_id, min_order, drainage_area_size


def stage_4_assign_pra(
	pra_path: Path,
	best_junction_id: np.ndarray,
	min_order: np.ndarray,
	junctions: Sequence[dict],
	drainage_area_size: Dict[int, int],
	out_dir: Path,
	base_profile: dict,
) -> None:
	"""Stage 4: assign PRA cells to the smallest available Strahler-order basin."""
	with rasterio.open(pra_path) as src:
		pra = src.read(1)

	pra_mask = pra > 0
	assigned_junction = np.zeros(best_junction_id.shape, dtype=np.int32)
	assigned_order = np.zeros(min_order.shape, dtype=np.int16)

	assigned_junction[pra_mask] = best_junction_id[pra_mask]
	assigned_order[pra_mask] = min_order[pra_mask]

	write_raster(out_dir / "pra_assigned_junction.tif", assigned_junction, base_profile, nodata=0)
	write_raster(out_dir / "pra_assigned_strahler_order.tif", assigned_order, base_profile, nodata=0)

	# Per-junction stats for PRA assignment.
	row_lookup = {int(j["junction_id"]): j for j in junctions}
	unique_ids, counts = np.unique(assigned_junction[pra_mask], return_counts=True)
	stats_path = out_dir / "pra_assignment_stats.csv"
	with open(stats_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f,
			fieldnames=[
				"junction_id",
				"strahler_order",
				"drainage_area_cells",
				"pra_cells_assigned",
			],
		)
		writer.writeheader()
		for jid, cnt in zip(unique_ids, counts):
			jid = int(jid)
			if jid == 0:
				continue
			j = row_lookup[jid]
			writer.writerow(
				{
					"junction_id": jid,
					"strahler_order": int(j["strahler_order"]),
					"drainage_area_cells": int(drainage_area_size.get(jid, 0)),
					"pra_cells_assigned": int(cnt),
				}
			)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Subdivide PRA raster into drainage-based PRA subzones using Whitebox.",
	)
	parser.add_argument("--dem", required=True, help="Input DEM raster (.tif)")
	parser.add_argument("--pra", required=True, help="Input PRA mask raster (.tif)")
	parser.add_argument("--out-dir", default="outputs", help="Output directory")
	parser.add_argument(
		"--stream-threshold",
		type=float,
		default=210,
		help="PARAMETRE_2 in A*s^(PARAMETRE_1) > PARAMETRE_2 (default: 100)",
	)
	parser.add_argument(
		"--channel-init-exponent",
		type=float,
		default=0.47,
		help="PARAMETRE_1 (m) in A*s^m > threshold. m=0 reproduces accumulation-only threshold.",
	)
	parser.add_argument(
		"--channel-min-slope",
		type=float,
		default=1e-4,
		help="Minimum slope term (tan slope) to avoid zero-slope collapse when m>0.",
	)
	parser.add_argument(
		"--quiet",
		action="store_true",
		help="Disable verbose Whitebox logging",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	dem_path = abs_path(args.dem)
	pra_path = abs_path(args.pra)
	out_dir = abs_path(args.out_dir)
	temp_dir = out_dir / "temp_pra_pipeline"
	ensure_dir(out_dir)
	ensure_dir(temp_dir)

	if not dem_path.exists():
		raise FileNotFoundError(f"DEM not found: {dem_path}")
	if not pra_path.exists():
		raise FileNotFoundError(f"PRA raster not found: {pra_path}")

	wbt = build_whitebox(temp_dir, verbose=not args.quiet)

	# PART 1: preprocessing + Strahler
	bundle = stage_1_preprocess_and_strahler(
		wbt=wbt,
		dem_path=dem_path,
		out_dir=out_dir,
		stream_threshold=args.stream_threshold,
		channel_init_exponent=args.channel_init_exponent,
		channel_min_slope=args.channel_min_slope,
	)

	# PART 2: junction detection
	junctions = stage_2_detect_junction_cells(
		d8_pointer_path=bundle.d8_pointer,
		streams_path=bundle.streams,
		strahler_path=bundle.strahler,
		out_dir=out_dir,
		base_profile=bundle.profile,
	)

	# PART 3: drainage areas A(p)
	best_junction_id, min_order, drainage_area_size = stage_3_drainage_areas(
		dem_filled_path=bundle.dem_filled,
		d8_pointer_path=bundle.d8_pointer,
		junctions=junctions,
		out_dir=out_dir,
		base_profile=bundle.profile,
	)

	# PART 4: PRA assignment
	stage_4_assign_pra(
		pra_path=pra_path,
		best_junction_id=best_junction_id,
		min_order=min_order,
		junctions=junctions,
		drainage_area_size=drainage_area_size,
		out_dir=out_dir,
		base_profile=bundle.profile,
	)

	print("Pipeline completed successfully.")
	print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
	main()
