#!/usr/bin/env python3
"""APP_ATES_PABLO main pipeline.

Execution order (current MVP):
1) Read/validate inputs (DEM + forest density raster)
2) Preprocess DEM (simple nodata filling)
3) Compute PRA rasters (AutoATES PRA implementation)
4) Subdivide PRA (PRA divisor)
5) Watershed subdivision + PRA split (GRASS)
6) Run Flow-Py once per `pra_basin_*.tif`
7) Post-process Flow-Py outputs to one GeoJSON
8) Compute `exposure_zdelta_cellcount.tif` for each new Flow-Py `res_*`
9) Compute slope+forest ATES classes from DEM slope + forest density
10) Compute multiscale curvature-based landforms from DEM (3x3, 6x6, 12x12)
11) Compute terrain traps from DEM + forest + landforms + Flow-Py z_delta
12) Compute start/propagating/ending zones per avalanche and basin from flux

Each step writes its own folder under `outputs/` so results can be reviewed
individually.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PostProcess_FlowPY.overhead_exposure import compute_overhead_exposure_from_files
from PREPROCESSING.preprocess import align_forest_to_dem, fill_dem_simple, normalize_forest_for_flowpy
from PostProcess_FlowPY.SlopeandForest_Classification import run_slope_and_forest_classification
from PostProcess_FlowPY.landforms_multiscale import run_landforms_multiscale
from PostProcess_FlowPY.terrain_traps import detect_terrain_traps
from PostProcess_FlowPY.start_propagating_ending_zones import run_for_all_basins as run_start_propagating_ending_zones


def _abs_path_from_app(path_str: str) -> Path:
	app_dir = Path(__file__).resolve().parent
	path = Path(path_str).expanduser()
	if not path.is_absolute():
		path = (app_dir / path).resolve()
	return path


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _latest_results_dir(outputs_root: Path) -> Optional[Path]:
	if not outputs_root.exists():
		return None
	candidates = [p for p in outputs_root.glob("results_*") if p.is_dir()]
	if not candidates:
		return None
	return max(candidates, key=lambda p: p.stat().st_mtime)


def _raster_epsg(path: Path) -> Optional[str]:
	"""Return EPSG code from raster CRS when available."""
	try:
		import rasterio
	except Exception:
		return None

	with rasterio.open(path) as src:
		if src.crs is None:
			return None
		epsg = src.crs.to_epsg()
		if epsg is None:
			return None
		return str(epsg)


def step_01_inputs(dem_path: Path, forest_path: Optional[Path], out_dir: Path) -> Dict[str, Any]:
	_ensure_dir(out_dir)
	if not dem_path.exists():
		raise FileNotFoundError(f"DEM not found: {dem_path}")
	if forest_path is not None and not forest_path.exists():
		raise FileNotFoundError(f"Forest raster not found: {forest_path}")

	try:
		import rasterio
	except Exception as e:  # pragma: no cover
		raise RuntimeError("Missing dependency: rasterio") from e

	def _raster_meta(path: Path) -> Dict[str, Any]:
		with rasterio.open(path) as src:
			return {
				"path": str(path),
				"crs": None if src.crs is None else src.crs.to_string(),
				"transform": tuple(src.transform)[:6],
				"width": int(src.width),
				"height": int(src.height),
				"count": int(src.count),
				"dtype": str(src.dtypes[0]) if src.count else None,
				"nodata": src.nodata,
			}

	manifest: Dict[str, Any] = {
		"dem": _raster_meta(dem_path),
		"forest": None if forest_path is None else _raster_meta(forest_path),
	}

	(out_dir / "inputs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	return manifest


def step_02_preprocess_dem(
	dem_path: Path,
	out_dir: Path,
	forest_path: Optional[Path] = None,
	forest_crs: Optional[str] = None,
) -> tuple[Path, Optional[Path], Optional[Path]]:
	_ensure_dir(out_dir)
	out_dem = out_dir / "dem_filled_simple.tif"
	fill_dem_simple(in_dem=dem_path, out_dem=out_dem)

	out_forest: Optional[Path] = None
	out_forest_normalized: Optional[Path] = None
	if forest_path is not None:
		out_forest = out_dir / "forest_aligned.tif"
		align_forest_to_dem(
			in_forest=forest_path,
			ref_dem=out_dem,
			out_forest=out_forest,
			forest_crs=forest_crs,
		)
		out_forest_normalized = out_dir / "FOREST_NORMALIZED.tif"
		normalize_forest_for_flowpy(
			in_forest=out_forest,
			out_forest=out_forest_normalized,
		)

	return out_dem, out_forest, out_forest_normalized


def step_03_pra_autoates(
	forest_type: str,
	dem_path: Path,
	forest_path: Optional[Path],
	out_dir: Path,
	radius: int,
	prob: float,
	winddir: int,
	windtol: int,
	pra_thd: float,
	sf: int,
) -> Dict[str, Path]:
	_ensure_dir(out_dir)

	script = (Path(__file__).resolve().parent / "PRAs" / "PRA_AutoATES-v2.0.py").resolve()
	if not script.exists():
		raise FileNotFoundError(f"PRA AutoATES script not found: {script}")

	if forest_type == "no_forest":
		cmd = [
			sys.executable,
			str(script),
			forest_type,
			str(dem_path),
			str(radius),
			str(prob),
			str(winddir),
			str(windtol),
			str(pra_thd),
			str(sf),
			str(out_dir),
		]
	else:
		if forest_path is None:
			raise ValueError("forest_path is required unless forest_type=no_forest")
		cmd = [
			sys.executable,
			str(script),
			forest_type,
			str(dem_path),
			str(forest_path),
			str(radius),
			str(prob),
			str(winddir),
			str(windtol),
			str(pra_thd),
			str(sf),
			str(out_dir),
		]

	subprocess.run(cmd, check=True)

	return {
		"windshelter": out_dir / "windshelter.tif",
		"pra_continuous": out_dir / "PRA_continous.tif",
		"pra_binary": out_dir / "PRA_binary.tif",
		"log": out_dir / "log.txt",
	}


def step_04_pra_divisor(
	dem_path: Path,
	pra_binary_path: Path,
	out_dir: Path,
	quiet: bool,
	stream_threshold: float,
	channel_init_exponent: float,
	channel_min_slope: float,
) -> None:
	_ensure_dir(out_dir)

	script = (Path(__file__).resolve().parent / "PRAs" / "PRA_Divisor.py").resolve()
	if not script.exists():
		raise FileNotFoundError(f"PRA divisor script not found: {script}")

	cmd = [
		sys.executable,
		str(script),
		"--dem",
		str(dem_path),
		"--pra",
		str(pra_binary_path),
		"--out-dir",
		str(out_dir),
		"--stream-threshold",
		str(stream_threshold),
		"--channel-init-exponent",
		str(channel_init_exponent),
		"--channel-min-slope",
		str(channel_min_slope),
	]
	if quiet:
		cmd.append("--quiet")

	subprocess.run(cmd, check=True)


def step_05_watershed_subdivision(
	dem_path: Path,
	pra_assigned_path: Path,
	out_dir: Path,
	watershed_threshold: int,
	watershed_memory: int,
	grass_exe: str,
	grass_epsg: str,
	grass_db: str,
	grass_location: str,
	grass_mapset: str,
) -> None:
	"""Run GRASS-based watershed subdivision and export per-basin PRA rasters."""
	_ensure_dir(out_dir)

	script = (Path(__file__).resolve().parent / "PRAs" / "PRA_Watershed_Subdivision.py").resolve()
	if not script.exists():
		raise FileNotFoundError(f"Watershed subdivision script not found: {script}")

	cmd = [
		sys.executable,
		str(script),
		"--dem",
		str(dem_path),
		"--pra-assigned",
		str(pra_assigned_path),
		"--out-dir",
		str(out_dir),
		"--watershed-threshold",
		str(watershed_threshold),
		"--watershed-memory",
		str(watershed_memory),
		"--grass-exe",
		str(grass_exe),
		"--grass-epsg",
		str(grass_epsg),
		"--grass-db",
		str(grass_db),
		"--grass-location",
		str(grass_location),
		"--grass-mapset",
		str(grass_mapset),
	]
	subprocess.run(cmd, check=True)


def step_07_postprocess_flowpy(
	flowpy_out_dir: Path,
	out_dir: Path,
	dem_original_path: Path,
) -> Path:
	"""Run Flow-Py postprocess and write a single GeoJSON output."""
	_ensure_dir(out_dir)

	script = (Path(__file__).resolve().parent / "PostProcess_FlowPY" / "post_FlowPy.py").resolve()
	if not script.exists():
		raise FileNotFoundError(f"PostProcess script not found: {script}")

	out_geojson = out_dir / "avalanche_shapes.geojson"
	cmd = [
		sys.executable,
		str(script),
		"--flowpy-root",
		str(flowpy_out_dir),
		"--output-geojson",
		str(out_geojson),
		"--dem-crs-source",
		str(dem_original_path),
	]
	subprocess.run(cmd, check=True)
	return out_geojson


def step_09_slope_and_forest_classification(
	dem_path: Path,
	forest_pcc_path: Path,
	out_dir: Path,
	forest_window: int,
	slope_sigma: float,
	forest_adjustment: str,
) -> Path:
	"""Compute ATES classes from slope and forest density (0..4: null-simple-challenging-complex-extreme)."""
	_ensure_dir(out_dir)
	out_ates = out_dir / "SlopeandForest_Classification.tif"
	return run_slope_and_forest_classification(
		dem_path=dem_path,
		pcc_path=forest_pcc_path,
		out_path=out_ates,
		forest_window=forest_window,
		slope_sigma=slope_sigma,
		forest_adjustment=forest_adjustment,
	)


def step_10_landforms_multiscale(
	dem_path: Path,
	out_dir: Path,
	landform_windows: str,
	curvature_threshold: float,
	flat_gradient_eps: float,
) -> List[Path]:
	"""Compute 9-class landforms from profile+plan curvature at multiple scales."""
	_ensure_dir(out_dir)
	return run_landforms_multiscale(
		dem_path=dem_path,
		out_dir=out_dir,
		windows=landform_windows,
		curvature_threshold=curvature_threshold,
		flat_gradient_eps=flat_gradient_eps,
	)


def step_11_terrain_traps(
	dem_path: Path,
	forest_path: Path,
	definitive_layers_dir: Path,
	flowpy_out_dir: Path,
	forest_tree_threshold: float,
	energy_trauma_threshold: float,
	gully_energy_threshold: float,
) -> List[Path]:
	"""Detect terrain traps and write rasters in Definitive_Layers."""
	_ensure_dir(definitive_layers_dir)
	return detect_terrain_traps(
		dem_path=dem_path,
		forest_path=forest_path,
		definitive_layers_dir=definitive_layers_dir,
		flowpy_root=flowpy_out_dir,
		out_dir=definitive_layers_dir,
		forest_tree_threshold=forest_tree_threshold,
		energy_trauma_threshold=energy_trauma_threshold,
		gully_energy_threshold=gully_energy_threshold,
	)


def step_12_start_propagating_ending_zones(
	flowpy_out_dir: Path,
	definitive_layers_dir: Path,
	start_threshold: float,
	ending_threshold: float,
) -> List[Path]:
	"""Compute start/propagating/ending zones per avalanche in each basin."""
	_ensure_dir(definitive_layers_dir)
	return run_start_propagating_ending_zones(
		flowpy_root=flowpy_out_dir,
		definitive_layers_dir=definitive_layers_dir,
		start_threshold=start_threshold,
		ending_threshold=ending_threshold,
	)


def _load_flowpy_entrypoint(flowpy_dir: Path) -> Callable[[List[str], Dict[str, str]], None]:
	"""Load Flow-Py terminal entrypoint without executing its __main__ block."""
	flowpy_main = (flowpy_dir / "main.py").resolve()
	if not flowpy_main.exists():
		raise FileNotFoundError(f"Flow-Py main.py not found: {flowpy_main}")

	# Flow-Py imports sibling modules (flow_core, raster_io, ...), so ensure the
	# package directory is importable before loading main.py dynamically.
	if str(flowpy_dir) not in sys.path:
		sys.path.insert(0, str(flowpy_dir))

	spec = importlib.util.spec_from_file_location("flowpy_detrainment_main", str(flowpy_main))
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Unable to load Flow-Py module spec from: {flowpy_main}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	entrypoint = getattr(module, "main", None)
	if not callable(entrypoint):
		raise RuntimeError("Flow-Py main entrypoint is missing or not callable")
	return entrypoint


def _flowpy_release_input_for_basin(release_path: Path, run_dir: Path) -> Path:
	"""Ensure Flow-Py release values are >0 where valid (Flow-Py requirement)."""
	try:
		import numpy as np
		import rasterio
	except Exception as e:  # pragma: no cover
		raise RuntimeError("Missing dependencies for Flow-Py release preparation (numpy/rasterio)") from e

	with rasterio.open(release_path) as src:
		release = src.read(1)
		profile = src.profile.copy()
		nodata = src.nodata

	if nodata is None:
		valid_mask = release >= 0
	else:
		valid_mask = release != nodata

	if not valid_mask.any():
		return release_path

	valid_values = release[valid_mask]
	if valid_values.min() > 0:
		return release_path

	adjusted = release.copy()
	adjusted[valid_mask] = adjusted[valid_mask] + 1
	adjusted_path = run_dir / "release_for_flowpy.tif"
	with rasterio.open(adjusted_path, "w", **profile) as dst:
		dst.write(adjusted, 1)
	return adjusted_path


def _latest_flowpy_result_dir(run_dir: Path) -> Optional[Path]:
	result_dirs = [p for p in run_dir.glob("res_*") if p.is_dir()]
	if not result_dirs:
		return None
	return max(result_dirs, key=lambda p: p.stat().st_mtime)


def _create_flowpy_exposure_layer(flowpy_res_dir: Path) -> Optional[Path]:
	"""Create exposure.tif from Flow-Py outputs.

	Priority:
	1) backcalculation.tif (if back-tracking/infra is enabled)
	2) binary mask from cell_counts.tif (>0)
	"""
	try:
		import numpy as np
		import rasterio
	except Exception as e:  # pragma: no cover
		raise RuntimeError("Missing dependencies for exposure layer generation (numpy/rasterio)") from e

	backcalc_path = flowpy_res_dir / "backcalculation.tif"
	cell_counts_path = flowpy_res_dir / "cell_counts.tif"
	exposure_path = flowpy_res_dir / "exposure.tif"

	if backcalc_path.exists():
		with rasterio.open(backcalc_path) as src:
			arr = src.read(1)
			profile = src.profile.copy()
		with rasterio.open(exposure_path, "w", **profile) as dst:
			dst.write(arr, 1)
		return exposure_path

	if cell_counts_path.exists():
		with rasterio.open(cell_counts_path) as src:
			counts = src.read(1)
			profile = src.profile.copy()
			nodata = src.nodata

		if nodata is None:
			exposure = (counts > 0).astype(np.uint8)
		else:
			valid = counts != nodata
			exposure = np.zeros_like(counts, dtype=np.uint8)
			exposure[np.logical_and(valid, counts > 0)] = 1

		profile.update(dtype="uint8", nodata=0, compress="deflate")
		with rasterio.open(exposure_path, "w", **profile) as dst:
			dst.write(exposure, 1)
		return exposure_path

	return None


def _create_flowpy_zdelta_cellcount_exposure_layer(
	flowpy_res_dir: Path,
	definitive_layers_dir: Path,
	basin_id: int,
) -> Optional[Path]:
	"""Create BasinX/Exposure_zdelta_cellcount.tif in Definitive_Layers."""
	cell_counts_path = flowpy_res_dir / "cell_counts.tif"
	z_delta_path = flowpy_res_dir / "z_delta.tif"
	basin_dir = definitive_layers_dir / f"Basin{basin_id}"
	_ensure_dir(basin_dir)
	out_path = basin_dir / "Exposure_zdelta_cellcount.tif"

	if not cell_counts_path.exists() or not z_delta_path.exists():
		return None

	return compute_overhead_exposure_from_files(
		cell_count_path=cell_counts_path,
		z_delta_path=z_delta_path,
		output_path=out_path,
	)


def step_06_flowpy_per_basin(
	dem_path: Path,
	watershed_out_dir: Path,
	flowpy_out_dir: Path,
	definitive_layers_dir: Path,
	flowpy_dir: Path,
	alpha: int,
	exponent: int,
	flux: float,
	max_z: float,
	forest_path: Optional[Path],
	infra_path: Optional[Path],
) -> List[Path]:
	"""Run Flow-Py once per pra_basin raster and store each run in its own folder."""
	if not watershed_out_dir.exists():
		raise FileNotFoundError(f"Watershed output folder not found: {watershed_out_dir}")

	pattern = re.compile(r"^pra_basin_(\d+)\.tif$", re.IGNORECASE)
	pra_basin_files = []
	for tif in watershed_out_dir.glob("pra_basin_*.tif"):
		match = pattern.match(tif.name)
		if match:
			pra_basin_files.append((int(match.group(1)), tif))

	if not pra_basin_files:
		raise RuntimeError(f"No pra_basin_*.tif found in: {watershed_out_dir}")

	pra_basin_files.sort(key=lambda x: x[0])
	_ensure_dir(flowpy_out_dir)

	flowpy_main = _load_flowpy_entrypoint(flowpy_dir=flowpy_dir)
	created_run_dirs: List[Path] = []

	total = len(pra_basin_files)
	for idx, (basin_id, pra_basin_path) in enumerate(pra_basin_files, start=1):
		run_dir = flowpy_out_dir / pra_basin_path.stem
		_ensure_dir(run_dir)
		prev_res_dirs = {p.resolve() for p in run_dir.glob("res_*") if p.is_dir()}

		release_input = _flowpy_release_input_for_basin(release_path=pra_basin_path, run_dir=run_dir)

		args = [
			str(alpha),
			str(exponent),
			str(run_dir),
			str(dem_path),
			str(release_input),
		]
		kwargs: Dict[str, str] = {
			"flux": str(flux),
			"max_z": str(max_z),
		}
		if forest_path is not None:
			kwargs["forest"] = str(forest_path)
		if infra_path is not None:
			kwargs["infra"] = str(infra_path)

		print(f"[6/6] Flow-Py {idx}/{total}: {pra_basin_path.name}")
		flowpy_main(args, kwargs)

		new_res_dirs = [p for p in run_dir.glob("res_*") if p.is_dir() and p.resolve() not in prev_res_dirs]
		if new_res_dirs:
			flowpy_res_dir = max(new_res_dirs, key=lambda p: p.stat().st_mtime)
		else:
			flowpy_res_dir = _latest_flowpy_result_dir(run_dir)

		if flowpy_res_dir is not None:
			exposure_path = _create_flowpy_exposure_layer(flowpy_res_dir)
			if exposure_path is not None:
				print(f"        exposure: {exposure_path.name}")
			exposure_zdelta_cellcount_path = _create_flowpy_zdelta_cellcount_exposure_layer(
				flowpy_res_dir=flowpy_res_dir,
				definitive_layers_dir=definitive_layers_dir,
				basin_id=basin_id,
			)
			if exposure_zdelta_cellcount_path is not None:
				print(f"        exposure_zdelta_cellcount: {exposure_zdelta_cellcount_path.name}")
		created_run_dirs.append(run_dir)

	return created_run_dirs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run APP_ATES_PABLO pipeline (steps 1-12).")
	parser.add_argument("--dem", default="inputs/DEM_BOW_SUMMIT.tif", help="Path to input DEM (GeoTIFF)")
	parser.add_argument("--forest", default="inputs/FOREST_BOW_SUMMIT.tif", help="Path to forest density raster (GeoTIFF)")
	parser.add_argument(
		"--forest-crs",
		default=None,
		help="Optional CRS for forest raster when missing in metadata (example: EPSG:25833)",
	)
	parser.add_argument(
		"--forest-type",
		default="pcc",
		choices=["stems", "bav", "pcc", "sen2cc", "no_forest"],
		help="Forest input type expected by PRA module",
	)
	parser.add_argument(
		"--outputs-dir",
		default=None,
		help=(
			"Base output folder. If omitted, full runs create outputs/results_DDHHMM "
			"and --only-step6 uses latest outputs/results_*"
		),
	)
	parser.add_argument(
		"--only-step6",
		action="store_true",
		help="Run only step 6 (Flow-Py per basin) using existing outputs from previous steps",
	)
	parser.add_argument(
		"--until-n",
		type=int,
		choices=range(1, 13),
		default=None,
		help="Run pipeline from step 1 up to step N (N in 1..12)",
	)

	# PRA parameters (keep defaults aligned with script docstring)
	parser.add_argument("--radius", type=int, default=6) #default dem_gran 6
	parser.add_argument("--prob", type=float, default=0.6) #default dem_gran 0.5
	parser.add_argument("--winddir", type=int, default=0) #default dem_gran 0 (wind from north)
	parser.add_argument("--windtol", type=int, default=180) #default dem_gran 180 (no wind direction preference)
	parser.add_argument("--pra-thd", type=float, default=0.15) #default dem_gran 0.15 (PRA threshold for binary classification)
	parser.add_argument("--sf", type=int, default=3) #default dem_gran 3 (smoothing factor for final PRA binary output, higher = smoother)

	parser.add_argument("--quiet", action="store_true", help="Reduce verbose logs for some steps")

	# --- PRA_Divisor parameterization (defaults match PRA_Divisor.py)
	parser.add_argument("--divisor-stream-threshold", type=float, default=850) #default dem_gran 850
	parser.add_argument("--divisor-channel-init-exponent", type=float, default=0) #default dem_gran v0.7
	parser.add_argument("--divisor-channel-min-slope", type=float, default=0.005) #default dem_gran v1 005

	# --- Watershed_Subdivisions parameterization (defaults match PRA_Watershed_Subdivision.py)
	parser.add_argument("--watershed-threshold", type=int, default=12000)
	parser.add_argument("--watershed-memory", type=int, default=500)
	parser.add_argument("--grass-exe", default=r"C:\Program Files\QGIS 3.40.13\bin\grass84.bat")
	parser.add_argument(
		"--grass-epsg",
		default=None,
		help="EPSG for GRASS location. If omitted, inferred from preprocessed DEM CRS.",
	)
	parser.add_argument("--grass-db", default="grassdata")
	parser.add_argument("--grass-location", default="watershed_project")
	parser.add_argument("--grass-mapset", default="NOUDIRECTORIDEMAPES")

	# --- Flow-Py per basin (step 6)
	parser.add_argument(
		"--flowpy-dir",
		default="Flow-py_Autoates_Editat/FlowPy_detrainment",
		help="Path to Flow-Py code directory containing main.py",
	)
	parser.add_argument("--flowpy-alpha", type=int, default=22)
	parser.add_argument("--flowpy-exponent", type=int, default=8)
	parser.add_argument("--flowpy-flux", type=float, default=0.003)
	parser.add_argument("--flowpy-max-z", type=float, default=8000) #default ESTANDARD 270
	parser.add_argument(
		"--flowpy-infra",
		default=None,
		help="Optional infrastructure raster passed to Flow-Py as infra=...",
	)

	# --- Slope + Forest classification (step 9)
	parser.add_argument(
		"--ates-forest-window",
		type=int,
		default=5,
		help="Moving window size for forest density smoothing (odd integer, default: 5)",
	)
	parser.add_argument(
		"--ates-slope-sigma",
		type=float,
		default=1.0,
		help="Gaussian sigma to smooth slope in degrees (default: 1.0)",
	)
	parser.add_argument(
		"--ates-forest-adjustment",
		choices=("legacy", "conservative", "paper_pra", "paper_runout"),
		default="paper_pra",
		help=(
			"Forest downgrading profile for step 9: paper_pra (default), "
			"paper_runout, legacy, or conservative (alias of paper_pra)."
		),
	)

	# --- Landforms multiscale (step 10)
	parser.add_argument(
		"--landform-windows",
		default="3,6,12",
		help="Comma-separated neighborhood sizes for landform classification (default: 3,6,12)",
	)
	parser.add_argument(
		"--landform-curvature-threshold",
		type=float,
		default=1e-4,
		help="Absolute threshold for convex/even/concave separation in curvature (default: 1e-4)",
	)
	parser.add_argument(
		"--landform-flat-gradient-eps",
		type=float,
		default=1e-10,
		help="Minimum gradient^2 to compute curvature and avoid unstable flat areas",
	)

	# --- Terrain traps (step 11)
	parser.add_argument(
		"--terrain-forest-tree-threshold",
		type=float,
		default=35.0,
		help="Forest threshold (PCC-like units) for tree terrain traps",
	)
	parser.add_argument(
		"--terrain-energy-trauma-threshold",
		type=float,
		default=0.35,
		help="Normalized z_delta threshold for trauma terrain traps (trees/cliffs)",
	)
	parser.add_argument(
		"--terrain-gully-energy-threshold",
		type=float,
		default=0.2,
		help="Normalized z_delta threshold used by gully detection",
	)

	# --- Start/propagating/ending zones (step 12)
	parser.add_argument(
		"--zones-start-threshold",
		type=float,
		default=0.99,
		help="Flux threshold for start zone (flux >= threshold)",
	)
	parser.add_argument(
		"--zones-ending-threshold",
		type=float,
		default=0.075,
		help="Flux threshold for ending zone (flux < threshold)",
	)

	args = parser.parse_args()
	if args.only_step6 and args.until_n is not None:
		parser.error("--only-step6 cannot be used together with --until-n")
	if args.zones_start_threshold <= args.zones_ending_threshold:
		parser.error("--zones-start-threshold must be greater than --zones-ending-threshold")
	return args


def main() -> None:
	args = parse_args()
	until_n = args.until_n
	dem_path = _abs_path_from_app(args.dem)
	forest_path = None if args.forest is None else _abs_path_from_app(args.forest)
	app_root = Path(__file__).resolve().parent
	outputs_root = (app_root / "outputs").resolve()
	if args.outputs_dir is None:
		if args.only_step6:
			latest = _latest_results_dir(outputs_root)
			if latest is None:
				raise RuntimeError(
					"--only-step6 needs previous outputs, but no outputs/results_* folder was found. "
					"Run full pipeline first or pass --outputs-dir explicitly."
				)
			outputs_dir = latest
		else:
			run_stamp = datetime.now().strftime("%d%H%M")
			outputs_dir = (outputs_root / f"results_{run_stamp}").resolve()
	else:
		outputs_dir = _abs_path_from_app(args.outputs_dir)
	flowpy_dir = _abs_path_from_app(args.flowpy_dir)
	flowpy_infra = None if args.flowpy_infra is None else _abs_path_from_app(args.flowpy_infra)

	# Step folders (align with existing project outputs naming)
	out_01 = outputs_dir / "Inputs"
	out_02 = outputs_dir / "Preprocess"
	out_03 = outputs_dir / "PRA_AutoATES"
	out_04 = outputs_dir / "PRA_Divisor"
	out_05 = outputs_dir / "Watershed_Subdivisions"
	out_06 = outputs_dir / "Flow-Py"
	out_08 = outputs_dir / "Definitive_Layers"
	out_07 = out_08
	out_09 = out_08

	if args.only_step6:
		dem_filled = out_02 / "dem_filled_simple.tif"
		forest_normalized = out_02 / "FOREST_NORMALIZED.tif"
		if not dem_filled.exists():
			raise RuntimeError(
				"Missing preprocessed DEM for step 6 only mode: "
				f"{dem_filled}. Run full pipeline first (steps 1-5)."
			)
		if forest_path is not None and not forest_normalized.exists():
			raise RuntimeError(
				"Missing normalized forest raster for step 6 only mode: "
				f"{forest_normalized}. Run full pipeline first (steps 1-5)."
			)

		flowpy_forest_for_step6 = forest_normalized if forest_normalized.exists() else None

		print("[only-step6] Running Flow-Py per basin using existing outputs...")
		step_06_flowpy_per_basin(
			dem_path=dem_filled,
			watershed_out_dir=out_05,
			flowpy_out_dir=out_06,
			definitive_layers_dir=out_08,
			flowpy_dir=flowpy_dir,
			alpha=args.flowpy_alpha,
			exponent=args.flowpy_exponent,
			flux=args.flowpy_flux,
			max_z=args.flowpy_max_z,
			forest_path=flowpy_forest_for_step6,
			infra_path=flowpy_infra,
		)
		print("Done (step 6 only).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[1] Validating inputs...")
	step_01_inputs(dem_path=dem_path, forest_path=forest_path, out_dir=out_01)
	if until_n == 1:
		print("Stopped at step 1 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[2] Preprocessing DEM and aligning forest raster...")
	dem_filled, forest_aligned, forest_normalized = step_02_preprocess_dem(
		dem_path=dem_path,
		out_dir=out_02,
		forest_path=forest_path,
		forest_crs=args.forest_crs,
	)
	if until_n == 2:
		print("Stopped at step 2 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	pra_forest_path = forest_aligned if forest_aligned is not None else forest_path
	flowpy_forest_for_run = forest_normalized

	print("[3] Computing PRA...")
	pra_outputs = step_03_pra_autoates(
		forest_type=args.forest_type,
		dem_path=dem_filled,
		forest_path=pra_forest_path,
		out_dir=out_03,
		radius=args.radius,
		prob=args.prob,
		winddir=args.winddir,
		windtol=args.windtol,
		pra_thd=args.pra_thd,
		sf=args.sf,
	)

	pra_binary = pra_outputs["pra_binary"]
	if not pra_binary.exists():
		raise RuntimeError(f"Expected PRA binary output not found: {pra_binary}")
	if until_n == 3:
		print("Stopped at step 3 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[4] Dividing PRA (basin-based)...")
	step_04_pra_divisor(
		dem_path=dem_filled,
		pra_binary_path=pra_binary,
		out_dir=out_04,
		quiet=args.quiet,
		stream_threshold=args.divisor_stream_threshold,
		channel_init_exponent=args.divisor_channel_init_exponent,
		channel_min_slope=args.divisor_channel_min_slope,
	)

	pra_assigned = out_04 / "pra_assigned_junction.tif"
	if not pra_assigned.exists():
		raise RuntimeError(f"Expected PRA divisor output not found: {pra_assigned}")
	if until_n == 4:
		print("Stopped at step 4 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	grass_epsg_value = args.grass_epsg
	if grass_epsg_value is None:
		grass_epsg_value = _raster_epsg(dem_filled)
		if grass_epsg_value is None:
			raise RuntimeError(
				"Could not infer DEM EPSG for watershed subdivision. "
				"Use --grass-epsg <EPSG> (for example --grass-epsg 25833)."
			)
		print(f"[5] Using DEM EPSG for GRASS location: {grass_epsg_value}")

	print("[5] Watershed subdivision + PRA split...")
	step_05_watershed_subdivision(
		dem_path=dem_filled,
		pra_assigned_path=pra_assigned,
		out_dir=out_05,
		watershed_threshold=args.watershed_threshold,
		watershed_memory=args.watershed_memory,
		grass_exe=args.grass_exe,
		grass_epsg=grass_epsg_value,
		grass_db=args.grass_db,
		grass_location=args.grass_location,
		grass_mapset=args.grass_mapset,
	)
	if until_n == 5:
		print("Stopped at step 5 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[6] Running Flow-Py per basin...")
	step_06_flowpy_per_basin(
		dem_path=dem_filled,
		watershed_out_dir=out_05,
		flowpy_out_dir=out_06,
		definitive_layers_dir=out_08,
		flowpy_dir=flowpy_dir,
		alpha=args.flowpy_alpha,
		exponent=args.flowpy_exponent,
		flux=args.flowpy_flux,
		max_z=args.flowpy_max_z,
		forest_path=flowpy_forest_for_run,
		infra_path=flowpy_infra,
	)
	if until_n == 6:
		print("Stopped at step 6 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[7] Post-processing Flow-Py outputs...")
	postprocess_geojson = step_07_postprocess_flowpy(
		flowpy_out_dir=out_06,
		out_dir=out_07,
		dem_original_path=dem_path,
	)
	print(f"        postprocess: {postprocess_geojson.name}")
	if until_n == 7:
		print("Stopped at step 7 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[8] Overhead exposure (z_delta + cell_count) generated per new Flow-Py result.")
	if until_n == 8:
		print("Stopped at step 8 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[9] Computing slope + forest ATES classification...")
	forest_for_ates = forest_aligned if forest_aligned is not None else forest_path
	if forest_for_ates is None:
		raise RuntimeError("Step 9 requires a forest PCC raster (provide --forest).")
	ates_path = step_09_slope_and_forest_classification(
		dem_path=dem_filled,
		forest_pcc_path=forest_for_ates,
		out_dir=out_09,
		forest_window=args.ates_forest_window,
		slope_sigma=args.ates_slope_sigma,
		forest_adjustment=args.ates_forest_adjustment,
	)
	print(f"        slope_forest_classification: {ates_path.name}")
	if until_n == 9:
		print("Stopped at step 9 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[10] Computing multiscale curvature-based landforms...")
	landform_outputs = step_10_landforms_multiscale(
		dem_path=dem_filled,
		out_dir=out_08,
		landform_windows=args.landform_windows,
		curvature_threshold=args.landform_curvature_threshold,
		flat_gradient_eps=args.landform_flat_gradient_eps,
	)
	for landform_path in landform_outputs:
		print(f"        landforms: {landform_path.name}")
	if until_n == 10:
		print("Stopped at step 10 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[11] Detecting terrain traps (trees, cliffs/rocks, gullies, benches, lakes/creeks)...")
	forest_for_terrain = forest_aligned if forest_aligned is not None else forest_path
	if forest_for_terrain is None:
		raise RuntimeError("Step 11 requires a forest raster (provide --forest).")
	terrain_outputs = step_11_terrain_traps(
		dem_path=dem_filled,
		forest_path=forest_for_terrain,
		definitive_layers_dir=out_08,
		flowpy_out_dir=out_06,
		forest_tree_threshold=args.terrain_forest_tree_threshold,
		energy_trauma_threshold=args.terrain_energy_trauma_threshold,
		gully_energy_threshold=args.terrain_gully_energy_threshold,
	)
	for terrain_path in terrain_outputs:
		print(f"        terrain_traps: {terrain_path.name}")
	if until_n == 11:
		print("Stopped at step 11 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("[12] Computing start/propagating/ending zones per basin and avalanche...")
	zone_outputs = step_12_start_propagating_ending_zones(
		flowpy_out_dir=out_06,
		definitive_layers_dir=out_08,
		start_threshold=args.zones_start_threshold,
		ending_threshold=args.zones_ending_threshold,
	)
	for zone_path in zone_outputs:
		print(f"        zones: {zone_path}")
	if until_n == 12:
		print("Stopped at step 12 (--until-n).")
		print(f"Outputs base dir: {outputs_dir}")
		return

	print("Done.")
	print(f"Outputs base dir: {outputs_dir}")


if __name__ == "__main__":
	main()
