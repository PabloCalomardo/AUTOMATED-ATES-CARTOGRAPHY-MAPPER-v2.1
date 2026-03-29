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

Each step writes its own folder under `outputs/` so results can be reviewed
individually.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PREPROCESSING.preprocess import align_forest_to_dem, fill_dem_simple


def _abs_path_from_app(path_str: str) -> Path:
	app_dir = Path(__file__).resolve().parent
	path = Path(path_str).expanduser()
	if not path.is_absolute():
		path = (app_dir / path).resolve()
	return path


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


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
) -> tuple[Path, Optional[Path]]:
	_ensure_dir(out_dir)
	out_dem = out_dir / "dem_filled_simple.tif"
	fill_dem_simple(in_dem=dem_path, out_dem=out_dem)

	out_forest: Optional[Path] = None
	if forest_path is not None:
		out_forest = out_dir / "forest_aligned.tif"
		align_forest_to_dem(
			in_forest=forest_path,
			ref_dem=out_dem,
			out_forest=out_forest,
			forest_crs=forest_crs,
		)

	return out_dem, out_forest


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


def step_06_flowpy_per_basin(
	dem_path: Path,
	watershed_out_dir: Path,
	flowpy_out_dir: Path,
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
	for idx, (_, pra_basin_path) in enumerate(pra_basin_files, start=1):
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
		created_run_dirs.append(run_dir)

	return created_run_dirs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run APP_ATES_PABLO pipeline (steps 1-7).")
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
		default="outputs",
		help="Base output folder (relative to APP_ATES_PABLO by default)",
	)
	parser.add_argument(
		"--only-step6",
		action="store_true",
		help="Run only step 6 (Flow-Py per basin) using existing outputs from previous steps",
	)

	# PRA parameters (keep defaults aligned with script docstring)
	parser.add_argument("--radius", type=int, default=6)
	parser.add_argument("--prob", type=float, default=0.5)
	parser.add_argument("--winddir", type=int, default=0)
	parser.add_argument("--windtol", type=int, default=180)
	parser.add_argument("--pra-thd", type=float, default=0.15)
	parser.add_argument("--sf", type=int, default=3)

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
	parser.add_argument("--flowpy-alpha", type=int, default=25)
	parser.add_argument("--flowpy-exponent", type=int, default=8)
	parser.add_argument("--flowpy-flux", type=float, default=0.003)
	parser.add_argument("--flowpy-max-z", type=float, default=270) #default dem_gran 270
	parser.add_argument(
		"--flowpy-forest",
		default="inputs/FOREST_BOW_SUMMIT.tif",
		help="Forest raster passed to Flow-Py as forest=...",
	)
	parser.add_argument(
		"--flowpy-infra",
		default=None,
		help="Optional infrastructure raster passed to Flow-Py as infra=...",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	dem_path = _abs_path_from_app(args.dem)
	forest_path = None if args.forest is None else _abs_path_from_app(args.forest)
	outputs_dir = _abs_path_from_app(args.outputs_dir)
	flowpy_dir = _abs_path_from_app(args.flowpy_dir)
	flowpy_forest = _abs_path_from_app(args.flowpy_forest)
	flowpy_infra = None if args.flowpy_infra is None else _abs_path_from_app(args.flowpy_infra)

	# Step folders (align with existing project outputs naming)
	out_01 = outputs_dir / "Inputs"
	out_02 = outputs_dir / "Preprocess"
	out_03 = outputs_dir / "PRA_AutoATES"
	out_04 = outputs_dir / "PRA_Divisor"
	out_05 = outputs_dir / "Watershed_Subdivisions"
	out_06 = outputs_dir / "Flow-Py"
	out_07 = outputs_dir / "Avalanche_Shapes"

	if args.only_step6:
		dem_filled = out_02 / "dem_filled_simple.tif"
		forest_aligned = out_02 / "forest_aligned.tif"
		if not dem_filled.exists():
			raise RuntimeError(
				"Missing preprocessed DEM for step 6 only mode: "
				f"{dem_filled}. Run full pipeline first (steps 1-5)."
			)

		flowpy_forest_for_step6 = flowpy_forest
		if (
			forest_path is not None
			and flowpy_forest.resolve() == forest_path.resolve()
			and forest_aligned.exists()
		):
			flowpy_forest_for_step6 = forest_aligned

		print("[only-step6] Running Flow-Py per basin using existing outputs...")
		step_06_flowpy_per_basin(
			dem_path=dem_filled,
			watershed_out_dir=out_05,
			flowpy_out_dir=out_06,
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

	print("[1/6] Validating inputs...")
	step_01_inputs(dem_path=dem_path, forest_path=forest_path, out_dir=out_01)

	print("[2/6] Preprocessing DEM and aligning forest raster...")
	dem_filled, forest_aligned = step_02_preprocess_dem(
		dem_path=dem_path,
		out_dir=out_02,
		forest_path=forest_path,
		forest_crs=args.forest_crs,
	)

	pra_forest_path = forest_aligned if forest_aligned is not None else forest_path
	flowpy_forest_for_run = flowpy_forest
	if forest_path is not None and flowpy_forest.resolve() == forest_path.resolve() and forest_aligned is not None:
		flowpy_forest_for_run = forest_aligned

	print("[3/6] Computing PRA...")
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

	print("[4/6] Dividing PRA (basin-based)...")
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

	grass_epsg_value = args.grass_epsg
	if grass_epsg_value is None:
		grass_epsg_value = _raster_epsg(dem_filled)
		if grass_epsg_value is None:
			raise RuntimeError(
				"Could not infer DEM EPSG for watershed subdivision. "
				"Use --grass-epsg <EPSG> (for example --grass-epsg 25833)."
			)
		print(f"[5/6] Using DEM EPSG for GRASS location: {grass_epsg_value}")

	print("[5/6] Watershed subdivision + PRA split...")
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

	step_06_flowpy_per_basin(
		dem_path=dem_filled,
		watershed_out_dir=out_05,
		flowpy_out_dir=out_06,
		flowpy_dir=flowpy_dir,
		alpha=args.flowpy_alpha,
		exponent=args.flowpy_exponent,
		flux=args.flowpy_flux,
		max_z=args.flowpy_max_z,
		forest_path=flowpy_forest_for_run,
		infra_path=flowpy_infra,
	)

	print("[7/7] Post-processing Flow-Py outputs...")
	postprocess_geojson = step_07_postprocess_flowpy(
		flowpy_out_dir=out_06,
		out_dir=out_07,
		dem_original_path=dem_path,
	)
	print(f"        postprocess: {postprocess_geojson.name}")

	print("Done.")
	print(f"Outputs base dir: {outputs_dir}")


if __name__ == "__main__":
	main()
