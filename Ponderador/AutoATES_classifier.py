import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio, rasterio.mask
import scipy.ndimage
from osgeo import gdal
from rasterio.fill import fillnodata
from skimage import morphology

# --- Set Input Files
DEM = 'test-data/Bow Summit/dem.tif'
canopy = 'test-data/Bow Summit/forest.tif'
cell_count = 'test-data/Bow Summit/Overhead.tif'  # replace with z_delta in next iteration
FP = 'test-data/Bow Summit/FP_int16.tif'
SZ = 'test-data/Bow Summit/pra_binary.tif'
forest_type = 'bav'  # 'bav', 'stems', 'pcc', 'sen2cc'

wd = 'test-data/Bow Summit/outputs'

# --- Set default input parameters

# Moving window size to smooth slope angle layer for calcuation of Class 4 extreme
WIN_SIZE = 3

# --- Define slope angle Thresholds
# Should I increase these to capture more real world numbers or keep values based on Consensus map test areas?
# Class 0 / 1 Slope Angle Threshold (Default 15)
SAT01 = 15
# Class 1 / 2 Slope Angle Threshold (Default 18)
SAT12 = 17
# Class 2 / 3 Slope Angle Threshold (Default 28)
SAT23 = 26
# Class 3 / 4 Slope Angle Threshold (Default 39)
# This is calculated on a smoothed raster layer, so the slope angle value is not representative of real world values
SAT34 = 39  # stereo

# --- Define alpha angle thresholds
# Class 1 Alpha Angle Threshold (Default 18)
AAT1 = 18
# Class 2 Alpha Angle Threshold (Default 25)
AAT2 = 22
# Class 3 Alpha Angle Threshold (Default 38)
AAT3 = 33

if forest_type in ['pcc']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 10
    # Tree classification: "sparse" (upper bound)
    TREE2 = 50
    # Tree classification: "mixed" (upper bound)
    TREE3 = 65

if forest_type in ['bav']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 10
    # Tree classification: "sparse" (upper bound)
    TREE2 = 20
    # Tree classification: "mixed" (upper bound)
    TREE3 = 25

if forest_type in ['stems']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 100
    # Tree classification: "sparse" (upper bound)
    TREE2 = 250
    # Tree classification: "mixed" (upper bound)
    TREE3 = 500

if forest_type in ['sen2ccc']:
    # --- Add tree coverage criteria
    # Tree classification: "open" (upper bound)
    TREE1 = 20
    # Tree classification: "sparse" (upper bound)
    TREE2 = 60
    # Tree classification: "mixed" (upper bound)
    TREE3 = 85

# --- Add cell count criteria
CC1 = 3
CC2 = 40

# --- Threshold for number of cells in a cluster to be removed (generalization)
ISL_SIZE = 30000

# Conservative hybrid model weights (sum = 100)
HYBRID_WEIGHTS = {
    'exposure': 20.0,
    'runout': 15.0,
    'terrain_traps': 20.0,
    'slope_forest': 15.0,
    'flowpy': 10.0,
    'landforms': 10.0,
    'start_hazard': 5.0,
    'start_coverage': 5.0,
}

# Weighted score thresholds -> ATES class (0..4)
HYBRID_CLASS_THRESHOLDS = {
    'c1': 15.0,
    'c2': 35.0,
    'c3': 55.0,
    'c4': 78.0,
}


def _tree_thresholds_for_forest_type(forest_type: str) -> Tuple[int, int, int]:
    if forest_type in ['pcc']:
        return 12, 55, 75
    if forest_type in ['bav']:
        return 12, 25, 35
    if forest_type in ['stems']:
        return 120, 300, 600
    if forest_type in ['sen2cc', 'sen2ccc']:
        return 25, 65, 90
    raise ValueError(f"Unsupported forest_type for ponderador: {forest_type}")


def _safe_write_scaled_component(
    profile: dict,
    out_path: Path,
    component_score_01: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    arr = np.clip(component_score_01 * 100.0, 0.0, 100.0).astype('int16')
    arr[~valid_mask] = -9999
    comp_profile = profile.copy()
    comp_profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})
    with rasterio.open(out_path, 'w', **comp_profile) as dst:
        dst.write(arr.reshape(1, arr.shape[0], arr.shape[1]))


def _read_optional_layer_as_score(
    path: Path,
    ref_shape: Tuple[int, int],
    ref_transform,
    ref_crs,
) -> Optional[np.ndarray]:
    if not path.exists():
        return None

    with rasterio.open(path) as src:
        if (src.height, src.width) != ref_shape or src.transform != ref_transform or src.crs != ref_crs:
            return None
        arr = src.read(1).astype('float32')
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata

    if not np.any(valid):
        return None

    out = np.zeros_like(arr, dtype='float32')
    values = arr[valid]

    # If layer already in [0..1], keep it. Otherwise robust min-max with p2/p98.
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmin >= 0.0 and vmax <= 1.0:
        out[valid] = values
    else:
        p2 = float(np.nanpercentile(values, 2))
        p98 = float(np.nanpercentile(values, 98))
        if p98 <= p2:
            out[valid] = 0.0
        else:
            out[valid] = np.clip((values - p2) / (p98 - p2), 0.0, 1.0)
    return out


def _pick_first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _discover_context_layers(wd: Path) -> Dict[str, Optional[Path]]:
    definitive_dir = wd.parent
    landforms_dir = definitive_dir / '2_Landforms'

    landform_path = _pick_first_existing(
        [
            landforms_dir / '2_Landforms_curvature_20x20.tif',
            landforms_dir / '2_Landforms_curvature_15x15.tif',
            landforms_dir / '2_Landforms_curvature_10x10.tif',
            landforms_dir / '2_Landforms_curvature_25x25.tif',
            landforms_dir / '2_Landforms_curvature_30x30.tif',
            landforms_dir / '2_Landforms_curvature_5x5.tif',
        ]
    )

    return {
        'definitive_dir': definitive_dir,
        'landform': landform_path,
        'landform_entropy': definitive_dir / '2_Landforms_entropy_5to30.tif',
        'terrain_trauma': definitive_dir / '3_Terrain_Traps_trauma_bitmask.tif',
        'terrain_burial': definitive_dir / '3_Terrain_Traps_burial_bitmask.tif',
        'terrain_energy': definitive_dir / '3_Terrain_Traps_energy_proxy.tif',
        'runout': definitive_dir / '6_Runout_Zone_Characteristics.tif',
        'start_hazard': definitive_dir / '4_StartingZones_Hazard_Adjusted.tif',
        'start_coverage': definitive_dir / '5_StartingZones_Coverages.tif',
        'start_zone_folder': wd / 'Star_propagating_Ending_Zones',
    }


def _starting_zone_fallback_scores(
    zones_dir: Path,
    ref_shape: Tuple[int, int],
    ref_transform,
    ref_crs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    if not zones_dir.exists() or not zones_dir.is_dir():
        return None, None, 0

    files = sorted(zones_dir.glob('Ava_*.tif'))
    if not files:
        return None, None, 0

    start_hits = np.zeros(ref_shape, dtype='float32')
    prop_hits = np.zeros(ref_shape, dtype='float32')
    any_hits = np.zeros(ref_shape, dtype='float32')
    used = 0

    for f in files:
        with rasterio.open(f) as src:
            if (src.height, src.width) != ref_shape or src.transform != ref_transform or src.crs != ref_crs:
                continue
            arr = src.read(1)
            nodata = src.nodata
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata

            start_hits += ((arr == 1) & valid).astype('float32')
            prop_hits += ((arr == 2) & valid).astype('float32')
            any_hits += ((arr > 0) & valid).astype('float32')
            used += 1

    if used == 0:
        return None, None, 0

    denom = np.maximum(any_hits, 1.0)
    hazard = np.clip(0.75 * (start_hits / denom) + 0.25 * (prop_hits / denom), 0.0, 1.0)

    coverage = np.zeros_like(any_hits, dtype='float32')
    positive = any_hits > 0
    if np.any(positive):
        p95 = float(np.percentile(any_hits[positive], 95))
        scale = max(p95, 1.0)
        coverage[positive] = np.clip(any_hits[positive] / scale, 0.0, 1.0)

    return hazard, coverage, used


def _legacy_lookup_to_class(sum_codes: np.ndarray) -> np.ndarray:
    mapping = {
        10: 0, 11: 1, 12: 2, 13: 3, 14: 4,
        20: 0, 21: 1, 22: 2, 23: 3, 24: 4,
        30: 0, 31: 1, 32: 2, 33: 2, 34: 3,
        40: 0, 41: 1, 42: 2, 43: 2, 44: 3,
        110: 0, 111: 1, 112: 2, 113: 3, 114: 4,
        120: 0, 121: 1, 122: 2, 123: 3, 124: 3,
        130: 0, 131: 1, 132: 2, 133: 2, 134: 3,
        140: 0, 141: 1, 142: 2, 143: 2, 144: 3,
    }
    out = np.zeros_like(sum_codes, dtype='int16')
    for code, klass in mapping.items():
        out[sum_codes == code] = klass
    out[sum_codes < 0] = 0
    return out


def _weighted_score_to_class(score_0_100: np.ndarray) -> np.ndarray:
    out = np.zeros_like(score_0_100, dtype='int16')
    out[score_0_100 >= HYBRID_CLASS_THRESHOLDS['c1']] = 1
    out[score_0_100 >= HYBRID_CLASS_THRESHOLDS['c2']] = 2
    out[score_0_100 >= HYBRID_CLASS_THRESHOLDS['c3']] = 3
    out[score_0_100 >= HYBRID_CLASS_THRESHOLDS['c4']] = 4
    return out


def AutoATES(wd, DEM, canopy, cell_count, FP, SZ, SAT01, SAT12, SAT23, SAT34, AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE):
    wd_path = Path(wd)
    wd_path.mkdir(parents=True, exist_ok=True)
    aux_layers = _discover_context_layers(wd_path)

    # --- Write input parameters to CSV file
    labels = [
        'DEM', 'canopy', 'cell_count', 'FP', 'SZ', 'SAT01', 'SAT12', 'SAT23',
        'SAT34', 'AAT1', 'AAT2', 'AAT3', 'TREE1', 'TREE2', 'TREE3', 'CC1',
        'CC2', 'ISL_SIZE', 'WIN_SIZE', 'HYBRID_WEIGHTS'
    ]
    csvRow = [
        DEM, canopy, cell_count, FP, SZ, SAT01, SAT12, SAT23, SAT34, AAT1,
        AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE,
        HYBRID_WEIGHTS
    ]
    csvfile = os.path.join(wd, "inputpara.csv")
    with open(csvfile, "a") as fp_csv:
        wr = csv.writer(fp_csv, dialect='excel')
        wr.writerow(labels)
        wr.writerow(csvRow)

    # --- Calculate slope angle
    def calculate_slope(dem_path):
        gdal.DEMProcessing(os.path.join(wd, 'slope.tif'), dem_path, 'slope')
        with rasterio.open(os.path.join(wd, 'slope.tif')) as src:
            slope_local = src.read()
            profile_local = src.profile
        return slope_local, profile_local

    slope, profile = calculate_slope(DEM)
    slope = slope.astype('int16')

    slope_nd = np.where(slope < 0, 0, slope)

    # Optional function to calculat class 4 slope using a neighborhood function - controlled by WIN_SIZE input parameter
    # If WIN_SIZE is set to 1 this function does not do anything to the SAT34 threshold calculation
    slope_smooth = scipy.ndimage.uniform_filter(slope_nd, size=WIN_SIZE, mode='nearest')

    # Update metadata
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

    # Reclassify
    slope[np.where((0 < slope) & (slope <= SAT01))] = 0
    slope[np.where((SAT01 < slope) & (slope <= SAT12))] = 1
    slope[np.where((SAT12 < slope) & (slope <= SAT23))] = 2
    slope[np.where((SAT23 < slope) & (slope <= 100))] = 3
    slope[np.where((SAT34 < slope_smooth) & (slope_smooth <= 100))] = 4

    with rasterio.open(os.path.join(wd, "slope.tif"), 'w', **profile) as dst:
        dst.write(slope)

    with rasterio.open(os.path.join(wd, "slope_smooth.tif"), 'w', **profile) as dst:
        dst.write(slope_smooth)

    # --- Open Flow-Py data, reclassify by thresholds and combine class 1, 2, and 3 runout zones into one raster

    # --- AAT1
    with rasterio.open(FP) as src:
        array = src.read(1)
        profile = src.profile
        array = array.astype('int16')

    flow_py18 = array
    # Changed to 0 from AAT1 because we are not using Non-Avalanche Terrain - class 0
    flow_py18[np.where((flow_py18 >= 0) & (flow_py18 < 90))] = 1

    # --- AAT2
    with rasterio.open(FP) as src:
        array = src.read(1)
        profile = src.profile
        array = array.astype('int16')

    flow_py25 = array
    flow_py25[np.where((flow_py25 < AAT2))] = 0
    flow_py25[np.where((flow_py25 >= AAT2) & (flow_py25 < 90))] = 2

    # --- AAT3
    with rasterio.open(FP) as src:
        array = src.read(1)
        profile = src.profile
        array = array.astype('int16')

    flow_py38 = array
    flow_py38[np.where((flow_py38 < AAT3))] = 0
    flow_py38[np.where((flow_py38 >= AAT3) & (flow_py38 < 90))] = 3

    flowpy = np.maximum(flow_py18, flow_py25)
    flowpy = np.maximum(flowpy, flow_py38)
    flowpy = flowpy.reshape(1, flowpy.shape[0], flowpy.shape[1])

    # Update metadata
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

    with rasterio.open(os.path.join(wd, "flowpy.tif"), 'w', **profile) as dst:
        dst.write(flowpy)

    # --- Add cell count criteria

    # --- Reclassify cell count criteria
    with rasterio.open(cell_count) as src:
        raw_exposure = src.read(1).astype('float32')
        exposure_nodata = src.nodata
        exposure_present = np.isfinite(raw_exposure)
        if exposure_nodata is not None:
            exposure_present &= raw_exposure != exposure_nodata
        exposure_present &= raw_exposure > 0

        array = raw_exposure.reshape(1, raw_exposure.shape[0], raw_exposure.shape[1]).astype('int16')
        profile = src.profile

        # Update metadata
        profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

        # Reclassify
        array[np.where(array == -9999)] = 0
        array[np.where((0 <= array) & (array <= CC1))] = 1
        array[np.where((CC1 < array) & (array <= CC2))] = 2
        array[np.where((CC2 < array) & (array <= 20000))] = 3

    with rasterio.open(os.path.join(wd, "cellcount_reclass.tif"), 'w', **profile) as dst:
        dst.write(array)

    exposure_score = np.zeros(raw_exposure.shape, dtype='float32')
    exposure_score[(raw_exposure > 0) & (raw_exposure <= CC1)] = 0.40
    exposure_score[(raw_exposure > CC1) & (raw_exposure <= CC2)] = 0.70
    exposure_score[raw_exposure > CC2] = 1.00
    exposure_score[~exposure_present] = 0.0

    # --- Combine Tree coverage, slope class and cell count

    src1 = rasterio.open(os.path.join(wd, "slope.tif"))
    src1 = src1.read()

    src2 = rasterio.open(os.path.join(wd, "flowpy.tif"))
    src2 = src2.read()

    src3 = rasterio.open(os.path.join(wd, "cellcount_reclass.tif"))
    src3 = src3.read()

    ates = np.maximum(src1, src2)
    ates = np.maximum(ates, src3)

    with rasterio.open(os.path.join(wd, "merge_new.tif"), 'w', **profile) as dst:
        dst.write(ates)

    # --- Add tree coverage criteria

    src1 = rasterio.open(os.path.join(wd, "merge_new.tif"))
    src1 = src1.read()
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

    # --- Reclassify using the forest criteria
    # Smooth canopy slightly to reduce hard class boundaries and make forest-type differences less rigid.
    forest_raw = rasterio.open(canopy).read().astype('float32')
    forest = scipy.ndimage.uniform_filter(forest_raw, size=(1, 3, 3), mode='nearest').astype('int16')
    forest_open = forest.copy()
    forest_open[forest_open > TREE1] = -1
    forest_open[(forest_open >= 0) & (forest_open <= TREE1)] = 10

    forest_sparse = forest.copy()
    forest_sparse[forest_sparse > TREE2] = -1
    forest_sparse[forest <= TREE1] = -1
    forest_sparse[(forest > TREE1) & (forest <= TREE2)] = 20

    forest_dense = forest.copy()
    forest_dense[forest_dense > TREE3] = -1
    forest_dense[forest_dense <= TREE2] = -1
    forest_dense[(forest_dense > TREE2) & (forest_dense <= TREE3)] = 30

    forest_vdense = forest.copy()
    forest_vdense[forest_vdense < TREE3] = -1
    forest_vdense[forest_vdense >= TREE3] = 40

    src2 = np.maximum(forest_open, forest_sparse)
    src2 = np.maximum(src2, forest_dense)
    src2 = np.maximum(src2, forest_vdense)

    with rasterio.open(os.path.join(wd, "forest_reclass.tif"), 'w', **profile) as dst:
        dst.write(src2)

    # --- Add PRA criteria
    src3 = rasterio.open(SZ)
    src3 = src3.read()

    # Keep original binary-PRA behavior but make it robust to basin masks with IDs > 1.
    # Any positive value means PRA presence.
    src3 = src3.astype('int16')
    src3[src3 <= 0] = 0
    src3[src3 > 0] = 100

    with rasterio.open(os.path.join(wd, "SZ_reclass.tif"), 'w', **profile) as dst:
        dst.write(src3)

    legacy_sum_codes = np.sum([src1, src2, src3], axis=0)
    legacy_ates = _legacy_lookup_to_class(legacy_sum_codes)

    # Exposure must dominate zoning from class 2 upwards: where there is no
    # exposure evidence, cap the final class to 1.
    no_exposure_mask = ~exposure_present.reshape(1, exposure_present.shape[0], exposure_present.shape[1])
    legacy_ates[np.where(no_exposure_mask & (legacy_ates > 1))] = 1

    # --- Hybrid model components (all normalized to [0..1])
    shape2d = exposure_present.shape
    ref_transform = profile.get('transform')
    ref_crs = profile.get('crs')

    slope_2d = np.squeeze(slope).astype('float32')
    flowpy_2d = np.squeeze(flowpy).astype('float32')
    forest_reclass_2d = np.squeeze(src2).astype('float32')
    pra_2d = np.squeeze(src3).astype('float32')

    slope_norm = np.clip(slope_2d / 4.0, 0.0, 1.0)
    forest_factor = np.ones(shape2d, dtype='float32')
    forest_factor[forest_reclass_2d == 10] = 1.00
    forest_factor[forest_reclass_2d == 20] = 0.90
    forest_factor[forest_reclass_2d == 30] = 0.70
    forest_factor[forest_reclass_2d == 40] = 0.55
    pra_factor = np.where(pra_2d > 0, 1.00, 0.85).astype('float32')
    slope_forest_score = np.clip(slope_norm * forest_factor * pra_factor, 0.0, 1.0)

    flowpy_score = np.clip(flowpy_2d / 3.0, 0.0, 1.0)

    landform_score = np.zeros(shape2d, dtype='float32')
    terrain_traps_score = np.zeros(shape2d, dtype='float32')
    runout_score = np.zeros(shape2d, dtype='float32')
    start_hazard_score = np.zeros(shape2d, dtype='float32')
    start_coverage_score = np.zeros(shape2d, dtype='float32')

    available = {
        'landform': False,
        'landform_entropy': False,
        'terrain_trauma': False,
        'terrain_burial': False,
        'terrain_energy': False,
        'runout': False,
        'start_hazard': False,
        'start_coverage': False,
        'fallback_starting_zones': False,
    }

    # Landform + entropy component
    landform_path = aux_layers['landform']
    landform_entropy_path = aux_layers['landform_entropy']

    if landform_path is not None:
        with rasterio.open(landform_path) as src:
            if (src.height, src.width) == shape2d and src.transform == ref_transform and src.crs == ref_crs:
                code = src.read(1)
                class_score = np.zeros(shape2d, dtype='float32')
                class_score[np.isin(code, [5, 3, 2])] = 1.00
                class_score[np.isin(code, [6, 1, 8, 7])] = 0.65
                class_score[np.isin(code, [9, 4])] = 0.30
                landform_score = np.maximum(landform_score, class_score)
                available['landform'] = True

    if landform_entropy_path.exists():
        entropy = _read_optional_layer_as_score(landform_entropy_path, shape2d, ref_transform, ref_crs)
        if entropy is not None:
            available['landform_entropy'] = True
            landform_score = np.clip(0.75 * landform_score + 0.25 * entropy, 0.0, 1.0)

    # Terrain-traps component
    trauma_path = aux_layers['terrain_trauma']
    burial_path = aux_layers['terrain_burial']
    energy_path = aux_layers['terrain_energy']
    trauma_score = np.zeros(shape2d, dtype='float32')
    burial_score = np.zeros(shape2d, dtype='float32')
    energy_score = np.zeros(shape2d, dtype='float32')

    if trauma_path.exists():
        with rasterio.open(trauma_path) as src:
            if (src.height, src.width) == shape2d and src.transform == ref_transform and src.crs == ref_crs:
                arr = src.read(1)
                trauma_score = (arr > 0).astype('float32')
                available['terrain_trauma'] = True

    if burial_path.exists():
        with rasterio.open(burial_path) as src:
            if (src.height, src.width) == shape2d and src.transform == ref_transform and src.crs == ref_crs:
                arr = src.read(1)
                burial_score = (arr > 0).astype('float32')
                available['terrain_burial'] = True

    if energy_path.exists():
        energy_opt = _read_optional_layer_as_score(energy_path, shape2d, ref_transform, ref_crs)
        if energy_opt is not None:
            energy_score = energy_opt
            available['terrain_energy'] = True

    terrain_traps_score = np.clip(0.45 * trauma_score + 0.75 * burial_score + 0.20 * energy_score, 0.0, 1.0)

    # Runout component
    runout_path = aux_layers['runout']
    if runout_path.exists():
        runout_opt = _read_optional_layer_as_score(runout_path, shape2d, ref_transform, ref_crs)
        if runout_opt is not None:
            runout_score = runout_opt
            available['runout'] = True

    # Starting hazard/coverage component (direct or fallback)
    start_hazard_path = aux_layers['start_hazard']
    if start_hazard_path.exists():
        with rasterio.open(start_hazard_path) as src:
            if (src.height, src.width) == shape2d and src.transform == ref_transform and src.crs == ref_crs:
                arr = src.read(1).astype('float32')
                nodata = src.nodata
                valid = np.isfinite(arr)
                if nodata is not None:
                    valid &= arr != nodata
                score = np.zeros(shape2d, dtype='float32')
                score[valid] = np.clip(arr[valid] / 3.0, 0.0, 1.0)
                start_hazard_score = score
                available['start_hazard'] = True

    start_coverage_path = aux_layers['start_coverage']
    if start_coverage_path.exists():
        cov_opt = _read_optional_layer_as_score(start_coverage_path, shape2d, ref_transform, ref_crs)
        if cov_opt is not None:
            start_coverage_score = cov_opt
            available['start_coverage'] = True

    used_avalanches = 0
    if not available['start_hazard'] or not available['start_coverage']:
        fallback_hazard, fallback_coverage, used_avalanches = _starting_zone_fallback_scores(
            aux_layers['start_zone_folder'], shape2d, ref_transform, ref_crs
        )
        if fallback_hazard is not None and not available['start_hazard']:
            start_hazard_score = fallback_hazard
            available['start_hazard'] = True
            available['fallback_starting_zones'] = True
        if fallback_coverage is not None and not available['start_coverage']:
            start_coverage_score = fallback_coverage
            available['start_coverage'] = True
            available['fallback_starting_zones'] = True

    # Weighted score
    weighted_score = (
        HYBRID_WEIGHTS['exposure'] * exposure_score
        + HYBRID_WEIGHTS['runout'] * runout_score
        + HYBRID_WEIGHTS['terrain_traps'] * terrain_traps_score
        + HYBRID_WEIGHTS['slope_forest'] * slope_forest_score
        + HYBRID_WEIGHTS['flowpy'] * flowpy_score
        + HYBRID_WEIGHTS['landforms'] * landform_score
        + HYBRID_WEIGHTS['start_hazard'] * start_hazard_score
        + HYBRID_WEIGHTS['start_coverage'] * start_coverage_score
    )

    hybrid_ates = _weighted_score_to_class(weighted_score)

    # Gating rules (conservative)
    high_exp = exposure_score >= 0.80
    severe_runout = runout_score >= 0.70
    severe_traps = terrain_traps_score >= 0.65
    combo_class3 = high_exp & severe_runout & severe_traps
    hybrid_ates[combo_class3] = np.maximum(hybrid_ates[combo_class3], 3)

    extreme_slope = slope_2d >= 4.0
    very_severe_runout = runout_score >= 0.85
    combo_class4 = extreme_slope & high_exp & very_severe_runout
    hybrid_ates[combo_class4] = 4

    # Keep legacy behavior as baseline safety floor.
    legacy_2d = np.squeeze(legacy_ates).astype('int16')
    final_ates_2d = np.maximum(hybrid_ates, legacy_2d).astype('int16')

    # Exposure must dominate from class 2 upwards.
    final_ates_2d[(~exposure_present) & (final_ates_2d > 1)] = 1
    final_ates_2d = np.clip(final_ates_2d, 0, 4).astype('int16')

    array = final_ates_2d.reshape(1, final_ates_2d.shape[0], final_ates_2d.shape[1])

    # Diagnostics outputs
    valid_mask_2d = np.isfinite(slope_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_exposure_score.tif', exposure_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_runout_score.tif', runout_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_terrain_traps_score.tif', terrain_traps_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_slope_forest_score.tif', slope_forest_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_flowpy_score.tif', flowpy_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_landforms_score.tif', landform_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_start_hazard_score.tif', start_hazard_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'component_start_coverage_score.tif', start_coverage_score, valid_mask_2d)
    _safe_write_scaled_component(profile, wd_path / 'hybrid_weighted_score.tif', weighted_score / 100.0, valid_mask_2d)

    with open(wd_path / 'ates_hybrid_diagnostics.csv', 'w', newline='') as fdiag:
        writer = csv.writer(fdiag)
        writer.writerow(['key', 'value'])
        writer.writerow(['weights', HYBRID_WEIGHTS])
        writer.writerow(['thresholds', HYBRID_CLASS_THRESHOLDS])
        for k, v in available.items():
            writer.writerow([f'available_{k}', int(bool(v))])
        writer.writerow(['fallback_ava_files_used', int(used_avalanches)])
        writer.writerow(['landform_path', '' if aux_layers['landform'] is None else str(aux_layers['landform'])])
        writer.writerow(['runout_path', str(aux_layers['runout'])])
        writer.writerow(['start_hazard_path', str(aux_layers['start_hazard'])])
        writer.writerow(['start_coverage_path', str(aux_layers['start_coverage'])])

    # --- Save raster to path
    with rasterio.open(os.path.join(wd, "merge_all.tif"), "w", **profile) as dest:
        dest.write(array)

    # --- Remove clusters of raster cells smaller than ISL_SIZE
    raster = gdal.Open(DEM)
    gt = raster.GetGeoTransform()
    pixelSizeX = gt[1]
    pixelSizeY = -gt[5]
    num_cells = int(max(1, np.around(ISL_SIZE / (pixelSizeX * pixelSizeY))))

    # --- Open file
    src1 = rasterio.open(os.path.join(wd, "merge_all.tif"))
    src1 = src1.read(1)

    # --- Change values to prepare for morphology and rasterio.fill
    src1 = src1 + 1
    src1 = src1.reshape(1, src1.shape[0], src1.shape[1])

    # --- Same as region group in arcmap. Each cluster gets a value between 1 and num_labels (number of clusters)
    # 20210430 JS changed connectivity to 2
    lab, num_labels = morphology.label(src1, connectivity=2, return_num=True)

    rg = np.arange(1, num_labels + 1, 1)

    # --- Loop through all clusters and assign all clusters with less then ISL_SIZE to the value 0 (set null)
    for i in rg:
        occurrences = np.count_nonzero(lab == i)
        if occurrences < num_cells:
            lab[np.where(lab == i)] = 0

    # --- Save as dtype int16
    lab = lab.astype('int16')

    search_dist = num_cells / 4

    # --- This algorithm will interpolate values for all designated nodata pixels (marked by zeros) (nibble)
    data = fillnodata(src1, lab, max_search_distance=search_dist, smoothing_iterations=0)

    # --- Change values back to standardized way of plotting ATES (0, 1, 2, 3 and 4)
    data = data - 1
    data[np.where(data == 0)] = -9999
    data = data.astype('int16')
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

    # --- Save raster to path
    with rasterio.open(os.path.join(wd, "ates_gen.tif"), "w", **profile) as dest:
        dest.write(data)


def run_autoates_weighted(
    dem_path,
    canopy_path,
    cell_count_path,
    fp_path,
    sz_path,
    out_dir,
    forest_type='bav',
    sat01=15,
    sat12=17,
    sat23=26,
    sat34=39,
    aat1=18,
    aat2=22,
    aat3=33,
    cc1=3,
    cc2=40,
    isl_size=30000,
    win_size=3,
    output_name='ates_gen.tif',
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree1, tree2, tree3 = _tree_thresholds_for_forest_type(str(forest_type).lower())

    AutoATES(
        str(out_dir),
        str(dem_path),
        str(canopy_path),
        str(cell_count_path),
        str(fp_path),
        str(sz_path),
        sat01,
        sat12,
        sat23,
        sat34,
        aat1,
        aat2,
        aat3,
        tree1,
        tree2,
        tree3,
        cc1,
        cc2,
        isl_size,
        win_size,
    )

    default_output = out_dir / 'ates_gen.tif'
    if not default_output.exists():
        raise RuntimeError(f"Ponderador output not generated: {default_output}")

    final_output = out_dir / output_name
    if final_output.resolve() != default_output.resolve():
        shutil.copy2(default_output, final_output)
    return final_output


if __name__ == "__main__":
    if forest_type in ['pcc', 'bav', 'stems', 'sen2cc', 'sen2ccc']:
        TREE1, TREE2, TREE3 = _tree_thresholds_for_forest_type(forest_type)
    else:
        raise ValueError(f"Unsupported forest_type: {forest_type}")
    AutoATES(
        wd, DEM, canopy, cell_count, FP, SZ, SAT01, SAT12, SAT23, SAT34,
        AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE
    )
