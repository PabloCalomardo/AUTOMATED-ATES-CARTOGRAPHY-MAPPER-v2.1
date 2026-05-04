import numpy as np
import rasterio, rasterio.mask
from osgeo import gdal
import os
import shutil
from skimage import morphology
import csv
import scipy.ndimage
from rasterio.fill import fillnodata
from pathlib import Path
import re

# --- Set Input Files
DEM = 'test-data/Bow Summit/dem.tif'
canopy = 'test-data/Bow Summit/forest.tif'
cell_count = 'test-data/Bow Summit/Overhead.tif' # replace with z_delta in next iteration
FP = 'test-data/Bow Summit/FP_int16.tif'
SZ = 'test-data/Bow Summit/pra_binary.tif'
forest_type = 'bav' # 'bav', 'stems', 'pcc', 'sen2cc'

wd = 'test-data/Bow Summit/outputs'

# --- Set default input parameters

# Moving window size to smooth slope angle layer for calcuation of Class 4 extreme
WIN_SIZE= 3

# --- Define slope angle Thresholds
# Should I increase these to capture more real world numbers or keep values based on Consensus map test areas?
# Class 0 / 1 Slope Angle Threshold (Default 15)
SAT01 = 15
# Class 1 / 2 Slope Angle Threshold (Default 18)
SAT12 = 18
# Class 2 / 3 Slope Angle Threshold (Default 28)
SAT23 = 28
# Class 3 / 4 Slope Angle Threshold (Default 39)
# This is calculated on a smoothed raster layer, so the slope angle value is not representative of real world values
SAT34 = 39 # stereo

# --- Define alpha angle thresholds
# Class 1 Alpha Angle Threshold (Default 18)
AAT1 = 18
# Class 2 Alpha Angle Threshold (Default 25)
AAT2 = 24
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
CC1 = 5
CC2 = 40

# --- Threshold for number of cells in a cluster to be removed (generalization)
ISL_SIZE = 30000


def _tree_thresholds_for_forest_type(forest_type_value):
    forest_type_norm = str(forest_type_value).lower()
    if forest_type_norm == 'pcc':
        return 10, 50, 65
    if forest_type_norm == 'bav':
        return 10, 20, 25
    if forest_type_norm == 'stems':
        return 100, 250, 500
    if forest_type_norm in ['sen2cc', 'sen2ccc']:
        return 20, 60, 85
    raise ValueError(
        "Unsupported forest_type for ponderador. Use one of: stems, bav, pcc, sen2cc"
    )


def _same_grid(ref_src, other_src):
    return (
        ref_src.width == other_src.width
        and ref_src.height == other_src.height
        and ref_src.transform == other_src.transform
        and ref_src.crs == other_src.crs
    )


def _parse_basin_id(out_dir_path):
    match = re.match(r"^Basin(\d+)$", out_dir_path.name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _parse_zone_avalanche_id(path_obj):
    match = re.match(r"^Ava_(\d+)\.tif$", path_obj.name, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _find_first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _load_entropy_cluster_mask(
    definitive_layers_dir,
    ref_src,
    entropy_threshold,
    min_cluster_cells,
):
    clustered_candidates = [
        definitive_layers_dir / 'Landforms_entropy_5to30_clustered.tif',
        definitive_layers_dir / '2_Landforms_entropy_5to30_clustered.tif',
        definitive_layers_dir / '2_Landforms' / 'Landforms_entropy_5to30_clustered.tif',
        definitive_layers_dir / '2_Landforms' / '2_Landforms_entropy_5to30_clustered.tif',
    ]
    clustered_path = _find_first_existing(clustered_candidates)
    if clustered_path is not None:
        with rasterio.open(str(clustered_path)) as src_clustered:
            if not _same_grid(ref_src, src_clustered):
                return None
            arr = src_clustered.read(1)
            nodata = src_clustered.nodata

        if nodata is None:
            return arr > 0
        return np.logical_and(arr != nodata, arr > 0)

    entropy_candidates = [
        definitive_layers_dir / '2_Landforms_entropy_5to30.tif',
        definitive_layers_dir / 'Landforms_entropy_5to30.tif',
        definitive_layers_dir / '2_Landforms' / '2_Landforms_entropy_5to30.tif',
        definitive_layers_dir / '2_Landforms' / 'Landforms_entropy_5to30.tif',
    ]
    entropy_path = _find_first_existing(entropy_candidates)
    if entropy_path is None:
        return None

    with rasterio.open(str(entropy_path)) as src_entropy:
        if not _same_grid(ref_src, src_entropy):
            return None
        entropy = src_entropy.read(1).astype(np.float32, copy=False)
        nodata = src_entropy.nodata

    if nodata is None:
        valid = np.isfinite(entropy)
    elif isinstance(nodata, float) and np.isnan(nodata):
        valid = np.isfinite(entropy)
    else:
        valid = np.logical_and(np.isfinite(entropy), entropy != nodata)

    high_entropy = np.logical_and(valid, entropy >= float(entropy_threshold))
    if min_cluster_cells is None or int(min_cluster_cells) <= 1:
        return high_entropy

    labels, num_labels = scipy.ndimage.label(high_entropy)
    if num_labels == 0:
        return high_entropy

    out_mask = np.zeros(high_entropy.shape, dtype=bool)
    min_cells = int(min_cluster_cells)
    for label_id in range(1, num_labels + 1):
        mask = labels == label_id
        if np.count_nonzero(mask) >= min_cells:
            out_mask[mask] = True
    return out_mask


def _reclassify_class4_by_runout(
    ates_path,
    basin_dir,
    definitive_layers_dir,
    landform_window,
    safe_classes,
    unsafe_classes,
    safe_pct_threshold,
    unsafe_pct_keep_threshold,
    entropy_pct_keep_threshold,
    entropy_max_for_downgrade,
    entropy_threshold,
    entropy_min_cluster_cells,
):
    zones_dir = basin_dir / 'Star_propagating_Ending_Zones'
    if not zones_dir.exists():
        return ates_path

    basin_id = _parse_basin_id(basin_dir)
    if basin_id is None:
        return ates_path

    landforms_path = (
        definitive_layers_dir
        / '2_Landforms'
        / f'2_Landforms_curvature_{int(landform_window)}x{int(landform_window)}.tif'
    )
    if not landforms_path.exists():
        return ates_path

    safe_values = np.array(list(safe_classes), dtype=np.int16)
    unsafe_values = np.array(list(unsafe_classes), dtype=np.int16)

    decision_codes = {
        'keep_4_rule_not_met': 1,
        'keep_4_high_unsafe_pct': 2,
        'keep_4_entropy_cluster_presence': 3,
        'keep_4_no_start_overlap': 4,
        'keep_4_empty_propagation': 5,
        'downgrade_to_3': 6,
    }

    with rasterio.open(str(ates_path)) as src_ates:
        ates = src_ates.read(1)
        nodata_ates = src_ates.nodata
        ates_profile = src_ates.profile.copy()

    with rasterio.open(str(landforms_path)) as src_landforms:
        landforms = src_landforms.read(1)
        landforms_nodata = src_landforms.nodata

    entropy_cluster_mask = _load_entropy_cluster_mask(
        definitive_layers_dir=definitive_layers_dir,
        ref_src=src_ates,
        entropy_threshold=entropy_threshold,
        min_cluster_cells=entropy_min_cluster_cells,
    )
    if entropy_cluster_mask is None:
        entropy_cluster_mask = np.zeros(ates.shape, dtype=bool)

    ava_start_masks = {}
    ava_prop_masks = {}
    for ava_path in sorted(zones_dir.glob('Ava_*.tif')):
        avalanche_id = _parse_zone_avalanche_id(ava_path)
        if avalanche_id is None:
            continue

        with rasterio.open(str(ava_path)) as src_zone:
            zones = src_zone.read(1)
            nodata_zone = src_zone.nodata

        if nodata_zone is None:
            valid_zone = np.isfinite(zones)
        else:
            valid_zone = zones != nodata_zone

        start_mask = np.logical_and(valid_zone, zones == 1)
        prop_mask = np.logical_and(valid_zone, zones == 2)
        if np.count_nonzero(start_mask) == 0 and np.count_nonzero(prop_mask) == 0:
            continue

        ava_start_masks[avalanche_id] = start_mask
        ava_prop_masks[avalanche_id] = prop_mask

    if not ava_start_masks:
        return ates_path

    class4_mask = ates == 4
    if nodata_ates is not None:
        class4_mask = np.logical_and(class4_mask, ates != nodata_ates)

    labels, num_labels = scipy.ndimage.label(class4_mask)
    if num_labels == 0:
        return ates_path

    cluster_id_layer = np.zeros(ates.shape, dtype=np.int32)
    decision_layer = np.zeros(ates.shape, dtype=np.uint8)

    if landforms_nodata is None:
        valid_landforms = np.isfinite(landforms)
    else:
        valid_landforms = landforms != landforms_nodata

    audit_rows = []
    reclassified_clusters = 0

    for cluster_id in range(1, num_labels + 1):
            cluster_mask = labels == cluster_id
            cluster_cells = int(np.count_nonzero(cluster_mask))
            if cluster_cells == 0:
                continue

            cluster_id_layer[cluster_mask] = cluster_id

            selected_ava = None
            best_overlap = 0
            for avalanche_id, start_mask in ava_start_masks.items():
                overlap = int(np.count_nonzero(np.logical_and(cluster_mask, start_mask)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    selected_ava = avalanche_id

            if selected_ava is None or best_overlap == 0:
                decision_layer[cluster_mask] = decision_codes['keep_4_no_start_overlap']
                audit_rows.append({
                    'basin_id': basin_id,
                    'cluster_id': cluster_id,
                    'cluster_cells': cluster_cells,
                    'selected_avalanche_id': '',
                    'start_overlap_cells': 0,
                    'propagation_cells': 0,
                    'safe_pct_789': 0.0,
                    'unsafe_pct_123': 0.0,
                    'entropy_cluster_pct': 0.0,
                    'decision': 'keep_4',
                    'reason': 'no_start_overlap',
                    'decision_code': decision_codes['keep_4_no_start_overlap'],
                })
                continue

            propagation_mask = ava_prop_masks.get(selected_ava)
            if propagation_mask is None:
                propagation_mask = np.zeros(cluster_mask.shape, dtype=bool)

            propagation_mask = np.logical_and(propagation_mask, valid_landforms)
            propagation_cells = int(np.count_nonzero(propagation_mask))
            if propagation_cells == 0:
                decision_layer[cluster_mask] = decision_codes['keep_4_empty_propagation']
                audit_rows.append({
                    'basin_id': basin_id,
                    'cluster_id': cluster_id,
                    'cluster_cells': cluster_cells,
                    'selected_avalanche_id': selected_ava,
                    'start_overlap_cells': best_overlap,
                    'propagation_cells': 0,
                    'safe_pct_789': 0.0,
                    'unsafe_pct_123': 0.0,
                    'entropy_cluster_pct': 0.0,
                    'decision': 'keep_4',
                    'reason': 'empty_propagation',
                    'decision_code': decision_codes['keep_4_empty_propagation'],
                })
                continue

            prop_landforms = landforms[propagation_mask].astype(np.int16, copy=False)
            safe_count = int(np.count_nonzero(np.isin(prop_landforms, safe_values)))
            unsafe_count = int(np.count_nonzero(np.isin(prop_landforms, unsafe_values)))
            entropy_count = int(np.count_nonzero(np.logical_and(propagation_mask, entropy_cluster_mask)))

            safe_pct = 100.0 * safe_count / float(propagation_cells)
            unsafe_pct = 100.0 * unsafe_count / float(propagation_cells)
            entropy_pct = 100.0 * entropy_count / float(propagation_cells)

            decision = 'keep_4'
            reason = 'rule_not_met'

            if unsafe_pct > float(unsafe_pct_keep_threshold):
                decision = 'keep_4'
                reason = 'high_unsafe_pct'
                decision_layer[cluster_mask] = decision_codes['keep_4_high_unsafe_pct']
            elif safe_pct >= float(safe_pct_threshold) and entropy_pct <= float(entropy_max_for_downgrade):
                ates[cluster_mask] = 3
                reclassified_clusters += 1
                decision = 'downgrade_to_3'
                reason = 'high_safe_low_entropy'
                decision_layer[cluster_mask] = decision_codes['downgrade_to_3']
            elif safe_pct >= 60.0 and entropy_pct >= float(entropy_pct_keep_threshold):
                decision = 'keep_4'
                reason = 'entropy_cluster_presence'
                decision_layer[cluster_mask] = decision_codes['keep_4_entropy_cluster_presence']
            else:
                decision_layer[cluster_mask] = decision_codes['keep_4_rule_not_met']

            audit_rows.append({
                'basin_id': basin_id,
                'cluster_id': cluster_id,
                'cluster_cells': cluster_cells,
                'selected_avalanche_id': selected_ava,
                'start_overlap_cells': best_overlap,
                'propagation_cells': propagation_cells,
                'safe_pct_789': round(safe_pct, 3),
                'unsafe_pct_123': round(unsafe_pct, 3),
                'entropy_cluster_pct': round(entropy_pct, 3),
                'decision': decision,
                'reason': reason,
                'decision_code': int(decision_layer[cluster_mask][0]),
            })

    if reclassified_clusters > 0:
        with rasterio.open(str(ates_path), 'r+') as dst_ates:
            dst_ates.write(ates.astype(np.int16, copy=False), 1)

    audit_csv = basin_dir / 'class4_runout_reclassification.csv'
    with audit_csv.open('w', newline='', encoding='utf-8') as fp_csv:
        writer = csv.DictWriter(
            fp_csv,
            fieldnames=[
                'basin_id',
                'cluster_id',
                'cluster_cells',
                'selected_avalanche_id',
                'start_overlap_cells',
                'propagation_cells',
                'safe_pct_789',
                'unsafe_pct_123',
                'entropy_cluster_pct',
                'decision',
                'reason',
                'decision_code',
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    cluster_id_path = basin_dir / 'class4_clusters_id.tif'
    cluster_profile = ates_profile.copy()
    cluster_profile.update(dtype='int32', nodata=0, count=1, compress='deflate')
    with rasterio.open(str(cluster_id_path), 'w', **cluster_profile) as dst_cluster:
        dst_cluster.write(cluster_id_layer.astype(np.int32, copy=False), 1)

    decision_path = basin_dir / 'class4_reclass_decision.tif'
    decision_profile = ates_profile.copy()
    decision_profile.update(dtype='uint8', nodata=0, count=1, compress='deflate')
    with rasterio.open(str(decision_path), 'w', **decision_profile) as dst_decision:
        dst_decision.write(decision_layer.astype(np.uint8, copy=False), 1)

    reason_counts = {
        'rule_not_met': 0,
        'high_unsafe_pct': 0,
        'entropy_cluster_presence': 0,
        'no_start_overlap': 0,
        'empty_propagation': 0,
        'high_safe_low_entropy': 0,
    }
    for row in audit_rows:
        reason = row.get('reason')
        if reason in reason_counts:
            reason_counts[reason] += 1

    total_clusters = len(audit_rows)
    downgraded_clusters = int(np.count_nonzero(decision_layer == decision_codes['downgrade_to_3']))
    kept_clusters = total_clusters - downgraded_clusters

    summary_csv = basin_dir / 'class4_runout_reclassification_summary.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as fp_summary:
        writer = csv.DictWriter(
            fp_summary,
            fieldnames=[
                'basin_id',
                'total_class4_clusters',
                'downgraded_to_3_clusters',
                'kept_as_4_clusters',
                'downgraded_pct',
                'kept_pct',
                'count_keep_rule_not_met',
                'count_keep_high_unsafe_pct',
                'count_keep_entropy_cluster_presence',
                'count_keep_no_start_overlap',
                'count_keep_empty_propagation',
                'count_downgrade_high_safe_low_entropy',
            ],
        )
        writer.writeheader()
        writer.writerow({
            'basin_id': basin_id,
            'total_class4_clusters': total_clusters,
            'downgraded_to_3_clusters': downgraded_clusters,
            'kept_as_4_clusters': kept_clusters,
            'downgraded_pct': 0.0 if total_clusters == 0 else round(100.0 * downgraded_clusters / total_clusters, 3),
            'kept_pct': 0.0 if total_clusters == 0 else round(100.0 * kept_clusters / total_clusters, 3),
            'count_keep_rule_not_met': reason_counts['rule_not_met'],
            'count_keep_high_unsafe_pct': reason_counts['high_unsafe_pct'],
            'count_keep_entropy_cluster_presence': reason_counts['entropy_cluster_presence'],
            'count_keep_no_start_overlap': reason_counts['no_start_overlap'],
            'count_keep_empty_propagation': reason_counts['empty_propagation'],
            'count_downgrade_high_safe_low_entropy': reason_counts['high_safe_low_entropy'],
        })

    legend_txt = basin_dir / 'class4_reclass_decision_legend.txt'
    legend_txt.write_text(
        "0 = no original class-4 cluster\n"
        "1 = kept class 4 (rule_not_met)\n"
        "2 = kept class 4 (high_unsafe_pct)\n"
        "3 = kept class 4 (entropy_cluster_presence)\n"
        "4 = kept class 4 (no_start_overlap)\n"
        "5 = kept class 4 (empty_propagation)\n"
        "6 = downgraded to class 3 (high_safe_low_entropy)\n",
        encoding='utf-8',
    )

    return ates_path


def run_autoates_weighted(
    dem_path,
    canopy_path,
    cell_count_path,
    fp_path,
    sz_path,
    out_dir,
    forest_type,
    output_name='Ponderador_ATES.tif',
    class4_reclass_enabled=True,
    class4_landform_window=10,
    class4_safe_classes=(7, 8, 9),
    class4_unsafe_classes=(1, 2, 3),
    class4_safe_pct_threshold=80.0,
    class4_unsafe_pct_keep_threshold=30.0,
    class4_entropy_pct_keep_threshold=8.0,
    class4_entropy_max_for_downgrade=10.0,
    class4_entropy_threshold=0.55,
    class4_entropy_min_cluster_cells=25,
):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    dem_path = Path(dem_path)
    canopy_path = Path(canopy_path)
    cell_count_path = Path(cell_count_path)
    fp_path = Path(fp_path)
    sz_path = Path(sz_path)

    for input_path in [dem_path, canopy_path, cell_count_path, fp_path, sz_path]:
        if not input_path.exists():
            raise FileNotFoundError(f"Missing ponderador input raster: {input_path}")

    tree1, tree2, tree3 = _tree_thresholds_for_forest_type(forest_type)

    global SZ
    sz_binary_path = out_dir_path / 'sz_binary_for_ponderador.tif'
    with rasterio.open(str(sz_path)) as src_sz:
        sz_arr = src_sz.read(1)
        sz_profile = src_sz.profile.copy()
        sz_nodata = src_sz.nodata

    if sz_nodata is None:
        sz_binary = (sz_arr > 0).astype(np.int16)
    else:
        sz_binary = np.where(sz_arr == sz_nodata, 0, (sz_arr > 0).astype(np.int16)).astype(np.int16)

    sz_profile.update(dtype='int16', nodata=0, count=1, compress='deflate')
    with rasterio.open(str(sz_binary_path), 'w', **sz_profile) as dst_sz:
        dst_sz.write(sz_binary, 1)

    SZ = str(sz_binary_path)

    AutoATES(
        str(out_dir_path),
        str(dem_path),
        str(canopy_path),
        str(cell_count_path),
        str(fp_path),
        SAT01,
        SAT12,
        SAT23,
        SAT34,
        AAT1,
        AAT2,
        AAT3,
        tree1,
        tree2,
        tree3,
        CC1,
        CC2,
        ISL_SIZE,
        WIN_SIZE,
    )

    generated_output = out_dir_path / 'ates_gen.tif'
    if not generated_output.exists():
        raise RuntimeError(f"Expected ponderador output not generated: {generated_output}")

    if class4_reclass_enabled:
        definitive_layers_dir = out_dir_path.parent
        _reclassify_class4_by_runout(
            ates_path=generated_output,
            basin_dir=out_dir_path,
            definitive_layers_dir=definitive_layers_dir,
            landform_window=class4_landform_window,
            safe_classes=class4_safe_classes,
            unsafe_classes=class4_unsafe_classes,
            safe_pct_threshold=class4_safe_pct_threshold,
            unsafe_pct_keep_threshold=class4_unsafe_pct_keep_threshold,
            entropy_pct_keep_threshold=class4_entropy_pct_keep_threshold,
            entropy_max_for_downgrade=class4_entropy_max_for_downgrade,
            entropy_threshold=class4_entropy_threshold,
            entropy_min_cluster_cells=class4_entropy_min_cluster_cells,
        )

    final_output = out_dir_path / output_name
    if generated_output.resolve() != final_output.resolve():
        shutil.copyfile(generated_output, final_output)
    return final_output

def AutoATES(wd, DEM, canopy, cell_count, FP, SAT01, SAT12, SAT23, SAT34, AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE):
    
    # --- Write input parameters to CSV file
    labels = ['DEM', 'canopy', 'cell_count', 'FP', 'SAT01', 'SAT12', 'SAT23', 'SAT34', 'AAT1', 'AAT2', 'AAT3', 'TREE1', 'TREE2', 'TREE3', 'CC1', 'CC2', 'ISL_SIZE', 'WIN_SIZE']
    csvRow = [DEM, canopy, cell_count, FP, SAT01, SAT12, SAT23, SAT34, AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE]
    csvfile = os.path.join(wd, "inputpara.csv")
    with open(csvfile, "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(labels)
        wr.writerow(csvRow)
    
    # --- Calculate slope angle
    def calculate_slope(DEM):
        gdal.DEMProcessing(os.path.join(wd, 'slope.tif'), DEM, 'slope')
        with rasterio.open(os.path.join(wd, 'slope.tif')) as src:
            slope = src.read()
            profile = src.profile
        return slope, profile

    slope, profile = calculate_slope(DEM)
    slope = slope.astype('int16')
    
    slope_nd = np.where(slope < 0, 0, slope)
    
    # Optional function to calculat class 4 slope using a neighborhood function - controlled by WIN_SIZE input parameter
    # If WIN_SIZE is set to 1 this function does not do anything to the SAT34 threshold calculation
    slope_smooth = scipy.ndimage.uniform_filter(slope_nd, size = WIN_SIZE, mode = 'nearest')
    
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
    flow_py18[np.where((flow_py18 >= 0) & (flow_py18 < 90))] = 1 # Changed to 0 from AAT1 because we are not using Non-Avalanche Terrain - class 0

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
        array = src.read()
        array = array.astype('int16')
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
    forest = rasterio.open(canopy).read()
    forest_open=forest
    forest_open[forest_open > TREE1] = -1
    forest_open[(forest_open >= 0) & (forest_open <= TREE1)] = 10
        
    forest = rasterio.open(canopy).read()
    forest_sparse=forest
    forest_sparse[forest_sparse > TREE2] = -1
    forest_sparse[forest <= TREE1] = -1
    forest_sparse[(forest > TREE1) & (forest <= TREE2)] = 20
    
    forest = rasterio.open(canopy).read()
    forest_dense=forest
    forest_dense[forest_dense > TREE3] = -1
    forest_dense[forest_dense <= TREE2] = -1
    forest_dense[(forest_dense > TREE2) & (forest_dense <= TREE3)] = 30
    
    forest = rasterio.open(canopy).read()
    forest_vdense=forest
    forest_vdense[forest_vdense < TREE3] = -1
    forest_vdense[forest_vdense >= TREE3] = 40
    
    src2=np.maximum(forest_open, forest_sparse)
    src2=np.maximum(src2, forest_dense)
    src2=np.maximum(src2, forest_vdense)
    
    with rasterio.open(os.path.join(wd, "forest_reclass.tif"), 'w', **profile) as dst:
        dst.write(src2)
    
    # --- Add PRA criteria
    src3 = rasterio.open(SZ)
    src3 = src3.read()
    
    src3[np.where(0 == src3)] = 0
    src3[np.where(1 == src3)] = 100

    with rasterio.open(os.path.join(wd, "SZ_reclass.tif"), 'w', **profile) as dst:
        dst.write(src3)

    array = np.sum([src1, src2, src3], axis=0)

    array[np.where(array == 10)] = 0
    array[np.where(array == 11)] = 1
    array[np.where(array == 12)] = 2
    array[np.where(array == 13)] = 3
    array[np.where(array == 14)] = 4
    array[np.where(array == 20)] = 0
    array[np.where(array == 21)] = 1
    array[np.where(array == 22)] = 1
    array[np.where(array == 23)] = 2
    array[np.where(array == 24)] = 3
    array[np.where(array == 30)] = 0
    array[np.where(array == 31)] = 1
    array[np.where(array == 32)] = 1
    array[np.where(array == 33)] = 1
    array[np.where(array == 34)] = 3
    array[np.where(array == 40)] = 0
    array[np.where(array == 41)] = 1
    array[np.where(array == 42)] = 1
    array[np.where(array == 43)] = 1
    array[np.where(array == 44)] = 2
    array[np.where(array == 110)] = 0
    array[np.where(array == 111)] = 1
    array[np.where(array == 112)] = 2
    array[np.where(array == 113)] = 3
    array[np.where(array == 114)] = 4
    array[np.where(array == 120)] = 0
    array[np.where(array == 121)] = 1
    array[np.where(array == 122)] = 1
    array[np.where(array == 123)] = 2
    array[np.where(array == 124)] = 3
    array[np.where(array == 130)] = 0
    array[np.where(array == 131)] = 1
    array[np.where(array == 132)] = 1
    array[np.where(array == 133)] = 2
    array[np.where(array == 134)] = 3
    array[np.where(array == 140)] = 0
    array[np.where(array == 141)] = 1
    array[np.where(array == 142)] = 1
    array[np.where(array == 143)] = 2
    array[np.where(array == 144)] = 2
    array[np.where(array < 0)] = 0

    array = array.astype('int16')

    # --- Save raster to path
    with rasterio.open(os.path.join(wd, "merge_all.tif"), "w", **profile) as dest:
        dest.write(array)

    # --- Remove clusters of raster cells smaller than ISL_SIZE
    raster = gdal.Open(DEM)
    gt =raster.GetGeoTransform()
    pixelSizeX = gt[1]
    pixelSizeY =-gt[5]
    num_cells = np.around(ISL_SIZE / (pixelSizeX * pixelSizeY)) 
    #print(num_cells)
    # --- Open file
    src1 = rasterio.open(os.path.join(wd, "merge_all.tif"))
    src1 = src1.read(1)

    # --- Change values to prepare for morphology and rasterio.fill
    src1 = src1 + 1
    src1 = src1.reshape(1, src1.shape[0], src1.shape[1])

    # --- Same as region group in arcmap. Each cluster gets a value between 1 and num_labels (number of clusters)
    # 20210430 JS changed connectivity to 2
    lab, num_labels = morphology.label(src1, connectivity=2, return_num=True)

    rg = np.arange(1, num_labels+1, 1)

    # --- Loop through all clusters and assign all clusters with less then ISL_SIZE to the value 0 (set null)
    for i in rg:
        occurrences = np.count_nonzero(lab == i)
        if occurrences < num_cells:
            lab[np.where(lab == i)] = 0

    # --- Save as dtype int16
    lab = lab.astype('int16')

    search_dist = num_cells / 4
    #search_dist = num_cells

    # --- This algorithm will interpolate values for all designated nodata pixels (marked by zeros) (nibble)
    data = rasterio.fill.fillnodata(src1, lab, max_search_distance=search_dist, smoothing_iterations=0)
    
    # --- Change values back to standardized way of plotting ATES (0, 1, 2, 3 and 4)
    data = data - 1
    data[np.where(data == 0)] = -9999
    data = data.astype('int16')
    profile.update({"driver": "GTiff", "nodata": -9999, 'dtype': 'int16'})

    # --- Save raster to path
    with rasterio.open(os.path.join(wd, "ates_gen.tif"), "w", **profile) as dest:
        dest.write(data)

if __name__ == "__main__":
    AutoATES(wd, DEM, canopy, cell_count, FP, SAT01, SAT12, SAT23, SAT34, AAT1, AAT2, AAT3, TREE1, TREE2, TREE3, CC1, CC2, ISL_SIZE, WIN_SIZE)
