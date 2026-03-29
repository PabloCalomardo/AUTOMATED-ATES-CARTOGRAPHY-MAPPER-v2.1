#!/usr/bin/env python3
"""PRAs · Watershed subdivision

Crea conques (basins) a partir del DEM d'entrada usant GRASS GIS.

Outputs:
- Raster de conques (basins)

Pas extra final:
- Subdivideix el raster `pra_assigned_junction.tif` (PRA_Divisor) en un GeoTIFF
    per cada basin. Dins de cada raster, els junction IDs es remapejen a 0..k-1.
"""

import os
import sys
import subprocess
import argparse
from collections import deque

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# --- Executable GRASS (dins QGIS) ---
GRASS_EXE = r"C:\Program Files\QGIS 3.40.13\bin\grass84.bat"

# --- Configuració ---
INPUTS_DIR      = os.path.abspath("inputs")
OUTPUT_DIR      = os.path.abspath(os.path.join("outputs", "Watershed_Subdivisions"))


def find_first_existing(base_dir: str, candidates: list[str]) -> str | None:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


DEM_INPUT = find_first_existing(INPUTS_DIR, ["DEM.tif", "dem.tif"])
GRASS_DB        = os.path.abspath("grassdata")
LOCATION        = "watershed_project"
MAPSET          = "NOUDIRECTORIDEMAPES"
RASTER_NAME     = "PEDRA_2METRES_ANSIDIF_AGRR_GAUSS_3"
FILLED_DEM_NAME = "PEDRA_2METRES_ANSIDIF_AGRR_GAUSS_3_FILLED"
FLOW_DIR_NAME   = "FLOWDIR_TMP"
BASIN_OUTPUT    = "BASINS_OUTPUTS"
MEMORY          = 500
THRESHOLD       = 12000

# --- Paths ---
LOCATION_PATH   = os.path.join(GRASS_DB, LOCATION)
PERMANENT_PATH  = os.path.join(LOCATION_PATH, "PERMANENT")
MAPSET_PATH     = os.path.join(LOCATION_PATH, MAPSET)


BASIN_TIF  = os.path.join(OUTPUT_DIR, "basins.tif")


PRA_ASSIGNED_JUNCTION_TIF = os.path.abspath(
    os.path.join("outputs", "PRA_Divisor", "pra_assigned_junction.tif")
)


def configure_runtime_paths(*, dem_path: str | None, pra_assigned_path: str | None, output_dir: str | None) -> None:
    """Override default IO paths at runtime (used by global main pipeline)."""
    global DEM_INPUT
    global OUTPUT_DIR, BASIN_TIF
    global PRA_ASSIGNED_JUNCTION_TIF

    if dem_path:
        DEM_INPUT = os.path.abspath(dem_path)
    if pra_assigned_path:
        PRA_ASSIGNED_JUNCTION_TIF = os.path.abspath(pra_assigned_path)
    if output_dir:
        OUTPUT_DIR = os.path.abspath(output_dir)
        BASIN_TIF = os.path.join(OUTPUT_DIR, "basins.tif")


def configure_runtime_settings(
    *,
    grass_exe: str | None,
    grass_epsg: str | None,
    grass_db: str | None,
    grass_location: str | None,
    grass_mapset: str | None,
    watershed_threshold: int | None,
    watershed_memory: int | None,
) -> None:
    """Override internal GRASS + watershed parameters (defaults preserved)."""
    global GRASS_EXE
    global GRASS_DB, LOCATION, MAPSET
    global LOCATION_PATH, PERMANENT_PATH, MAPSET_PATH
    global THRESHOLD, MEMORY
    global _GRASS_EPSG

    if grass_exe:
        GRASS_EXE = grass_exe
    if grass_db:
        GRASS_DB = os.path.abspath(grass_db)
    if grass_location:
        LOCATION = grass_location
    if grass_mapset:
        MAPSET = grass_mapset
    if watershed_threshold is not None:
        THRESHOLD = int(watershed_threshold)
    if watershed_memory is not None:
        MEMORY = int(watershed_memory)
    if grass_epsg:
        _GRASS_EPSG = str(grass_epsg)

    # Recompute dependent GRASS paths.
    LOCATION_PATH = os.path.join(GRASS_DB, LOCATION)
    PERMANENT_PATH = os.path.join(LOCATION_PATH, "PERMANENT")
    MAPSET_PATH = os.path.join(LOCATION_PATH, MAPSET)


# Default EPSG for GRASS location creation (string as GRASS expects)
_GRASS_EPSG = "25833"


def _recompute_grass_paths() -> None:
    global LOCATION_PATH, PERMANENT_PATH, MAPSET_PATH
    LOCATION_PATH = os.path.join(GRASS_DB, LOCATION)
    PERMANENT_PATH = os.path.join(LOCATION_PATH, "PERMANENT")
    MAPSET_PATH = os.path.join(LOCATION_PATH, MAPSET)


def infer_dem_epsg(dem_path: str) -> str | None:
    try:
        with rasterio.open(dem_path) as src:
            if src.crs is None:
                return None
            epsg = src.crs.to_epsg()
            if epsg is None:
                return None
            return str(epsg)
    except Exception:
        return None


def read_location_epsg(permanent_path: str) -> str | None:
    if not os.path.exists(permanent_path):
        return None
    proc = subprocess.run(
        [GRASS_EXE, permanent_path, "--exec", "g.proj", "-g"],
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return None

    epsg = None
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if line.startswith("epsg="):
            epsg = line.split("=", 1)[1].strip()
            break
        if line.startswith("srid="):
            srid = line.split("=", 1)[1].strip()
            if srid.upper().startswith("EPSG:"):
                epsg = srid.split(":", 1)[1].strip()
                break
    return epsg


def split_pras_by_basin(basins_tif: str, pra_assigned_tif: str, out_dir: str) -> list[str]:
    """Exporta un raster per basin preservant IDs de PRA i remapejant-los.

    - `basins_tif`: raster de conques (valors categòrics; NULL fora).
    - `pra_assigned_tif`: raster int (0=no PRA; >0 id junction assignat).
    - `out_dir`: carpeta on escriure `pra_basin_<idx>.tif`.

    Cada `pra_basin_<idx>.tif` conté només les PRAs que cauen dins del basin.
    Els valors (junction IDs) es remapejen a consecutius 0..k-1 dins de cada
    raster de sortida. El fons (no PRA) queda com a nodata.

    Retorna la llista de fitxers creats.
    """
    if not os.path.exists(basins_tif):
        raise FileNotFoundError(f"No s'ha trobat basins.tif: {basins_tif}")
    if not os.path.exists(pra_assigned_tif):
        raise FileNotFoundError(f"No s'ha trobat pra_assigned_junction.tif: {pra_assigned_tif}")

    with rasterio.open(pra_assigned_tif) as src_p:
        pra = src_p.read(1)
        pra_profile = src_p.profile.copy()

    with rasterio.open(basins_tif) as src_b:
        basins_src = src_b.read(1)
        basins_profile = src_b.profile.copy()

    target_shape = pra.shape
    target_transform = pra_profile.get("transform")
    target_crs = pra_profile.get("crs")

    src_transform = basins_profile.get("transform")
    src_crs = basins_profile.get("crs")
    src_nodata = basins_profile.get("nodata")

    same_grid = (
        basins_src.shape == target_shape
        and src_crs == target_crs
        and src_transform == target_transform
    )

    # Alinea basins al grid del PRA si cal (nearest per raster categòric).
    dst_nodata = -9999
    if same_grid:
        basins = basins_src.astype(np.int32, copy=False)
    else:
        print("Alineant basins.tif al grid del PRA (reproject/resample nearest)...")
        basins = np.full(target_shape, dst_nodata, dtype=np.int32)
        try:
            reproject(
                source=basins_src,
                destination=basins,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
                resampling=Resampling.nearest,
            )
        except Exception as e:
            # Alguns GeoTIFF (p.ex. AutoATES) porten la CRS com a EngineeringCRS.
            # PROJ no pot trobar una operació de transformació i peta; en aquest cas
            # assumim que les coordenades són les mateixes i només cal re-mostrejar.
            print(f"AVÍS: reprojecció CRS fallida ({e}); reintentant assumint mateixa CRS...")
            assumed_crs = src_crs or target_crs
            reproject(
                source=basins_src,
                destination=basins,
                src_transform=src_transform,
                src_crs=assumed_crs,
                dst_transform=target_transform,
                dst_crs=assumed_crs,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
                resampling=Resampling.nearest,
            )

    # Llista d'IDs reals del raster (ordenats); l'index (0..n-1) és el que fem servir al nom.
    basin_ids = np.unique(basins)
    basin_ids = [int(v) for v in basin_ids.tolist() if v is not None]
    basin_ids = [v for v in basin_ids if v != dst_nodata]
    basin_ids = sorted(set(basin_ids))

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        # QGIS/GDAL poden crear sidecars (p.ex. .tif.aux.xml) amb estadístiques.
        # Si re-generem el .tif però el sidecar queda, QGIS pot mostrar IDs inexistents.
        if name.lower().startswith("pra_basin_"):
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass
    created: list[str] = []

    pra_mask = pra != 0

    out_nodata = -9999
    out_profile = pra_profile.copy()
    out_profile.update(dtype="int32", count=1, compress="deflate", nodata=out_nodata)

    out_index = 0
    for basin_value in basin_ids:
        basin_mask = basins == basin_value
        mask = basin_mask & pra_mask
        if not np.any(mask):
            continue

        # IDs originals de PRA (junction id) dins d'aquest basin.
        original_ids = np.unique(pra[mask])
        original_ids = original_ids[original_ids != 0]
        if original_ids.size == 0:
            continue
        original_ids = np.sort(original_ids.astype(np.int64, copy=False))

        out = np.full(pra.shape, out_nodata, dtype=np.int32)

        # Remapeig vectoritzat: cada id original -> index 0..k-1.
        # searchsorted funciona perquè original_ids està ordenat.
        remapped = np.searchsorted(original_ids, pra[mask].astype(np.int64, copy=False)).astype(np.int32)
        out[mask] = remapped

        out_path = os.path.join(out_dir, f"pra_basin_{out_index}.tif")
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out, 1)
        created.append(out_path)
        out_index += 1

    return created


def ensure_full_dem_basin_coverage(basins_tif: str, dem_tif: str) -> int:
    """Fill uncovered DEM areas with new basin IDs using connected components.

    Returns number of new basins created.
    """
    with rasterio.open(dem_tif) as src_dem:
        dem_valid = ~src_dem.read(1, masked=True).mask

    with rasterio.open(basins_tif) as src_b:
        basins_ma = src_b.read(1, masked=True)
        profile = src_b.profile.copy()

    basins = basins_ma.filled(0).astype(np.int32, copy=False)
    has_basin = (~basins_ma.mask) & (basins > 0)
    uncovered = dem_valid & (~has_basin)

    if not np.any(uncovered):
        # Ensure pixels outside DEM footprint are explicit nodata.
        out_nodata = profile.get("nodata")
        if out_nodata is None:
            out_nodata = 0
            profile.update(nodata=out_nodata)
        basins_out = basins.copy()
        basins_out[~dem_valid] = int(out_nodata)
        with rasterio.open(basins_tif, "w", **profile) as dst:
            dst.write(basins_out, 1)
        return 0

    max_id = int(basins[has_basin].max()) if np.any(has_basin) else 0
    rows, cols = basins.shape
    visited = np.zeros(basins.shape, dtype=bool)
    new_count = 0

    # 8-neighbour connectivity, consistent with D8-style neighborhoods.
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    seed_rows, seed_cols = np.where(uncovered)
    for sr, sc in zip(seed_rows, seed_cols):
        if visited[sr, sc]:
            continue
        if not uncovered[sr, sc]:
            visited[sr, sc] = True
            continue

        max_id += 1
        new_count += 1
        q: deque[tuple[int, int]] = deque([(int(sr), int(sc))])
        visited[sr, sc] = True

        while q:
            r, c = q.popleft()
            basins[r, c] = max_id

            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if visited[nr, nc] or not uncovered[nr, nc]:
                    continue
                visited[nr, nc] = True
                q.append((nr, nc))

    out_nodata = profile.get("nodata")
    if out_nodata is None:
        out_nodata = 0
        profile.update(nodata=out_nodata)
    basins[~dem_valid] = int(out_nodata)

    with rasterio.open(basins_tif, "w", **profile) as dst:
        dst.write(basins, 1)

    return new_count


def run(args):
    """Executa una comanda GRASS i atura si hi ha error.
    - stdin=DEVNULL evita que el 'pause on error' del bat bloquegi el procés.
    - El 'pause' del bat pot resetejar ERRORLEVEL a 0, per tant comprovem
      també si hi ha 'ERROR:' a la sortida."""
    result = subprocess.run(args, text=True, capture_output=True,
                            stdin=subprocess.DEVNULL)
    combined = (result.stdout + result.stderr).strip()
    has_error = result.returncode != 0 or "\nERROR:" in combined or combined.startswith("ERROR:")
    if has_error:
        print(combined)
        print(f"ERROR detectat (codi {result.returncode})")
        sys.exit(1)
    return combined


def run_watershed():
    """Executa r.watershed amb el threshold fix i retorna el nombre de conques."""
    print(f"Executant r.watershed amb threshold={THRESHOLD}...")
    run([GRASS_EXE, MAPSET_PATH, "--exec",
         "r.watershed", "--overwrite",
         f"elevation={FILLED_DEM_NAME}@{MAPSET}",
         f"threshold={THRESHOLD}",
         f"basin={BASIN_OUTPUT}",
         f"memory={MEMORY}"])

    # Comptar conques úniques amb r.stats -n (una línia per valor no-null)
    stats = subprocess.run(
        [GRASS_EXE, MAPSET_PATH, "--exec", "r.stats", "-n", BASIN_OUTPUT],
        text=True, capture_output=True, stdin=subprocess.DEVNULL
    )
    n_basins = len([l for l in stats.stdout.splitlines() if l.strip()])
    print(f"Conques generades: {n_basins}")
    return n_basins


def main():
    global _GRASS_EPSG, LOCATION

    parser = argparse.ArgumentParser(description="Create drainage basins and split PRA-by-basin rasters.")
    parser.add_argument("--dem", default=None, help="DEM path to use (default: inputs/DEM.tif)")
    parser.add_argument(
        "--pra-assigned",
        default=None,
        help="Path to PRA_Divisor output pra_assigned_junction.tif (default: outputs/PRA_Divisor/pra_assigned_junction.tif)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output folder for basin/PRA results (default: outputs/Watershed_Subdivisions)",
    )

    # --- Watershed/GRASS parameterization (defaults match internal constants)
    parser.add_argument("--watershed-threshold", type=int, default=THRESHOLD)
    parser.add_argument("--watershed-memory", type=int, default=MEMORY)
    parser.add_argument("--grass-exe", default=GRASS_EXE)
    parser.add_argument(
        "--grass-epsg",
        default=None,
        help="EPSG for GRASS location. If omitted, inferred from DEM when possible.",
    )
    parser.add_argument("--grass-db", default=GRASS_DB)
    parser.add_argument("--grass-location", default=LOCATION)
    parser.add_argument("--grass-mapset", default=MAPSET)
    args = parser.parse_args()

    configure_runtime_paths(dem_path=args.dem, pra_assigned_path=args.pra_assigned, output_dir=args.out_dir)
    configure_runtime_settings(
        grass_exe=args.grass_exe,
        grass_epsg=args.grass_epsg,
        grass_db=args.grass_db,
        grass_location=args.grass_location,
        grass_mapset=args.grass_mapset,
        watershed_threshold=args.watershed_threshold,
        watershed_memory=args.watershed_memory,
    )

    if not DEM_INPUT or not os.path.exists(DEM_INPUT):
        wanted = os.path.join(INPUTS_DIR, "DEM.tif")
        print(f"ERROR: No s'ha trobat el DEM a {wanted}")
        sys.exit(1)

    if args.grass_epsg is None:
        inferred_epsg = infer_dem_epsg(DEM_INPUT)
        if inferred_epsg is not None:
            _GRASS_EPSG = inferred_epsg
            print(f"EPSG inferit del DEM: {_GRASS_EPSG}")
        else:
            print(f"AVÍS: no s'ha pogut inferir EPSG del DEM; s'usarà l'EPSG per defecte: {_GRASS_EPSG}")

    os.makedirs(GRASS_DB, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing_epsg = read_location_epsg(PERMANENT_PATH)
    if existing_epsg is not None and existing_epsg != _GRASS_EPSG:
        old_location = LOCATION
        LOCATION = f"{old_location}_epsg{_GRASS_EPSG}"
        _recompute_grass_paths()
        print(
            "AVÍS: la location GRASS existent té un CRS diferent "
            f"(EPSG:{existing_epsg} != EPSG:{_GRASS_EPSG}). "
            f"S'usarà una location nova: {LOCATION}"
        )

    # 1. Crear la localització GRASS amb EPSG (ETRS89 UTM Zone 33N per defecte)
    #    El DEM té una SRS codificada com a Engineering CRS que GRASS rebutja;
    #    per això creem la localització directament per codi EPSG.
    if not os.path.exists(PERMANENT_PATH):
        print(f"Creant localització GRASS (EPSG:{_GRASS_EPSG})...")
        run([GRASS_EXE, "-c", f"EPSG:{_GRASS_EPSG}", LOCATION_PATH, "--exec", "g.version"])

    # 2. Crear el mapset si no existeix
    if not os.path.exists(MAPSET_PATH):
        print(f"Creant mapset '{MAPSET}'...")
        run([GRASS_EXE, PERMANENT_PATH, "--exec", "g.mapset", "-c", f"mapset={MAPSET}"])

    # 3. Importar el DEM dins del mapset
    #    -o: override projection check (la SRS del DEM és ETRS89 UTM 33N però
    #    està codificada com a Engineering CRS, cosa que GRASS no pot parsejar)
    print("Important el DEM...")
    run([GRASS_EXE, MAPSET_PATH, "--exec",
         "r.in.gdal", "--overwrite", "-o",
         f"input={DEM_INPUT}", f"output={RASTER_NAME}"])

    # IMPORTANT: GRASS modules operate on the current computational region.
    # Ensure the region matches the imported DEM BEFORE doing any processing.
    run([GRASS_EXE, MAPSET_PATH, "--exec",
         "g.region", f"raster={RASTER_NAME}"])

    # 4. Filling previ del DEM per eliminar depressions tancades
    print("Fent filling del DEM...")
    run([GRASS_EXE, MAPSET_PATH, "--exec",
         "r.fill.dir", "--overwrite",
         f"input={RASTER_NAME}",
         f"output={FILLED_DEM_NAME}",
         f"direction={FLOW_DIR_NAME}"])

    # 5. Ajustar la regió computacional al DEM omplert
    print("Ajustant regió...")
    run([GRASS_EXE, MAPSET_PATH, "--exec",
         "g.region", f"raster={FILLED_DEM_NAME}"])

    # 6. Obtenir el total de cel·les del DEM (informatiu)
    info_out = subprocess.run(
        [GRASS_EXE, MAPSET_PATH, "--exec", "r.info", "-g", FILLED_DEM_NAME],
        text=True, capture_output=True, stdin=subprocess.DEVNULL
    ).stdout
    total_cells = 1
    for line in info_out.splitlines():
        if line.startswith("rows="):
            rows = int(line.split("=")[1])
        elif line.startswith("cols="):
            cols = int(line.split("=")[1])
    total_cells = rows * cols
    print(f"DEM: {rows} files x {cols} columnes = {total_cells} cel·les totals")

    # 7. Execucio directa amb threshold fix
    n_basins = run_watershed()

    # 8. Exportar el ràster de conques a output/basins.tif
    print(f"\nExportant {n_basins} conques a {BASIN_TIF}...")
    run([GRASS_EXE, MAPSET_PATH, "--exec",
         "r.out.gdal", "--overwrite",
         f"input={BASIN_OUTPUT}",
         f"output={BASIN_TIF}",
         "format=GTiff",
         "type=Int32",
         "createopt=COMPRESS=LZW"])

    # 8.1 Ensure every valid DEM cell belongs to a basin.
    # If watershed leaves uncovered islands (common with non-rectangular DEM masks),
    # assign each connected uncovered region to a new basin ID.
    new_basins = ensure_full_dem_basin_coverage(BASIN_TIF, DEM_INPUT)
    if new_basins > 0:
        print(f"Afegides {new_basins} conques noves per components connexes (zones DEM sense basin).")

    print(f"\nFet! {n_basins} conques exportades a: {BASIN_TIF}  (threshold={THRESHOLD})")

    # 9. Subdividir PRAs del PRA_Divisor per basin (rasters binaris)
    try:
        created = split_pras_by_basin(BASIN_TIF, PRA_ASSIGNED_JUNCTION_TIF, OUTPUT_DIR)
        if created:
            print(f"Rasters PRA per basin creats: {len(created)} (a {OUTPUT_DIR})")
        else:
            print("Cap basin conté PRA (no s'ha creat cap raster binari).")
    except FileNotFoundError as e:
        # Permet executar la subdivisió de conques encara que PRA_Divisor no s'hagi executat.
        print(f"AVÍS: pas PRA-per-basin omès: {e}")


if __name__ == "__main__":
    main()
