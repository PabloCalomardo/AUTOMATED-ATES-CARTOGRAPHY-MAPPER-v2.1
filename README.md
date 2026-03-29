# APP_ATES_PABLO

Pipeline en Python per generar **Potential Release Areas (PRA)** i productes derivats a partir d’un **DEM** (i un raster de bosc), amb divisió per conques i exportació de capes per basin.

## Què fa (estat actual)

Executa 6 passos consecutius:

1. **Validació d’inputs**
   - Comprova que existeixen els fitxers d’entrada.
   - Escriu un manifest amb metadades a `outputs/Inputs/inputs.json`.

2. **Preprocessat DEM (simple fill)**
   - Omple `nodata` del DEM (si n’hi ha) amb un mètode senzill.
   - Sortida: `outputs/Preprocess/dem_filled_simple.tif`.

3. **Càlcul de PRA (AutoATES)**
   - Calcula `windshelter`, PRA continu i PRA binari.
   - Sortides (a `outputs/PRA_AutoATES/`):
     - `windshelter.tif`
     - `PRA_continous.tif`
     - `PRA_binary.tif`
     - `log.txt`

4. **Divisor de PRA (per conques / junctions) — WhiteboxTools**
   - Subdivideix el PRA binari en funció de la xarxa de drenatge (Strahler + junctions).
   - Sortida clau (a `outputs/PRA_Divisor/`):
     - `pra_assigned_junction.tif`

5. **Watershed subdivision (GRASS) + split PRA per basin**
  - Calcula conques (`basins.tif`) amb GRASS.
   - Genera `pra_basin_*.tif`: un raster per basin que conté **IDs reclassificats 0..k-1** dins de cada basin (fons a `nodata`).

6. **Flow-Py per basin (N execucions automàtiques)**
  - Llegeix totes les capes `pra_basin_*.tif` de `outputs/Watershed_Subdivisions/`.
  - Executa Flow-Py una vegada per capa, usant com a DEM d'entrada el DEM postprocessat (`outputs/Preprocess/dem_filled_simple.tif`).
  - Genera també `exposure.tif` dins de cada carpeta `res_*` del Flow-Py.
    - Si existeix `backcalculation.tif`, `exposure.tif` es deriva d'aquesta capa.
    - Si no, es crea com a màscara binària de `cell_counts.tif` (`>0`).
  - Desa les sortides a `outputs/Flow-Py/`, amb una carpeta pròpia per cada basin (`pra_basin_0/`, `pra_basin_1/`, ...).

## Estructura de carpetes

- `inputs/`
  - `DEM.tif` (obligatori)
  - `FOREST.tif` (requerit pel pipeline actual; es valida sempre a l’inici)

- `outputs/`
  - `Inputs/`
  - `Preprocess/`
  - `PRA_AutoATES/`
  - `PRA_Divisor/`
  - `Watershed_Subdivisions/`
  - `Flow-Py/`

## Requisits

### Programari extern

- **QGIS amb GRASS** (el pipeline crida GRASS via el `grass84.bat` de QGIS).
  - Per defecte s’utilitza:
    - `C:\Program Files\QGIS 3.40.13\bin\grass84.bat`

### Python

- Python 3.x (recomanat: la mateixa versió que tens a `.venv`).
- Paquets típics utilitzats:
  - `rasterio`, `numpy`
  - `whitebox` (WhiteboxTools)
  - `scikit-image` (pot ser necessari com a fallback dins del mòdul PRA)

Nota: hi ha un fitxer de dependències a `Flow-py_Autoates_Editat/FlowPy_detrainment/requirements.txt` (encoding no estàndard). Si ja tens `.venv` configurat i el pipeline et funciona, no cal reinstal·lar res.

## Com executar el pipeline

Des de l’arrel del projecte (on hi ha `main.py`):

```bash
python main.py
```

Per executar **només el pas 6 (Flow-Py)** amb resultats ja calculats (sense repetir passos 1-5):

```bash
python main.py --only-step6
```

Prerequisits per `--only-step6`:
- `outputs/Preprocess/dem_filled_simple.tif`
- `outputs/Watershed_Subdivisions/pra_basin_*.tif`

Això assumeix els defaults:

- Inputs
  - `--dem inputs/DEM.tif`
  - `--forest inputs/FOREST.tif`
  - `--forest-type stems`
  - `--outputs-dir outputs`

- Paràmetres PRA (AutoATES)
  - `--radius 6`
  - `--prob 0.5`
  - `--winddir 0`
  - `--windtol 180`
  - `--pra-thd 0.15`
  - `--sf 3`

- Paràmetres PRA_Divisor
  - `--divisor-stream-threshold 210`
  - `--divisor-channel-init-exponent 1`
  - `--divisor-channel-min-slope 1e-4`

- Paràmetres Watershed (GRASS)
  - `--watershed-threshold 12000`
  - `--watershed-memory 500`
  - `--grass-exe "C:\\Program Files\\QGIS 3.40.13\\bin\\grass84.bat"`
  - `--grass-epsg 25833`
  - `--grass-db grassdata`
  - `--grass-location watershed_project`
  - `--grass-mapset NOUDIRECTORIDEMAPES`

### Exemple amb overrides

```bash
python main.py \
  --outputs-dir outputs \
  --watershed-threshold 15000 \
  --watershed-memory 2000 \
  --divisor-stream-threshold 300
```

## Sortides principals

- `outputs/Preprocess/dem_filled_simple.tif`
- `outputs/PRA_AutoATES/PRA_binary.tif`
- `outputs/PRA_Divisor/pra_assigned_junction.tif`
- `outputs/Watershed_Subdivisions/basins.tif`
- `outputs/Watershed_Subdivisions/pra_basin_0.tif`, `pra_basin_1.tif`, ...
- `outputs/Flow-Py/pra_basin_0/res_YYYYMMDD_HHMMSS/*`
- `outputs/Flow-Py/pra_basin_1/res_YYYYMMDD_HHMMSS/*`
- ...

En cada `res_*` del Flow-Py tindràs, a més de les capes estàndard, una capa `exposure.tif`.

## Paràmetres nous (pas 6: Flow-Py)

- `--flowpy-dir` (default: `Flow-py_Autoates_Editat/FlowPy_detrainment`)
- `--flowpy-alpha` (default: `25`)
- `--flowpy-exponent` (default: `8`)
- `--flowpy-flux` (default: `0.003`)
- `--flowpy-max-z` (default: `8848`)
- `--flowpy-forest` (default: `inputs/FOREST.tif`)
- `--flowpy-infra` (opcional)
- `--only-step6` (executa exclusivament el pas 6)

## Execució dels scripts per separat (opcional)

- PRA divisor:

```bash
python PRAs/PRA_Divisor.py --help
```

- Watershed subdivision + split PRA per basin:

```bash
python PRAs/PRA_Watershed_Subdivision.py --help
```

- Postprocess de Flow-Py (formes d'allau a partir de `source_ids_bitmask.tif`):

```bash
python PostProcess_FlowPY/post_FlowPy.py
```

Resultat per cada `res_*` de `outputs/Flow-Py/*`:
- `outputs/Avalanche_Shapes/avalanche_shapes.geojson`

Aquest GeoJSON únic conté només:
- `feature_type=avalanche`: polígons de cada allau individual (`avalanche_id`)

## Notes

- El pipeline està pensat per revisar resultats pas a pas (cada pas escriu a la seva carpeta dins `outputs/`).
- Si canvies el path de QGIS/GRASS, fes servir `--grass-exe`.
