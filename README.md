# APP_ATES_PABLO

Pipeline en Python per generar Potential Release Areas (PRA) i productes derivats a partir d'un DEM i un raster de bosc, amb divisio per conques, simulacio Flow-Py i postprocessos finals.

## Que fa (estat actual)

El pipeline principal ([main.py](main.py)) executa 12 passos:

1. Validacio d'inputs
   - Comprova que existeixen DEM i forest.
   - Escriu metadades a `outputs/Inputs/inputs.json`.

2. Preprocessat DEM i forest
   - Omple nodata del DEM (`dem_filled_simple.tif`).
   - Alinea el forest al DEM (`forest_aligned.tif`).
   - Normalitza forest per Flow-Py (`FOREST_NORMALIZED.tif`, escala 0..1).

3. Calcul PRA (AutoATES)
   - Genera `windshelter.tif`, `PRA_continous.tif`, `PRA_binary.tif`, `log.txt`.

4. Divisor PRA (WhiteboxTools)
   - Segmenta PRA binari per xarxa de drenatge.
   - Sortida clau: `outputs/PRA_Divisor/pra_assigned_junction.tif`.

5. Watershed subdivision (GRASS) + split per basin
   - Calcula conques i exporta `pra_basin_*.tif`.

6. Flow-Py per basin (N execucions automatiques)
   - Executa Flow-Py per cada `pra_basin_*.tif`.
   - Per cada resultat `res_*`, crea `exposure.tif`:
     - prioritat 1: copia `backcalculation.tif` si existeix,
     - prioritat 2: mascara binaria de `cell_counts.tif` (`>0`).

7. Postprocess Flow-Py a GeoJSON unic
   - Exporta poligons d'allaus des de `source_ids_bitmask.tif`.
   - Sortida: `outputs/results_DDHHMM/Definitive_Layers/avalanche_shapes.geojson`.

8. Overhead exposure (nou modul)
   - Per cada `res_*`, combina `cell_counts.tif` i `z_delta.tif`.
   - Escriu a `outputs/results_DDHHMM/Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`.
   - Implementat a [PostProcess_FlowPY/overhead_exposure.py](PostProcess_FlowPY/overhead_exposure.py).

9. Classificacio ATES per pendent + bosc (nou modul)
   - Calcula classes ATES finals des de DEM + PCC forestal.
   - Sortida: `outputs/results_DDHHMM/Definitive_Layers/SlopeandForest_Classification.tif`.
   - Implementat a [PostProcess_FlowPY/SlopeandForest_Classification.py](PostProcess_FlowPY/SlopeandForest_Classification.py).

10. Landforms per curvatura a multiples escales (nou modul)
   - Calcula landforms amb curvatura de perfil + curvatura de pla/tangencial.
   - Genera 3 capes amb diferents veinatges: 3x3, 6x6 i 12x12.
   - Per cada capa també genera un fitxer QML amb la simbologia de classes 1..9.
   - Sortides a `outputs/results_DDHHMM/Definitive_Layers/`:
     - `Landforms_curvature_3x3.tif`
     - `Landforms_curvature_6x6.tif`
     - `Landforms_curvature_12x12.tif`
   - Implementat a [PostProcess_FlowPY/landforms_multiscale.py](PostProcess_FlowPY/landforms_multiscale.py).

11. Terrain traps (nou modul)
      - Detecta terrain traps a partir de DEM, bosc, landforms i energia `z_delta` de Flow-Py.
   - Criteri clau: nomes classifica terrain trap on `z_delta > 0` (zona afectada per allau).
      - Tipus detectats (segons avalanche.org):
          - `Trees`
          - `Cliffs / Rocks`
          - `Gullies`
          - `Road cuts / Benches`
          - `Lakes / Creeks`
      - Genera un raster bitmask i capes derivades de trauma/enterrament a:
         - `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_bitmask.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_trauma_amplifiers.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_burial_amplifiers.tif`
      - Implementat a [PostProcess_FlowPY/terrain_traps.py](PostProcess_FlowPY/terrain_traps.py).

12. Zones d'inici, propagacio i frenada per allau (nou modul)
    - Usa `flux.tif` (energia 0..1) i `source_ids_bitmask.tif` per basin.
    - Llindars configurables:
       - inici: `flux >= 0.99` (default)
       - frenada: `flux < 0.075` (default)
       - propagacio: la resta
    - Guarda un raster per cada allau dins de cada basin.
    - Ruta de sortida per basin:
      - `outputs/results_DDHHMM/Definitive_Layers/BasinX/Star_propagating_Ending_Zones/Ava_Y.tif`
      - `outputs/results_DDHHMM/Definitive_Layers/BasinX/Star_propagating_Ending_Zones/index.csv`
    - Implementat a [PostProcess_FlowPY/start_propagating_ending_zones.py](PostProcess_FlowPY/start_propagating_ending_zones.py).

## Moduls del projecte

- [main.py](main.py): orquestrador principal (passos 1..12).
- [PREPROCESSING/preprocess.py](PREPROCESSING/preprocess.py): preprocessat DEM/forest.
- [PRAs/PRA_AutoATES-v2.0.py](PRAs/PRA_AutoATES-v2.0.py): calcul PRA.
- [PRAs/PRA_Divisor.py](PRAs/PRA_Divisor.py): divisio PRA per drenatge.
- [PRAs/PRA_Watershed_Subdivision.py](PRAs/PRA_Watershed_Subdivision.py): conques GRASS i split per basin.
- [PostProcess_FlowPY/post_FlowPy.py](PostProcess_FlowPY/post_FlowPy.py): export GeoJSON d'allaus.
- [PostProcess_FlowPY/overhead_exposure.py](PostProcess_FlowPY/overhead_exposure.py): capa d'exposicio z_delta + cell_count.
- [PostProcess_FlowPY/SlopeandForest_Classification.py](PostProcess_FlowPY/SlopeandForest_Classification.py): classes ATES de pendent+bosc.
- [PostProcess_FlowPY/landforms_multiscale.py](PostProcess_FlowPY/landforms_multiscale.py): landforms multiescala per curvatura (3x3, 6x6, 12x12).
- [PostProcess_FlowPY/terrain_traps.py](PostProcess_FlowPY/terrain_traps.py): deteccio de terrain traps (trauma/enterrament) amb raster bitmask.
- [PostProcess_FlowPY/start_propagating_ending_zones.py](PostProcess_FlowPY/start_propagating_ending_zones.py): zones inici/propagacio/frenada per allau i basin.
- [Flow-py_Autoates_Editat/FlowPy_detrainment/main.py](Flow-py_Autoates_Editat/FlowPy_detrainment/main.py): motor Flow-Py (invocat dinamicament).

## Estructura de carpetes

- `inputs/`
  - `DEM_BOW_SUMMIT.tif` (default)
  - `FOREST_BOW_SUMMIT.tif` (default)

- `outputs/`
   - `results_DDHHMM/`
  - `Inputs/`
  - `Preprocess/`
  - `PRA_AutoATES/`
  - `PRA_Divisor/`
  - `Watershed_Subdivisions/`
  - `Flow-Py/`
  - `Definitive_Layers/`

## Requisits

### Programari extern

- QGIS amb GRASS (default: `C:\Program Files\QGIS 3.40.13\bin\grass84.bat`).

### Python

- Python 3.x (idealment el `.venv` del projecte).
- Dependencias habituals:
  - `numpy`, `rasterio`
  - `whitebox`
  - `scikit-image`
  - `scipy` (pas 9: filtres gaussians i finestres)

## Maneres d'executar el projecte

Des de l'arrel del projecte:

```bash
python main.py
```

Aixo executa el pipeline complet (1..12).

Per defecte, cada execucio crea una carpeta nova a `outputs/results_DDHHMM` per evitar sobreescriptures.

### 1) Pipeline complet

```bash
python main.py
```

### 2) Nomes pas 6 (Flow-Py)

```bash
python main.py --only-step6
```

Prerequisits de `--only-step6`:
- `<outputs-dir>/Preprocess/dem_filled_simple.tif`
- `<outputs-dir>/Watershed_Subdivisions/pra_basin_*.tif`
- Si s'ha passat `--forest`, tambe `<outputs-dir>/Preprocess/FOREST_NORMALIZED.tif`

Nota addicional: si no passes `--outputs-dir`, `--only-step6` utilitza automaticament el darrer `outputs/results_*`.

Nota: `--only-step6` i `--until-n` son incompatibles.

### 3) Executar fins a un pas concret

```bash
python main.py --until-n N
```

On `N` pot ser de `1` a `12`.

Exemples:

```bash
# passos 1..3
python main.py --until-n 3

# passos 1..6
python main.py --until-n 6

# passos 1..7
python main.py --until-n 7

# passos 1..8
python main.py --until-n 8

# passos 1..10
python main.py --until-n 10

# passos 1..11
python main.py --until-n 11

# passos 1..12
python main.py --until-n 12
```

### 4) Execucio amb parametres personalitzats

```bash
python main.py \
   --outputs-dir outputs/results_custom \
  --watershed-threshold 15000 \
  --watershed-memory 2000 \
  --divisor-stream-threshold 300 \
  --flowpy-alpha 22 \
  --flowpy-max-z 8000
```

## Defaults actuals importants

Inputs:
- `--dem inputs/DEM_BOW_SUMMIT.tif`
- `--forest inputs/FOREST_BOW_SUMMIT.tif`
- `--forest-type pcc`
- `--outputs-dir` (opcional; per defecte es crea `outputs/results_DDHHMM`)

PRA (pas 3):
- `--radius 6`
- `--prob 0.6`
- `--winddir 0`
- `--windtol 180`
- `--pra-thd 0.15`
- `--sf 3`

PRA_Divisor (pas 4):
- `--divisor-stream-threshold 850`
- `--divisor-channel-init-exponent 0`
- `--divisor-channel-min-slope 0.005`

Watershed (pas 5):
- `--watershed-threshold 12000`
- `--watershed-memory 500`
- `--grass-exe C:\Program Files\QGIS 3.40.13\bin\grass84.bat`
- `--grass-epsg` (si no es passa, s'infereix del DEM preprocessat)
- `--grass-db grassdata`
- `--grass-location watershed_project`
- `--grass-mapset NOUDIRECTORIDEMAPES`

Flow-Py (pas 6):
- `--flowpy-dir Flow-py_Autoates_Editat/FlowPy_detrainment`
- `--flowpy-alpha 22`
- `--flowpy-exponent 8`
- `--flowpy-flux 0.003`
- `--flowpy-max-z 8000`
- `--flowpy-infra` (opcional)

Classificacio ATES (pas 9):
- `--ates-forest-window 5`
- `--ates-slope-sigma 1.0`
- `--ates-forest-adjustment paper_pra`

Landforms (pas 10):
- `--landform-windows 3,6,12`
- `--landform-curvature-threshold 1e-4`
- `--landform-flat-gradient-eps 1e-10`

Terrain traps (pas 11):
- `--terrain-forest-tree-threshold 35.0`
- `--terrain-energy-trauma-threshold 0.35`
- `--terrain-gully-energy-threshold 0.2`

Zones inici/propagacio/frenada (pas 12):
- `--zones-start-threshold 0.99`
- `--zones-ending-threshold 0.075`

## Execucio de moduls per separat (opcional)

PRA divisor:

```bash
python PRAs/PRA_Divisor.py --help
```

Watershed subdivision:

```bash
python PRAs/PRA_Watershed_Subdivision.py --help
```

Postprocess Flow-Py (GeoJSON d'allaus):

```bash
python PostProcess_FlowPY/post_FlowPy.py --help
```

Overhead exposure (z_delta + cell_count):

```bash
python PostProcess_FlowPY/overhead_exposure.py --help
```

Classificacio slope+forest:

```bash
python PostProcess_FlowPY/SlopeandForest_Classification.py --help
```

Landforms multiescala per curvatura:

```bash
python PostProcess_FlowPY/landforms_multiscale.py --help
```

Terrain traps:

```bash
python PostProcess_FlowPY/terrain_traps.py --help
```

Zones inici/propagacio/frenada per allau:

```bash
python PostProcess_FlowPY/start_propagating_ending_zones.py --help
```

## Sortides principals

- `outputs/results_DDHHMM/Inputs/inputs.json`
- `outputs/results_DDHHMM/Preprocess/dem_filled_simple.tif`
- `outputs/results_DDHHMM/Preprocess/forest_aligned.tif`
- `outputs/results_DDHHMM/Preprocess/FOREST_NORMALIZED.tif`
- `outputs/results_DDHHMM/PRA_AutoATES/PRA_binary.tif`
- `outputs/results_DDHHMM/PRA_Divisor/pra_assigned_junction.tif`
- `outputs/results_DDHHMM/Watershed_Subdivisions/basins.tif`
- `outputs/results_DDHHMM/Watershed_Subdivisions/pra_basin_*.tif`
- `outputs/results_DDHHMM/Flow-Py/pra_basin_*/res_YYYYMMDD_HHMMSS/*`
- `outputs/results_DDHHMM/Definitive_Layers/avalanche_shapes.geojson`
- `outputs/results_DDHHMM/Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Landforms_curvature_3x3.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Landforms_curvature_3x3.qml`
- `outputs/results_DDHHMM/Definitive_Layers/Landforms_curvature_6x6.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Landforms_curvature_6x6.qml`
- `outputs/results_DDHHMM/Definitive_Layers/Landforms_curvature_12x12.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Landforms_curvature_12x12.qml`
- `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_bitmask.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_bitmask.qml`
- `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_trauma_amplifiers.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_burial_amplifiers.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_energy_proxy.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_legend.csv`
- `outputs/results_DDHHMM/Definitive_Layers/Terrain_Traps_stats.csv`
- `outputs/results_DDHHMM/Definitive_Layers/BasinX/Star_propagating_Ending_Zones/Ava_Y.tif`
- `outputs/results_DDHHMM/Definitive_Layers/BasinX/Star_propagating_Ending_Zones/index.csv`
- `outputs/results_DDHHMM/Definitive_Layers/SlopeandForest_Classification.tif`

## Notes

- El pipeline esta pensat per revisar resultats pas a pas (carpetes separades per modul).
- Si canvies la instal.lacio de QGIS/GRASS, ajusta `--grass-exe`.
