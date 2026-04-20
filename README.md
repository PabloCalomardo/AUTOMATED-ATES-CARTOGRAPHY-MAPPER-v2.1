# APP_ATES_PABLO

Pipeline en Python per generar Potential Release Areas (PRA) i productes derivats a partir d'un DEM i un raster de bosc, amb divisio per conques, simulacio Flow-Py i postprocessos finals.

## Que fa (estat actual)

El pipeline principal ([main.py](main.py)) executa 14 passos:

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
   - Sortida: `outputs/results_DDHHMM/Definitive_Layers/0_Avalanche_Shapes.geojson`.

8. Overhead exposure (nou modul)
   - Per cada `res_*`, combina `cell_counts.tif` i `z_delta.tif` amb ponderacio configurable.
   - Formula: `exposure = w_cellcount * cellcount_norm + (1 - w_cellcount) * zdelta_norm`.
    - Parametre: `--overhead-cellcount-weight` (rang `[0,1]` o valor especial `2`).
     - `1.0`: nomes Cellcount
     - `0.0`: nomes Zdelta
     - `0.1`: 10% Cellcount + 90% Zdelta
       - `2`: maxim per cel.la entre les dues capes (`max(cellcount_norm, zdelta_norm)`)
   - Escriu a `outputs/results_DDHHMM/Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`.
   - Implementat a [PostProcess_FlowPY/overhead_exposure.py](PostProcess_FlowPY/overhead_exposure.py).

9. Classificacio ATES per pendent + bosc (nou modul)
   - Primer calcula la classificacio nomes per pendent (sense efecte del bosc).
   - Sortida previa: `outputs/results_DDHHMM/Definitive_Layers/1_Slope_Classification_NoForest.tif`.
   - Calcula classes ATES finals des de DEM + PCC forestal.
   - Sortida: `outputs/results_DDHHMM/Definitive_Layers/1_SlopeandForest_Classification.tif`.
   - Implementat a [PostProcess_FlowPY/SlopeandForest_Classification.py](PostProcess_FlowPY/SlopeandForest_Classification.py).

10. Landforms per curvatura a multiples escales (nou modul)
   - Calcula landforms amb curvatura de perfil + curvatura de pla/tangencial.
   - Calcula internament les escales 5x5, 6x6, ..., 30x30 per analitzar la variabilitat multiescala.
   - Desa nomes els multiples de 5 (5x5, 10x10, 15x15, 20x20, 25x25, 30x30) amb QML de classes 1..9.
   - Les capes de landforms es guarden a `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/`.
   - Amb la sequencia de classes per cel.la (5..30), calcula una capa d'entropia normalitzada 0..1:
     - 0: classe estable entre escales
     - 1: maxima variabilitat de classe entre escales
   - Sortides principals:
   - `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_5x5.tif`
   - `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_10x10.tif`
   - `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_15x15.tif`
   - `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_20x20.tif`
   - `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_25x25.tif`
   - `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_30x30.tif`
   - `outputs/results_DDHHMM/Definitive_Layers/2_Landforms_entropy_5to30.tif`
   - Implementat a [PostProcess_FlowPY/landforms_multiscale.py](PostProcess_FlowPY/landforms_multiscale.py).

11. Terrain traps (nou modul)
      - Detecta terrain traps a partir de DEM, bosc, landforms i energia `z_delta` de Flow-Py.
   - Criteri clau: nomes classifica terrain trap on `z_delta > 0` (zona afectada per allau).
      - Tipus detectats (segons avalanche.org):
          - `Trees`
          - `Cliffs / Rocks`
          - `Gullies`
          - `Road cuts / Benches`
          - `Lakes` (sense creeks/torrents)
      - Genera 5 rasters independents + 2 bitmasks derivats:
         - `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Trees.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Cliffs.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Gullies.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_RoadCuts.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Lakes.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_trauma_bitmask.tif`
         - `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_burial_bitmask.tif`
      - Deteccio de `Gullies` basada en Stream Power Index (SPI) parametritzable:
         - `SPI = A^m * S^n` (A = drainage area D8, S = pendent en gradient)
      - A mes, per defecte es calculen i guarden 3 variants de gullies per comparacio:
         - `3_Terrain_Traps_Gullies_conservative.tif`
         - `3_Terrain_Traps_Gullies_balanced.tif`
         - `3_Terrain_Traps_Gullies_aggressive.tif`
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

13. Runout zone characteristics (nou modul)
   - Calcula una metrica continua 0..1 de dificultat de runout.

14. Ponderador weighted ATES (nou modul)
   - Executa el ponderador per basin i fusiona un raster global.
   - Usa sempre `Exposure_zdelta_cellcount.tif` com a capa d'exposicio d'entrada.

## Moduls del projecte

- [main.py](main.py): orquestrador principal (passos 1..14).
- [PREPROCESSING/preprocess.py](PREPROCESSING/preprocess.py): preprocessat DEM/forest.
- [PRAs/PRA_AutoATES-v2.0.py](PRAs/PRA_AutoATES-v2.0.py): calcul PRA.
- [PRAs/PRA_Divisor.py](PRAs/PRA_Divisor.py): divisio PRA per drenatge.
- [PRAs/PRA_Watershed_Subdivision.py](PRAs/PRA_Watershed_Subdivision.py): conques GRASS i split per basin.
- [PostProcess_FlowPY/post_FlowPy.py](PostProcess_FlowPY/post_FlowPy.py): export GeoJSON d'allaus.
- [PostProcess_FlowPY/overhead_exposure.py](PostProcess_FlowPY/overhead_exposure.py): capa d'exposicio z_delta + cell_count.
- [PostProcess_FlowPY/SlopeandForest_Classification.py](PostProcess_FlowPY/SlopeandForest_Classification.py): classes ATES de pendent+bosc.
- [PostProcess_FlowPY/landforms_multiscale.py](PostProcess_FlowPY/landforms_multiscale.py): landforms multiescala (5..30), export multiples de 5 a `2_Landforms/` i raster d'entropia 0..1.
- [PostProcess_FlowPY/terrain_traps.py](PostProcess_FlowPY/terrain_traps.py): deteccio de terrain traps (trauma/enterrament) amb raster bitmask.
- [PostProcess_FlowPY/terrain_traps.py](PostProcess_FlowPY/terrain_traps.py): deteccio de terrain traps amb 5 rasters independents i bitmasks de trauma/enterrament.
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

Aixo executa el pipeline complet (1..14).

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

On `N` pot ser de `1` a `14`.

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

# passos 1..13
python main.py --until-n 13

# passos 1..14
python main.py --until-n 14
```

### 4) Execucio amb parametres personalitzats

```bash
python main.py \
   --outputs-dir outputs/results_custom \
  --watershed-threshold 15000 \
  --watershed-memory 2000 \
  --divisor-stream-threshold 300 \
  --flowpy-alpha 22 \
   --flowpy-max-z 8000 \
   --overhead-cellcount-weight 0.5
```

Per controlar com es calcula l'overhead exposure (pas 8), usa:
- `--overhead-cellcount-weight <0..1>` per mitjana ponderada.
- `--overhead-cellcount-weight 2` per usar el maxim per cel.la entre capes normalitzades.
- En mode ponderat, el pes de `z_delta` es calcula automaticament com `1 - overhead_cellcount_weight`.

### Tots els parametres personalitzables (`python main.py --help`)

Nota:
- `--only-step6` i `--until-n` son incompatibles.
- `--zones-start-threshold` ha de ser mes gran que `--zones-ending-threshold`.

#### Control general d'execucio

- `--dem` (default: `inputs/DEM_BOW_SUMMIT.tif`)
- `--forest` (default: `inputs/FOREST_BOW_SUMMIT.tif`)
- `--forest-crs` (default: `None`; CRS opcional si el raster no porta metadades)
- `--forest-type` (default: `pcc`; opcions: `stems`, `bav`, `pcc`, `sen2cc`, `no_forest`)
- `--outputs-dir` (default: `None`; si no es passa crea `outputs/results_DDHHMM`, i amb `--only-step6` usa l'ultim `outputs/results_*`)
- `--only-step6` (flag; executa nomes el pas 6)
- `--until-n` (default: `None`; valors `1..14`)
- `--quiet` (flag; redueix logs verbosos)

#### Pas 3 - PRA (AutoATES)

- `--radius` (default: `2`)
- `--prob` (default: `0.5`)
- `--winddir` (default: `0`)
- `--windtol` (default: `180`)
- `--pra-thd` (default: `0.15`)
- `--sf` (default: `3`)

#### Pas 4 - PRA_Divisor

- `--divisor-stream-threshold` (default: `850`)
- `--divisor-channel-init-exponent` (default: `0`)
- `--divisor-channel-min-slope` (default: `0.005`)

#### Pas 5 - Watershed subdivision (GRASS)

- `--watershed-threshold` (default: `12000`)
- `--watershed-memory` (default: `500`)
- `--grass-exe` (default: `C:\Program Files\QGIS 3.40.13\bin\grass84.bat`)
- `--grass-epsg` (default: `None`; si no es passa s'infereix del DEM preprocessat)
- `--grass-db` (default: `grassdata`)
- `--grass-location` (default: `watershed_project`)
- `--grass-mapset` (default: `NOUDIRECTORIDEMAPES`)

#### Pas 6 - Flow-Py per basin

- `--flowpy-dir` (default: `Flow-py_Autoates_Editat/FlowPy_detrainment`)
- `--flowpy-alpha` (default: `22`)
- `--flowpy-exponent` (default: `8`)
- `--flowpy-flux` (default: `0.003`)
- `--flowpy-max-z` (default: `8000`)
- `--overhead-cellcount-weight` (default: `0.5`; rang: `0..1` o `2` per max mode)
- `--flowpy-infra` (default: `None`; raster d'infraestructura opcional)

#### Pas 9 - Classificacio slope + forest

- `--ates-forest-window` (default: `5`)
- `--ates-slope-sigma` (default: `1.0`)
- `--ates-forest-adjustment` (default: `paper_pra`; opcions: `legacy`, `conservative`, `paper_pra`, `paper_runout`)

#### Pas 10 - Landforms multiescala

- `--landform-windows` (default: `5,6,7,...,30`)
- `--landform-curvature-threshold` (default: `1e-4`)
- `--landform-flat-gradient-eps` (default: `1e-10`)

#### Pas 11 - Terrain traps

- `--terrain-forest-tree-threshold` (default: `35.0`)
- `--terrain-energy-trauma-threshold` (default: `0.35`)
- `--terrain-gully-energy-threshold` (default: `0.22`)
- `--terrain-gully-spi-m` (default: `1.0`)
- `--terrain-gully-spi-n` (default: `1.0`)
- `--terrain-gully-spi-threshold` (default: `0.0`; si es `0` usa percentil)
- `--terrain-gully-spi-percentile` (default: `88.0`)
- `--terrain-gully-min-drainage-area-m2` (default: `4000.0`)
- `--terrain-gully-min-slope-deg` (default: `13.0`)
- `--terrain-gully-max-slope-deg` (default: `48.0`)
- `--terrain-lake-max-slope-deg` (default: `6.0`)
- `--terrain-lake-tpi-threshold` (default: `-1.8`)
- `--terrain-lake-max-spi-threshold` (default: `0.0`; si es `0` usa percentil)
- `--terrain-lake-max-spi-percentile` (default: `35.0`)

#### Pas 12 - Zones inici/propagacio/frenada

- `--zones-start-threshold` (default: `0.99`)
- `--zones-ending-threshold` (default: `0.075`)

#### Pas 13 - Runout zone characteristics

- `--runout-flux-min-threshold` (default: `0.01`)
- `--runout-min-evidence-threshold` (default: `0.03`)

#### Pas 14 - Ponderador weighted ATES

- El pas 14 usa sempre `Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`.
- `--ponderador-forest-type` (default: `None`; opcions: `stems`, `bav`, `pcc`, `sen2cc`; si no es passa hereta `--forest-type`)
- `--ponderador-output-name` (default: `Ponderador_ATES.tif`)

Exemple complet:

```bash
python main.py \
   --dem inputs/DEM_BOW_SUMMIT.tif \
   --forest inputs/FOREST_BOW_SUMMIT.tif \
   --forest-type pcc \
   --outputs-dir outputs/results_custom \
   --watershed-threshold 15000 \
   --divisor-stream-threshold 300 \
   --flowpy-alpha 22 \
   --flowpy-max-z 8000 \
   --overhead-cellcount-weight 0.1 \
   --ates-forest-adjustment paper_pra \
   --terrain-gully-spi-percentile 90 \
   --zones-start-threshold 0.98 \
   --zones-ending-threshold 0.08
```

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

Overhead exposure ponderada (z_delta + cell_count):

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
- `outputs/results_DDHHMM/Definitive_Layers/0_Avalanche_Shapes.geojson`
- `outputs/results_DDHHMM/Definitive_Layers/1_Slope_Classification_NoForest.tif`
- `outputs/results_DDHHMM/Definitive_Layers/1_SlopeandForest_Classification.tif`
- `outputs/results_DDHHMM/Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_5x5.tif`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_5x5.qml`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_10x10.tif`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_10x10.qml`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_15x15.tif`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_15x15.qml`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_20x20.tif`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_20x20.qml`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_25x25.tif`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_25x25.qml`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_30x30.tif`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms/2_Landforms_curvature_30x30.qml`
- `outputs/results_DDHHMM/Definitive_Layers/2_Landforms_entropy_5to30.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Trees.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Cliffs.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Gullies.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Gullies_conservative.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Gullies_balanced.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Gullies_aggressive.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_RoadCuts.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_Lakes.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_trauma_bitmask.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_burial_bitmask.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_energy_proxy.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_SPI_gullies.tif`
- `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_legend.csv`
- `outputs/results_DDHHMM/Definitive_Layers/3_Terrain_Traps_stats.csv`
- `outputs/results_DDHHMM/Definitive_Layers/BasinX/Star_propagating_Ending_Zones/Ava_Y.tif`
- `outputs/results_DDHHMM/Definitive_Layers/BasinX/Star_propagating_Ending_Zones/index.csv`
- `outputs/results_DDHHMM/Definitive_Layers/BasinX/Ponderador_ATES.tif`
- `outputs/results_DDHHMM/Definitive_Layers/Ponderador_ATES.tif`

## Notes

- El pipeline esta pensat per revisar resultats pas a pas (carpetes separades per modul).
- Si canvies la instal.lacio de QGIS/GRASS, ajusta `--grass-exe`.
