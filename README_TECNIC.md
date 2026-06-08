# README tecnic

Aquest document descriu el funcionament intern del repositori. El README principal es curt; aquest fitxer recull el detall tecnic per revisar, modificar o auditar el pipeline.

## Objectiu del pipeline

El projecte genera cartografia ATES automatitzada combinant:

- DEM.
- Raster de densitat o cobertura forestal.
- PRA calculat amb AutoATES.
- Propagacio Flow-Py per conques.
- Capa d'exposicio derivada de `cell_counts.tif` i `z_delta.tif`.
- Landforms multiescala.
- Terrain traps.
- Zones d'inici, propagacio i frenada.
- Ponderador final ATES per evidencies combinades.

L'orquestrador principal es [main.py](main.py).

## Estructura del repositori

- `main.py`: executa els passos 1..14 i gestiona carpetes d'output.
- `run_experiments.py`: llanca conjunts d'experiments.
- `PREPROCESSING/preprocess.py`: preprocessat DEM/forest.
- `PRAs/PRA_AutoATES-v2.0.py`: calcul PRA.
- `PRAs/PRA_Divisor.py`: divisio inicial del PRA.
- `PRAs/PRA_Watershed_Subdivision.py`: conques GRASS i export `pra_basin_*.tif`.
- `Flow-py_Autoates_Editat/FlowPy_detrainment/`: Flow-Py modificat.
- `PostProcess_FlowPY/overhead_exposure.py`: exposicio ponderada `cell_counts` + `z_delta`.
- `PostProcess_FlowPY/SlopeandForest_Classification.py`: classificacio pendent + bosc.
- `PostProcess_FlowPY/landforms_multiscale.py`: landforms i entropia.
- `PostProcess_FlowPY/terrain_traps.py`: terrain traps.
- `PostProcess_FlowPY/start_propagating_ending_zones.py`: zones per allau.
- `PostProcess_FlowPY/runout_zone_characteristics.py`: dificultat continua de runout.
- `Ponderador/AutoATES_classifier.py`: ponderador final en mode `hybrid`.
- `Ponderador/Filters.py`: filtres de smoothing del raster final.
- `Ponderador/PONDERADOR_CHANGES.md`: historial del ponderador.

## Execucio

Pipeline complet:

```bash
python main.py
```

Amb DEM i forest:

```bash
python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif
```

Amb delimitacio espacial:

```bash
python main.py --study-area inputs/DELIMITACIO.shp
```

Fins a un pas concret:

```bash
python main.py --until-n 10
```

Nomes pas 6 amb outputs previs:

```bash
python main.py --only-step6
```

## Carpetes d'output

Si no es passa `--outputs-dir`, una execucio completa crea:

```text
outputs/results_DDHHMM/
```

Subcarpetes principals:

- `Inputs/`: metadades d'entrada.
- `Preprocess/`: DEM omplert, forest alineat i forest normalitzat.
- `PRA_AutoATES/`: PRA continu i binari.
- `PRA_Divisor/`: PRA assignat/dividit.
- `Watershed_Subdivisions/`: conques i `pra_basin_*.tif`.
- `Flow-Py/`: resultats Flow-Py per basin.
- `Definitive_Layers/`: productes finals i intermedis finals.

## Passos del pipeline

### 1. Validacio d'inputs

Comprova que existeixen DEM, forest i delimitacio opcional. Escriu `Inputs/inputs.json`.

### 2. Preprocessat

Genera:

- `Preprocess/dem_filled_simple.tif`
- `Preprocess/forest_aligned.tif`
- `Preprocess/FOREST_NORMALIZED.tif`

Si es passa `--study-area`, DEM i forest es retallen al poligon.

### 3. PRA AutoATES

Executa `PRAs/PRA_AutoATES-v2.0.py` i genera:

- `PRA_AutoATES/windshelter.tif`
- `PRA_AutoATES/PRA_continous.tif`
- `PRA_AutoATES/PRA_binary.tif`
- `PRA_AutoATES/log.txt`

### 4. PRA Divisor

Segmenta o assigna el PRA segons drenatge i genera:

- `PRA_Divisor/pra_assigned_junction.tif`

### 5. Watershed subdivision

Usa GRASS per generar conques i exportar:

- `Watershed_Subdivisions/basins.tif`
- `Watershed_Subdivisions/pra_basin_*.tif`

### 6. Flow-Py per basin

Per cada `pra_basin_*.tif`, executa Flow-Py en una carpeta:

```text
Flow-Py/pra_basin_X/res_YYYYMMDD_HHMMSS/
```

Capes Flow-Py rellevants:

- `FP_travel_angle.tif`
- `cell_counts.tif`
- `z_delta.tif`
- `flux.tif`
- `source_ids_bitmask.tif`

Tambe crea `exposure.tif` quan pot, prioritzant `backcalculation.tif`; si no existeix, usa mascara binaria de `cell_counts.tif > 0`.

### 7. GeoJSON d'allaus

Postprocessa Flow-Py i escriu:

- `Definitive_Layers/0_Avalanche_Shapes.geojson`

### 8. Overhead exposure

Combina `cell_counts.tif` i `z_delta.tif` per basin:

```text
exposure = w_cellcount * cellcount_norm + (1 - w_cellcount) * zdelta_norm
```

Si `--overhead-cellcount-weight 2`, usa:

```text
exposure = max(cellcount_norm, zdelta_norm)
```

Sortida:

- `Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`

### 9. Classificacio pendent + bosc

Genera:

- `Definitive_Layers/1_Slope_Classification_NoForest.tif`
- `Definitive_Layers/1_SlopeandForest_Classification.tif`

### 10. Landforms multiescala

Calcula landforms amb finestres 5..30 per defecte. Desa multiples de 5:

- `Definitive_Layers/2_Landforms/2_Landforms_curvature_5x5.tif`
- `Definitive_Layers/2_Landforms/2_Landforms_curvature_10x10.tif`
- `Definitive_Layers/2_Landforms/2_Landforms_curvature_15x15.tif`
- `Definitive_Layers/2_Landforms/2_Landforms_curvature_20x20.tif`
- `Definitive_Layers/2_Landforms/2_Landforms_curvature_25x25.tif`
- `Definitive_Layers/2_Landforms/2_Landforms_curvature_30x30.tif`

Tambe genera:

- `Definitive_Layers/2_Landforms_entropy_5to30.tif`

### 11. Terrain traps

Detecta terrain traps nomes dins zona afectada per allau (`z_delta > 0`):

- Trees
- Cliffs / Rocks
- Gullies
- Road cuts / Benches
- Lakes

Sortides principals:

- `Definitive_Layers/3_TerrainTraps/3_Terrain_Traps_*.tif`
- `Definitive_Layers/3_Terrain_Traps_trauma_bitmask.tif`
- `Definitive_Layers/3_Terrain_Traps_burial_bitmask.tif`
- `Definitive_Layers/3_Terrain_Traps_stats.csv`
- `Definitive_Layers/3_Terrain_Traps_legend.csv`

### 12. Zones d'inici, propagacio i frenada

Usa `flux.tif` i `source_ids_bitmask.tif` per escriure un raster per allau:

```text
Definitive_Layers/BasinX/Star_propagating_Ending_Zones/Ava_Y.tif
```

Codificacio:

- `1`: inici
- `2`: propagacio
- `3`: frenada

### 13. Runout zone characteristics

Calcula una metrica continua 0..1:

- `Definitive_Layers/6_Runout_Zone_Characteristics.tif`
- `Definitive_Layers/6_Runout_Zone_Characteristics_stats.csv`
- `Definitive_Layers/6_Runout_Zone_Characteristics_legend.csv`

### 14. Ponderador ATES

Executa el ponderador per basin i despres fusiona el resultat global. El mode per defecte es `hybrid`.

Entrades per basin:

- DEM preprocessat.
- Forest alineat.
- `Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`.
- `Flow-Py/pra_basin_X/res_*/FP_travel_angle.tif`.
- `Watershed_Subdivisions/pra_basin_X.tif`.

Sortides:

- `Definitive_Layers/BasinX/Ponderador_ATES.tif`
- `Definitive_Layers/Ponderador_ATES.tif`
- `Definitive_Layers/Ponderador_ATES_smoothed.tif` si el smoothing final no sobreescriu

## Detall del ponderador

El nom historic "weighted ATES" pot apareixer en comentaris o identificadors, pero el codi actual no fa una mitjana ponderada final de classes ATES. El flux real es:

1. Binaritza el PRA de basin a `sz_binary_for_ponderador.tif`.
2. Calcula pendent amb GDAL (`slope.tif`).
3. Suavitza pendent amb finestra `WIN_SIZE = 3` per decidir classe 4.
4. Reclassifica pendent:
   - `0..15`: 0
   - `15..18`: 1
   - `18..26`: 2
   - `>26`: 3
   - pendent suavitzat `>39`: 4
5. Reclassifica `FP_travel_angle.tif`:
   - zona activa `0..90`: 1
   - `>=24`: 2
   - `>=33`: 3
6. Reclassifica `Exposure_zdelta_cellcount.tif`:
   - `0..5`: 1
   - `5..36`: 2
   - `>36`: 3
7. Combina pendent, Flow-Py i exposicio amb maxim cel.la a cel.la (`merge_new.tif`).
8. Reclassifica forest segons `forest_type`.
9. Codifica PRA com `0` fora i `100` dins.
10. Suma evidencia + forest + PRA.
11. Aplica una taula de mapping a classes ATES 0..4.
12. Escriu `merge_all.tif`.
13. Elimina clumps petits segons `ISL_SIZE = 30000`.
14. Fa `fillnodata`.
15. Escriu `ates_gen.tif`.
16. Aplica postproces classe 4 -> 3 en mode `hybrid`.
17. Copia a `Ponderador_ATES.tif`.

Llindars forest:

| Tipus | TREE1 | TREE2 | TREE3 |
| --- | ---: | ---: | ---: |
| `pcc` | 10 | 50 | 65 |
| `bav` | 10 | 20 | 25 |
| `stems` | 100 | 250 | 500 |
| `sen2cc` | 20 | 60 | 85 |

### Postproces classe 4 -> 3

Actiu per defecte en mode `hybrid`. Analitza clasters de classe 4 i els pot degradar a classe 3 segons:

- solapament amb zones d'inici d'una allau;
- landforms de propagacio de la mateixa allau;
- percentatge de landforms "safe" (`7,8,9` per defecte);
- percentatge de landforms "unsafe" (`1,2,3` per defecte);
- presencia d'entropia alta.

Sortides d'auditoria:

- `class4_runout_reclassification.csv`
- `class4_runout_reclassification_summary.csv`
- `class4_clusters_id.tif`
- `class4_reclass_decision.tif`
- `class4_reclass_decision_legend.txt`

### Fusio global

La fusio global fa maxim de classe entre outputs per basin. Si una cel.la ha estat degradada per `class4_reclass_decision.tif` amb codi `6`, el raster global preserva aquesta degradacio a classe 3.

Despres es pot aplicar el filtre direccional 2 -> 3:

- mira 8 direccions;
- per defecte raig de 25 cel.les;
- promou 2 -> 3 si veu classe 4 en almenys 6 direccions.

Finalment aplica smoothing:

- default: `modal`;
- elimina illes petites de classe amb `--ponderador-class-island-min-size 15`;
- si no es passa `--ponderador-smoothing-overwrite`, pot escriure `Ponderador_ATES_smoothed.tif`.

## Parametres principals

### Control general

- `--dem` default `inputs/DEM_BOW_SUMMIT.tif`
- `--forest` default `inputs/FOREST_BOW_SUMMIT.tif`
- `--forest-crs` default `None`
- `--forest-type` default `bav`; opcions `stems`, `bav`, `pcc`, `sen2cc`, `no_forest`
- `--outputs-dir` default `None`
- `--only-step6`
- `--until-n` valors `1..14`
- `--quiet`

### PRA

- `--radius` default `2`
- `--prob` default `0.5`
- `--winddir` default `0`
- `--windtol` default `180`
- `--pra-thd` default `0.15`
- `--sf` default `3`

### PRA Divisor

- `--divisor-stream-threshold` default `100`
- `--divisor-channel-init-exponent` default `1`
- `--divisor-channel-min-slope` default `0.005`

### Watershed GRASS

- `--watershed-threshold` default `12000`
- `--watershed-memory` default `500`
- `--grass-exe` default `C:\Program Files\QGIS 3.40.13\bin\grass84.bat`
- `--grass-epsg` default `None`
- `--grass-db` default `grassdata`
- `--grass-location` default `watershed_project`
- `--grass-mapset` default `NOUDIRECTORIDEMAPES`

### Flow-Py i overhead exposure

- `--flowpy-dir` default `Flow-py_Autoates_Editat/FlowPy_detrainment`
- `--flowpy-alpha` default `24`
- `--flowpy-exponent` default `8`
- `--flowpy-flux` default `0.003`
- `--flowpy-max-z` default `270`
- `--overhead-cellcount-weight` default `2`
- `--flowpy-infra` default `None`
- `--flowpy-forest-divisor` default `None`
- `--flowpy-no-forest`

`--overhead-cellcount-weight` accepta `0..1` o `2`. El valor `2` activa mode maxim.

### Slope + forest

- `--ates-forest-window` default `5`
- `--ates-slope-sigma` default `1.0`
- `--ates-forest-adjustment` default `paper_pra`; opcions `legacy`, `conservative`, `paper_pra`, `paper_runout`

### Landforms

- `--landform-windows` default `5,6,7,...,30`
- `--landform-curvature-threshold` default `1e-4`
- `--landform-flat-gradient-eps` default `1e-10`

### Terrain traps

- `--terrain-forest-tree-threshold` default `35.0`
- `--terrain-energy-trauma-threshold` default `0.35`
- `--terrain-gully-energy-threshold` default `0.22`
- `--terrain-gully-spi-m` default `1.0`
- `--terrain-gully-spi-n` default `1.0`
- `--terrain-gully-spi-threshold` default `0.0`
- `--terrain-gully-spi-percentile` default `88.0`
- `--terrain-gully-min-drainage-area-m2` default `4000.0`
- `--terrain-gully-min-slope-deg` default `13.0`
- `--terrain-gully-max-slope-deg` default `48.0`
- `--terrain-lake-max-slope-deg` default `6.0`
- `--terrain-lake-tpi-threshold` default `-1.8`
- `--terrain-lake-max-spi-threshold` default `0.0`
- `--terrain-lake-max-spi-percentile` default `35.0`

### Zones d'allau

- `--zones-start-threshold` default `0.99`
- `--zones-ending-threshold` default `0.19`

### Runout characteristics

- `--runout-flux-min-threshold` default `0.01`
- `--runout-min-evidence-threshold` default `0.03`

### Ponderador

- `--ponderador-forest-type` default `None`; hereta `--forest-type`
- `--ponderador-output-name` default `Ponderador_ATES.tif`
- `--ponderador-mode` default `hybrid`; opcions `hybrid`, `original`
- `--ponderador-class4-disable-reclass`
- `--ponderador-class4-landform-window` default `10`
- `--ponderador-class4-safe-classes` default `7,8,9`
- `--ponderador-class4-unsafe-classes` default `1,2,3`
- `--ponderador-class4-safe-pct-threshold` default `80.0`
- `--ponderador-class4-unsafe-pct-keep-threshold` default `15.0`
- `--ponderador-class4-entropy-pct-keep-threshold` default `5.0`
- `--ponderador-class4-entropy-max-for-downgrade` default `1.0`
- `--ponderador-class4-entropy-threshold` default `0.50`
- `--ponderador-class4-entropy-min-cluster-cells` default `25`
- `--ponderador-dir2to3-disable`
- `--ponderador-dir2to3-ray-lengths` default `25`
- `--ponderador-dir2to3-min-directions` default `6`
- `--ponderador-smoothing` default `modal`; opcions `none`, `modal`, `morph`, `vectorize`
- `--ponderador-smoothing-radius` default `1`
- `--ponderador-smoothing-iterations` default `1`
- `--ponderador-smoothing-vectorize-tolerance` default `0.0`
- `--ponderador-smoothing-chaikin-iterations` default `2`
- `--ponderador-smoothing-overwrite`
- `--ponderador-class-island-min-size` default `15`

## Dependencies i entorn

Python:

- `numpy`
- `rasterio`
- `scipy`
- `scikit-image`
- `whitebox`
- `osgeo/gdal`

Extern:

- QGIS amb GRASS, per defecte a `C:\Program Files\QGIS 3.40.13\bin\grass84.bat`.

## Notes de manteniment

- El pipeline esta pensat per revisar resultats pas a pas.
- `--only-step6` requereix outputs previs de preprocessat i watershed.
- Si `--outputs-dir` no es passa amb `--only-step6`, usa l'ultim `outputs/results_*`.
- El ponderador necessita forest; no accepta `--forest-type no_forest` al pas 14.
- Per canvis del ponderador, revisar tambe `Ponderador/PONDERADOR_CHANGES.md`.
