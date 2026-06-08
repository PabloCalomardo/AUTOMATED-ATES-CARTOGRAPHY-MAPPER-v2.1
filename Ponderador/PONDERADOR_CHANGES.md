# Ponderador Change Log

Aquest document registra l'evolucio del classificador del directori `Ponderador`.
Objectiu: poder fer canvis estructurals sense perdre tracabilitat i poder comparar resultats abans/despres.

## Scope

- Modul principal: `Ponderador/AutoATES_classifier.py`
- Integracio: `main.py` (step 14 invoca `run_autoates_weighted`)

## Baseline inicial

La versio de partida del ponderador funcionava aixi:

1. Entrades principals:
- DEM
- Canopy/forest
- Cell count / exposure (`Exposure_zdelta_cellcount.tif` a la pipeline)
- Flow-Py travel angle (`FP_travel_angle.tif`)
- PRA binari (`pra_basin_*.tif`)

2. Reclassificacions i combinacio:
- Reclassificacio de pendent (SAT01/SAT12/SAT23/SAT34)
- Reclassificacio de Flow-Py alpha angles (AAT1/AAT2/AAT3)
- Reclassificacio de cell count / exposicio (CC1/CC2)
- Reclassificacio de forest segons tipus (`pcc`, `bav`, `stems`, `sen2cc`)
- Aplicacio de mascara PRA
- Taula de mapping a classes finals ATES (0..4)

3. Generalitzacio:
- Eliminacio de clumps petits (< ISL_SIZE)
- `fillnodata` per suavitzar discontinuits
- Sortida principal: `ates_gen.tif` i copia a `Ponderador_ATES.tif`

4. Limitacio detectada a verificacio:
- Biaix conservador (underprediction), especialment:
  - classe 3 predita com 2
  - classe 4 predita com 3
- Errors mes alts en zones de frenada (`Star_propagating_Ending_Zones`, valor 3)

---

## Canvis provats el 2026-04-22

### Canvi 1: Post-ajust en zones de frenada (`ending-zone boost`)

Objectiu:
- Reduir underprediction en classes 3/4 a zones on l'analisi havia mostrat mes error.

Implementacio provada:
- Despres de generar `ates_gen.tif`, es construia la unio de zones de frenada llegint `BasinX/Star_propagating_Ending_Zones/Ava_*.tif` (valor 3 = ending).
- Es calculava un llindar d'exposicio per quantil sobre `cell_count_path` (q=0.60).
- Regla aplicada:
  - domini: pixels ending
  - condicio: exposicio >= quantil
  - promocio: 2 -> 3, 3 -> 4
- Es guardava `BasinX/ending_zone_boost_report.json`.

Funcions provades:
- `_ending_zone_union_mask(...)`
- `_apply_ending_zone_boost(...)`

### Canvi 2: Ajust de llindars per reduir biaix conservador

Llindars modificats en aquella prova:
- `SAT23`: 28 -> 27
- `SAT34`: 39 -> 38
- `AAT2`: 24 -> 23
- `AAT3`: 33 -> 32

Nota:
- Aquests canvis eren heuristics per validar tendencia.

---

## Rollback d'estrategia parcial

Per decisio de validacio, es va revertir l'estrategia d'`ending-zone boost` i part dels ajustos agressius.

Continua eliminat del flux actiu:
- post-ajust de zones de frenada (`ending-zone boost`)
- funcions auxiliars associades
- `ending_zone_boost_report.json`

Motivacio:
- Els resultats d'EXP2 mostraven reduccio d'underprediction pero increment important d'overprediction, especialment a Bow Summit.
- L'estrategia final activa evita el boost directe a ending zones i prioritza degradar clasters de classe 4 quan el context de propagacio/runout ho justifica.

---

## Estat actual actiu (2026-06-08)

### Llindars i constants

- `SAT01 = 15`
- `SAT12 = 18`
- `SAT23 = 26`
- `SAT34 = 39`
- `AAT1 = 18`
- `AAT2 = 24`
- `AAT3 = 33`
- `CC1 = 5`
- `CC2 = 36`
- `ISL_SIZE = 30000`
- `WIN_SIZE = 3`

Llindars de bosc per tipus:
- `pcc`: `TREE1 = 10`, `TREE2 = 50`, `TREE3 = 65`
- `bav`: `TREE1 = 10`, `TREE2 = 20`, `TREE3 = 25`
- `stems`: `TREE1 = 100`, `TREE2 = 250`, `TREE3 = 500`
- `sen2cc`: `TREE1 = 20`, `TREE2 = 60`, `TREE3 = 85`

### Entrades per basin

- DEM preprocessat.
- Forest alineat.
- `Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`.
- `Flow-Py/pra_basin_X/res_*/FP_travel_angle.tif`.
- `Watershed_Subdivisions/pra_basin_X.tif`.

### Flux del ponderador

1. Binaritza el PRA de basin a `sz_binary_for_ponderador.tif`.
2. Calcula pendent amb GDAL i reclassifica a classes 0..4.
3. Reclassifica `FP_travel_angle.tif` a classes 1..3 segons `AAT1/AAT2/AAT3`.
4. Reclassifica `Exposure_zdelta_cellcount.tif` a classes 1..3 segons `CC1/CC2`.
5. Combina pendent, Flow-Py i exposicio per maxim cel.la a cel.la (`merge_new.tif`).
6. Reclassifica bosc segons `forest_type` (`bav`, `pcc`, `stems`, `sen2cc`).
7. Aplica PRA com a mascara codificada (`0` fora, `100` dins).
8. Suma evidencia + bosc + PRA i aplica la taula final a classes ATES 0..4 (`merge_all.tif`).
9. Elimina clumps petits, fa `fillnodata` i escriu `ates_gen.tif`.
10. En mode `hybrid`, aplica per defecte el postproces classe 4 -> 3 basat en runout, landforms i entropia.
11. Copia `ates_gen.tif` a `Ponderador_ATES.tif`.
12. `main.py` fusiona els outputs per basin, aplica el filtre direccional 2 -> 3 si esta activat, i despres el smoothing final.

### Postproces actiu classe 4 -> 3

`run_autoates_weighted` executa per defecte una reclassificacio de classe 4 -> 3 diferent de l'antic `ending-zone boost`.

Usa:
- `BasinX/Star_propagating_Ending_Zones/Ava_*.tif`
- `2_Landforms/2_Landforms_curvature_10x10.tif` per defecte
- `2_Landforms_entropy_5to30.tif` o variants clustered si existeixen

Sortides d'auditoria per basin:
- `class4_runout_reclassification.csv`
- `class4_runout_reclassification_summary.csv`
- `class4_clusters_id.tif`
- `class4_reclass_decision.tif`
- `class4_reclass_decision_legend.txt`

### Fusio global i smoothing

- La fusio global combina rasters per basin amb maxim de classe.
- Si una cel.la consta com degradada a classe 3 al raster `class4_reclass_decision.tif` (codi 6), el merge global preserva aquesta degradacio.
- Despres pot aplicar una promocio direccional global de classe 2 -> 3.
- Finalment `main.py` aplica el suavitzat configurat amb `--ponderador-smoothing` (default: `modal`) i elimina illes petites de classe (default: `15` cel.les).

### Parametres de `main.py` associats al ponderador

- `--ponderador-forest-type` (default: hereta `--forest-type`)
- `--ponderador-output-name` (default: `Ponderador_ATES.tif`)
- `--ponderador-mode` (default: `hybrid`; opcions: `hybrid`, `original`)
- `--ponderador-class4-disable-reclass`
- `--ponderador-class4-landform-window` (default: `10`)
- `--ponderador-class4-safe-classes` (default: `7,8,9`)
- `--ponderador-class4-unsafe-classes` (default: `1,2,3`)
- `--ponderador-class4-safe-pct-threshold` (default: `80.0`)
- `--ponderador-class4-unsafe-pct-keep-threshold` (default: `15.0`)
- `--ponderador-class4-entropy-pct-keep-threshold` (default: `5.0`)
- `--ponderador-class4-entropy-max-for-downgrade` (default: `1.0`)
- `--ponderador-class4-entropy-threshold` (default: `0.50`)
- `--ponderador-class4-entropy-min-cluster-cells` (default: `25`)
- `--ponderador-dir2to3-disable`
- `--ponderador-dir2to3-ray-lengths` (default: `25`)
- `--ponderador-dir2to3-min-directions` (default: `6`)
- `--ponderador-smoothing` (default: `modal`)
- `--ponderador-smoothing-radius` (default: `1`)
- `--ponderador-smoothing-iterations` (default: `1`)
- `--ponderador-smoothing-vectorize-tolerance` (default: `0.0`)
- `--ponderador-smoothing-chaikin-iterations` (default: `2`)
- `--ponderador-smoothing-overwrite`
- `--ponderador-class-island-min-size` (default: `15`)

### Notes

- El nom "weighted ATES" es conserva en alguns identificadors antics, pero el flux actual no fa una mitjana ponderada final de classes ATES. Reclassifica evidencies, combina algunes capes per maxim, aplica bosc/PRA amb una taula de mapping i despres executa postprocessos espacials.
- Aquest document mante l'historial complet: els canvis provats no s'esborren encara que el codi actiu hagi canviat.
