# Automated ATES Cartography Mapper

Pipeline Python per generar cartografia ATES automatitzada a partir d'un DEM i un raster de bosc. El repositori calcula zones potencials de sortida d'allaus, simula propagacio amb Flow-Py i combina diverses evidencies geomorfologiques per produir capes finals ATES.

Per entendre el funcionament intern amb detall, consulta [README_TECNIC.md](README_TECNIC.md).

## Que conte

- `main.py`: orquestrador del pipeline complet.
- `PREPROCESSING/`: retall, omplert de nodata i alineacio DEM/forest.
- `PRAs/`: calcul PRA, divisio de PRA i subdivisio per conques.
- `Flow-py_Autoates_Editat/`: motor Flow-Py modificat.
- `PostProcess_FlowPY/`: capes derivades de Flow-Py, landforms, terrain traps, zones d'allau i runout.
- `Ponderador/`: classificador final ATES per evidencies combinades.
- `inputs/`: dades d'exemple.

## Flux resumit

1. Valida DEM i forest.
2. Preprocessa i alinea els rasters.
3. Calcula PRA.
4. Divideix PRA per conques.
5. Executa Flow-Py per cada conca.
6. Genera capes derivades: exposicio, landforms, terrain traps, zones d'inici/propagacio/frenada i runout.
7. Executa el ponderador per conca.
8. Fusiona el resultat global ATES.

## Execucio rapida

Des de l'arrel del repositori:

```bash
python main.py
```

Amb inputs personalitzats:

```bash
python main.py --dem inputs/DEM_BOW_SUMMIT.tif --forest inputs/FOREST_BOW_SUMMIT.tif
```

Executar fins a un pas:

```bash
python main.py --until-n 8
```

Reexecutar nomes Flow-Py amb resultats previs:

```bash
python main.py --only-step6
```

## Sortides principals

Cada execucio completa crea una carpeta:

```text
outputs/results_DDHHMM/
```

Resultats clau:

- `PRA_AutoATES/PRA_binary.tif`
- `Watershed_Subdivisions/pra_basin_*.tif`
- `Flow-Py/pra_basin_*/res_*/*`
- `Definitive_Layers/BasinX/Exposure_zdelta_cellcount.tif`
- `Definitive_Layers/BasinX/Ponderador_ATES.tif`
- `Definitive_Layers/Ponderador_ATES.tif`
- `Definitive_Layers/Ponderador_ATES_smoothed.tif` si el suavitzat no sobreescriu la sortida global

## Requisits

- Python 3.x
- `numpy`, `rasterio`, `scipy`, `scikit-image`, `whitebox`
- QGIS amb GRASS per la subdivisio de conques
- GDAL/OSGeo disponible per al calcul de pendent del ponderador

## Documentacio tecnica

El README curt nomes dona orientacio general. El detall de passos, parametres, capes d'entrada/sortida i logica del ponderador esta a:

[README_TECNIC.md](README_TECNIC.md)
