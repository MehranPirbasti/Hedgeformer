# data/

## splits/<scene>/

| File | Content |
|---|---|
| `train_grouped.csv`, `val_grouped.csv`, `test_grouped.csv` | the 512 m tiles of each split with their sub-patch rows (tile row/column, quadrant or centre, hedge and nodata flags). The 918 labelled tiles were assigned at random (70/15/15, fixed seed); the 46 label-free tiles are in the training set. |
| `band_statistics.csv` | per-scene, per-band mean and standard deviation used for standardisation (the two scenes have different radiometry, see the paper, Section III-A). |
| `clipped_lines.gpkg` | the UKCEH Land Cover Plus: Hedgerows 2016–2021 (England) centrelines clipped to the scene bounding box, with the height class attribute. British National Grid (EPSG:27700). 461.4 km (Scene 1) and 400.2 km (Scene 2). |

The `.npz` tiles that `main.py` reads (six Pléiades Neo bands, the height layer, the class rasters and proximity fields) are **not** distributed: they contain Pléiades Neo pixels, which are licensed from Airbus. A licence holder can rebuild them from the two scenes and the height layer with `dataprep.py`.

## Licences

- `clipped_lines.gpkg` is derived from the UKCEH Land Cover Plus: Hedgerows product, © UKCEH, available from the Environmental Information Data Centre under its licence. It is provided here to make the split and the reference length per tile checkable; any other use of these lines requires the UKCEH product licence.
- The height layer is the Environment Agency England 1 m Composite DTM/DSM (Google Earth Engine asset `UK/EA/ENGLAND_1M_TERRAIN/2022`), Open Government Licence v3.0. Environment Agency copyright and/or database right 2022.
- Pléiades Neo imagery © Airbus DS 2022 and 2024, all rights reserved; not redistributed.
