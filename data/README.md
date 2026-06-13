# Bundled SMARD.de Sample

`smard_de_load_sample_quarterhour.csv` is a compact offline sample downloaded from the public
SMARD.de chart-data endpoint:

```text
https://www.smard.de/app/chart_data/410/DE/410_DE_quarterhour_1759701600000.json
```

The file keeps the first 96 quarter-hour points from the downloaded block. The raw
`smard_quarterhour_value` is converted to `load_mw` with:

```text
load_mw = smard_quarterhour_value * 4
```

SMARD values are German aggregate system values, while the app's IEEE pandapower networks are
small test grids. The application therefore uses this data as a relative load profile multiplier
instead of replacing pandapower loads with Germany-wide absolute MW values.
