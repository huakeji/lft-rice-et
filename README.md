# lft-rice-et: seven-day reference and rice evapotranspiration forecasts from public weather forecasts

The package turns the public city forecasts that most farmers receive (daily maximum and minimum temperature, a short weather phrase and a wind-force text, as published by weather.com.cn) into forecasts of reference evapotranspiration (ETo, FAO-56/ASCE short reference) for the issue day and the following six days, and into standard-condition rice evapotranspiration (ETc = Kc × ETo). No station observation is needed at forecast time.

* **Models.** Lag-fusion Transformer (LFT, current issue plus the six earlier issues for the same dates) and a GRU on the current issue, three training seeds each; the operational forecast is the equal-weight LFT+GRU ensemble. A quantile LFT provides the 5–95% quantiles.
* **Training data.** 583 Chinese meteorological stations, forecasts and ETo targets with valid dates March 2015 to December 2018; 2019 for early stopping. The models were frozen before evaluation on the independent test year 2020.
* **Accuracy (2020, 582 stations, out of sample).** ETo RMSE 0.744 mm d⁻¹ (issue day 0.676, lead 7 0.829), NSE 0.81; 90% intervals cover 89.5% of observations; standard-condition rice ETc RMSE 1.08 mm d⁻¹ and 16% error on seven-day totals. See [MODEL_CARD.md](MODEL_CARD.md).

## Installation

```bash
pip install -e .                       # numpy, pandas, torch (CPU is sufficient); run from this directory
python -m unittest discover -s tests   # 5 tests, about 1 s
```

## Usage

Command line, with the JSON input format described in `lft_et/cli.py` and illustrated by the synthetic `examples/example_messages.json`:

```bash
python -m lft_et.cli --input examples/example_messages.json --output forecast.csv
```

Python:

```python
from datetime import date
from lft_et import Forecaster, Message, load_stations

fc = Forecaster(kc_path="kc.csv")       # LFT + GRU ensembles, quantile LFT; kc_path optional
station = load_stations()["57494"]      # or Station("id", lat, lon, alt, zone) for another site
messages = {date(2020, 7, 15): Message(weather, high, low, wind), ...}   # issue date and up to six earlier issues
table = fc.forecast(station, date(2020, 7, 15), messages)
```

Each `Message` holds four dicts keyed `"1"`…`"15"` (key `"1"` is the issue day). Earlier issues are optional; missing ones are flagged and the forecast falls back on the current issue. Output columns: `eto` (ensemble), `eto_lft`, `eto_gru` and `eto_q05`…`eto_q95` in mm d⁻¹. With a crop-coefficient file (`--kc kc.csv`, columns `station, month, Kc, rice_season_month`), the output adds `kc`, `rice_season` and `etc` = Kc × ETo. 

For a site outside the station list, give its latitude, longitude, altitude and the climate-zone token of the nearest station in `data/stations.csv` (`CC`, `SC`, `NC`, `NEC`, `NWC`, `IM`, `QTP`).

## Contents

| Path | Content |
|---|---|
| `lft_et/features.py` | Parsing of public forecast messages into model inputs (identical to the original training pipeline) |
| `lft_et/models.py` | LFT and GRU definitions, input encoder |
| `lft_et/predict.py`, `lft_et/cli.py` | Ensemble forecasting, ETc conversion, command line |
| `model/` | Encoders (JSON), weights of LFT (3 seeds), GRU (3 seeds) and quantile LFT, configurations, export checks |
| `data/stations.csv` | Station coordinates, climate-zone token, rice zone and rice-station flag |
| `VERIFICATION.json` | End-to-end check: inputs rebuilt from raw messages and forecasts equal those of the original training pipeline (30 stations, 10,574 sequences, max. difference 2×10⁻⁶ mm d⁻¹) |

Out-of-sample forecasts of 2018–2020 are available from the authors on request.

Training, evaluation and figure scripts, with the frozen job manifests and station lists, are in [`reproduction/`](reproduction/README.md) (code only; the input data are restricted).

## Notes

* Weather and wind texts are kept in Chinese as published (e.g. 多云转小雨, 3-4级): they are model inputs, matched against the vocabularies in `model/*_encoder.json`. All other files are in English; `README_zh.md` is a Chinese version of this README.

* ETo is calculated reference evapotranspiration; ETc is the requirement of a well-watered rice crop. Neither is an irrigation amount, which also depends on rainfall, ponding and percolation.
* The forecasts reproduce the information content of the messages; skill decreases with lead. In the Southwest and Northwest rice zones, models trained without local stations transferred poorly; the released national models were trained with all zones.
* Public forecast messages are not redistributed with this package.

## Licence

MIT ([LICENSE](LICENSE)), for code and model weights. Citation information: [CITATION.cff](CITATION.cff). Public forecast messages and station observations are not redistributed; operational services that use public forecast messages may require permission from their provider.
