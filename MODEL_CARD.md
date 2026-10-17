# Model card: lft-rice-et v1.0.0

## Intended use
Seven-day forecasts of reference evapotranspiration (ETo) and standard-condition rice evapotranspiration (ETc) in China from public city weather forecasts, for irrigation planning, crop-water-demand monitoring and research. The forecasts describe atmospheric evaporative demand and the requirement of a well-watered crop; they are not irrigation amounts and not actual paddy evapotranspiration.

## Inputs
For an issue date O: the message of issue O (daily maximum and minimum temperature, weather phrase and wind-force text for the issue day and the next six days) and, optionally, the messages of the six previous issues; station latitude, longitude, altitude and climate-zone token. Lead 1 is the issue day. Messages are archived once per day without the hour of issue, so lead 1 is a same-day forecast.

## Models
| Family | Inputs | Seeds | Parameters | Role |
|---|---|---|---|---|
| LFT (lag-fusion Transformer) | current + 6 earlier issues | 3 | 1.35 M | ensemble member |
| GRU | current issue | 3 | 4.0 M | ensemble member |
| Quantile LFT (light) | current + 6 earlier issues | 1 | 0.22 M | quantiles 5, 10, 25, 50, 75, 90, 95% |

Point forecast: mean of the three LFT seeds and of the three GRU seeds, averaged with equal weight and truncated at zero. Hyperparameters are listed in `model/*_config.json`.

## Training data
Public forecasts archived from weather.com.cn and ETo calculated with the standardized ASCE Penman–Monteith equation (short reference) from daily observations of 583 national meteorological stations (Tmax, Tmin, relative humidity, 10-m wind adjusted to 2 m, sunshine duration). Training: sequences with valid dates March 2015 to December 2018; early stopping on 2019. Input scaling and text vocabularies were fitted on the training years only; unseen weather or wind phrases map to an unknown token.

## Evaluation (out of sample)
| Test year | ETo RMSE (mm d⁻¹) | MAE | NSE | KGE | 7-day cumulative error |
|---|---|---|---|---|---|
| 2020 (confirmatory) | 0.744 | 0.537 | 0.81 | 0.85 | 16.6% |
| 2019 (retrospective) | 0.749 | 0.540 | 0.80 | 0.86 | 16.0% |
| 2018 (short training history) | 0.814 | 0.576 | 0.78 | 0.88 | 19.0% |

* By lead (2020): 0.676, 0.673, 0.703, 0.740, 0.772, 0.802, 0.829 mm d⁻¹ for leads 1–7; more accurate than station day-of-year climatology at all leads.
* By rice zone (2020): R1 South China 0.79, R2 Middle–lower Yangtze 0.77, R3 Southwest 0.69, R4 North China 0.76, R5 Northeast 0.70, R6 Northwest 0.73 mm d⁻¹.
* Quantiles (2020): coverage of the 90%, 80% and 50% intervals 0.895, 0.795 and 0.506; on the 10% highest-ETo days the 90% intervals cover 0.85 and the median is biased low by 0.70 mm d⁻¹.
* Rice ETc (2020, 447 rice stations, rice seasons): RMSE 1.08 mm d⁻¹ (25% of mean ETc), seven-day cumulative error 16%.
* Measured paddy ET (three eddy-covariance sites, crop coefficients calibrated on earlier years at each site): RMSE 1.20 mm d⁻¹ on the issue day and 1.30 mm d⁻¹ at lead 7, skill against an ETo climatology scaled by the same coefficients 0.30 and 0.19.

## Limitations
* Accuracy is limited by the information in the messages: the absence of radiation and humidity information sets most of the remaining error. Errors are largest on the highest-demand days, which the point forecasts underestimate (by 0.84 mm d⁻¹ on average in 2020).
* Valid for the message format of weather.com.cn city forecasts. A change of phrasing increases the share of unknown text tokens (reported by `Forecaster.predict_arrays` as `unknown_text_share`).
* Stations outside the 583-station network can be forecast with their coordinates and climate zone: models trained without any station of the Southwest or Northwest rice zones were 39–41% less accurate there, whereas unseen stations within represented climates were forecast with a 3–5% larger error.
* ETc requires crop coefficients supplied by the user; the evaluation used monthly, climatologically adjusted coefficients for single-season middle rice and, in the south, late rice, and local calibration improved the agreement with measured paddy ET.
* The ensemble is frozen as of the 2020 fold; retraining with recent years is recommended before operational use.
