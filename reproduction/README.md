# Reproduction scripts

This folder contains the training, evaluation and figure scripts behind the published models in this repository. It also holds the frozen job manifests and station lists that define every experiment.

**No data are included.** The archived public forecasts, the station observations used to compute the ETo targets, the eddy-covariance records and the crop-coefficient table are restricted (see *Data* below). The scripts document exactly how each result was produced. They run once the restricted inputs are placed at the paths listed below.

The released inference package (`../lft_et`, `../model`) does not depend on anything in this folder.

## Layout

The directory tree mirrors the original project root, so relative paths inside the scripts resolve unchanged.

| Path | Content |
|---|---|
| `*.py` (root) | Shared helpers: database access and degree-minute coordinate conversion (`db_io.py`), FAO-56 radiation terms (`fao56.py`), the `refet` loader (`compute_et0.py`), and weather-text parsing (`forecast_domain.py`). |
| `scripts/panel_dl.py`, `scripts/validation_protocol.py`, `scripts/run_pmt_m3_baseline.py` | Base network classes, encoders and validation helpers, imported by the trainers. |
| `scripts/audit_v11/` | Fold construction (`research_fold_data.py`: rolling roles, train-only vocabularies and scaling), model registry (`research_models.py`), training loop, and ETo target/QC builder (`build_qc_reference.py`). |
| `scripts/awm_repro/` | Experiment code for the paper: input shards, fast trainer, job specs, queue, summaries, statistics, transfer, eddy-covariance validation, ETc product, release export/verification. |
| `figures/scripts/` | Scripts for Figs. 1–12 and S1–S5 (Times New Roman; basemap shapefiles not included). |
| `reports/awm_repro_20261015/**/{*jobs*.json, manifest.json}` | Frozen job lists (model, fold, seed, configuration, station lists, initialisation) and shard manifests (with SHA-256 hashes). |
| `reports/awm_repro_20261015/group/lists/` | Station lists for the transfer, scaling and adaptation-budget experiments. |
| `reports/awm_repro_20261015/station_rice_zone.csv` | Station → rice zone (R1–R6) and rice-station flag. |
| `results/integrity/clean_smz_stations.txt` | Station list read by `scripts/panel_dl.py` at import. |

## Pipeline

All commands are run from this folder (`reproduction/`) with Python 3.13. The original runs used numpy 2.4, pandas 2, torch 2.10, scikit-learn 1.8, scipy 1.17, pyarrow 23, refet 0.4.2, openpyxl 3.1, matplotlib 3.10 and geopandas 1.1 (figures only).

1. **ETo targets and QC masks.** `python3 scripts/audit_v11/build_qc_reference.py`
   - Method: ASCE standardised short reference (`refet.Daily(..., method='asce')`), with 10-m wind (`uz=wind10, zw=10`).
   - Ea is computed from mean relative humidity.
   - Rs comes from sunshine duration with a = 0.25 and b = 0.50.
2. **Input shards.**
   - `build_aligned_shards.py`: the current issue. Lead 1 is the issue day, so the valid date is origin + lead − 1.
   - `build_lagged_shards.py`: adds up to six earlier issues of the same valid date, with ΔT revisions and availability flags.
   - `build_nolag_shards.py`: the control with every earlier issue masked.
   - `build_phys_shards.py`: deterministic physics covariates for the transfer analysis.
3. **Hyperparameter search and confirmation.**
   - `hpsearch_spec.py` builds the search jobs: 195-station subset, 2020 fold, selection by 2019 validation RMSE only.
   - `confirm_spec.py` re-runs the top configurations on all stations.
   - Jobs run through `bakeoff_queue.py <jobs.json> <parallel> <threads>`, which calls `pft_fast.py`.
4. **Formal comparison.**
   - `formal_spec.py` builds the frozen configurations for test years 2018, 2019 and 2020 with seeds 0–2.
   - `formal_summary.py`, `multi_metrics.py` and `paired_block_bootstrap.py` produce the paired 2° spatial-block bootstrap.
   - `national_baselines.py`: Hargreaves–Samani recalibration and gradient-boosted trees.
5. **Earlier-issue controls and statistics.**
   - `ablation/jobs.json`: LFT-current and GRU-lag.
   - `revision/revision_jobs.json`: pooling, issue-age and revision ablations; rice-only training; adaptation budget; quantile seeds.
   - `revision_stats.py`: interaction test, TOST, alternative resampling units, Holm adjustment.
6. **Information diagnostic.**
   - `national_info_ceiling.py` and `national_info_ceiling_order2.py`: single-order ladders.
   - `national_info_ceiling_shapley.py`: all 16 subsets, order-averaged shares.
7. **Probabilistic forecasts.** `revision_prob.py` computes joint lead × season × zone coverage and coverage on high-demand days.
8. **Transfer.**
   - `group_spec.py` and `group_opt_spec.py`: national, zone-only, leave-zone-out zero-shot and fine-tuning runs.
   - `group_summary.py`, `group_multiseed.py`, `transfer_physics_multiseed.py` and `revision_misc.py` summarise them.
   - `audit_group_leakage.py` checks that zero-shot evaluation stations never enter a training chain.
9. **ETc product.** `kc_etc_product.py` multiplies the forecast by the station-month Kc.
10. **Eddy-covariance validation.**
    - `ec_validation.py`: pairing, QC, Bowen-ratio closure.
    - `ec_actual_forecast.py` and `ec_extended.py`: forward Kc,EC calibration and skill against climatology.
    - `ec_supplement.py` and `revision_ec.py`: day accounting, Kc,EC uncertainty, closure intervals, pairing sensitivity.
11. **Release.**
    - `export_release.py` exports the 2020-fold LFT, GRU and quantile models.
    - `verify_release.py` checks the release package against the pipeline predictions.
12. **Figures.** `python3 figures/scripts/figXX_*.py`.

### Folds

| Test year | Training targets | Early stopping | Role |
|---|---|---|---|
| 2020 | Mar 2015 – Dec 2018 | 2019 | confirmatory; hyperparameters selected on this fold's 2019 validation only |
| 2019 | 2015 – 2017 | 2018 | retrospective |
| 2018 | 2015 – 2016 | 2017 | retrospective |

Vocabularies and input/target scaling are fitted on the training role only (`research_fold_data.FoldData`). The test year is read only after the checkpoint is frozen.

## Data (not distributed)

| Expected path | Content | Access |
|---|---|---|
| `data/db/rice_rl_data.db` (SQLite, opened read-only) | Tables `forecast_raw_daily` (archived city public forecasts, JSON per lead), `city_station_mapping`, `historical_weather_daily` (CMA station observations) | China Meteorological Administration / weather.com.cn terms; available from the corresponding author on reasonable request |
| `external/chinaflux/` | ChinaFLUX daily/half-hourly flux and meteorology files for Jurong, Changling and Panjin (original dataset folder and file names) | ChinaFLUX data service |
| `external/kc/kc_all_stations_monthly.csv`, `kc_site_matching.csv`, `rice_zone_stations.csv` | Station-by-month rice Kc table and rice-region station screening | Station-month Kc values are given in Supplementary Data S1 of the paper |
| `figures/basemap/` | Province boundary shapefiles | Standard map service of China |

Some strings in the code are in Chinese. These are file and column names of the original datasets, weather keywords of the Chinese public forecast (for example 晴 "sunny" and 小雨 "light rain") or log messages of imported helper modules. They are kept so that the scripts read the original data unchanged.

## Licence

MIT, as for the rest of the repository.
