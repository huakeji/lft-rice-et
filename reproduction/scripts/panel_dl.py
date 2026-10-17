"""Clean-SMZ 71-station panel deep-learning ablation.

Formal run for:
- S_single: station-local history model, no physical forecast features.
- U_pooled 2x2: history vs forecast-only crossed with no-phys vs M3-Rs/ea features.

Leakage guard:
- station list comes from results/integrity/clean_smz_stations.txt;
- alignment screening is computed before training and failed stations are dropped;
- weather/wind vocabularies, feature standardizers, anomaly scaling, M3
  coefficients, PET_BC AI/aT, and physical-feature fill values are train-only;
- validation is used only for early stopping and the optional global bias
  correction; test 2020 is evaluated after model selection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Data root; override (e.g. national_bigmodel sets pdl.DATA_DIR for the ERA5 panel).
DATA_DIR = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from db_io import ddmm_to_decimal
from fao56 import extraterrestrial_radiation, fao56_pm_daily_rs, saturation_vapor_pressure, wind10_to_wind2
from run_pmt_m3_baseline import calibrate_station, design_matrix
from validation_protocol import resolve_validation_stations, sequence_within_split


CLEAN_STATIONS_FILE = ROOT / "results" / "integrity" / "clean_smz_stations.txt"
WUHAN = "57494"
TRAIN_END = "2018-12-31"
TEST_START = "2020-01-01"
TEST_END = "2020-12-31"
HORIZON = 7
SEED_OFFSET = 20260611
OLD_MISALIGNED_S_SINGLE = 0.914
OLD_MISALIGNED_U_POOLED = 0.836

BASE_PAST_COLS = [
    "tmax_filled",
    "tmin_filled",
    "rh_filled",
    "wind10_filled",
    "rs_filled",
    "rain_filled",
    "et0_anomaly",
]
BASE_FUTURE_NUMERIC_COLS = [
    "tmax",
    "tmin",
    "dtr",
    "rh",
    "rain_signal",
    "sin_doy",
    "cos_doy",
    "longitude",
    "latitude",
    "altitude_m",
]
PHYS_FEATURE_COLS = ["rs_m3", "ea_phys"]
METRIC_COLS = ["rmse", "mae", "r2", "nse", "bias", "mean_et0", "nrmse"]
KEY_COLS = ["station", "origin_date", "target_date", "lead"]
PHYS_RAW_COLS = ["tmax", "tmin", "ra", "altitude_m"]
PHYS_OBS_COLS = ["rs_obs", "ea_obs", "u2_obs"]


def load_panel_stations(path: Path = CLEAN_STATIONS_FILE) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing clean SMZ station list: {path}")
    stations = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not stations:
        raise RuntimeError(f"No stations found in {path}")
    if len(stations) != len(set(stations)):
        duplicates = sorted({station for station in stations if stations.count(station) > 1})
        raise RuntimeError(f"Duplicate stations in {path}: {duplicates}")
    return stations


PANEL_STATIONS = load_panel_stations()


U_STRATEGIES: dict[str, dict[str, bool]] = {
    # Forecast-only cells first: low memory (no 90-day history tensor) and they are
    # the target deployable architecture, so they complete first under --jobs 1.
    "U_FO_nophys": {"forecast_only": True, "phys_feats": False},
    "U_FO_phys": {"forecast_only": True, "phys_feats": True},
    "U_hist_nophys": {"forecast_only": False, "phys_feats": False},
    "U_hist_phys": {"forecast_only": False, "phys_feats": True},
}
U_STRATEGY_ORDER = list(U_STRATEGIES)


def parse_csv_list(raw: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_torch_worker_threads(n: int) -> None:
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def station_meta(db_path: Path) -> dict[str, dict[str, Any]]:
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT station_code, province, station_name, longitude, latitude, altitude_m FROM stations",
        con,
    )
    con.close()
    df["station_code"] = df["station_code"].astype(str)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        lon = ddmm_to_decimal(row["longitude"])
        lat = ddmm_to_decimal(row["latitude"])
        alt = float(row["altitude_m"])
        out[row["station_code"]] = {
            "station_code": row["station_code"],
            "station": row["station_code"],
            "province": row.get("province"),
            "station_name": row.get("station_name"),
            "longitude": float(lon),
            "latitude": float(lat),
            "altitude_m": alt,
            "static": np.array([lon, lat, alt], dtype=np.float32),
        }
    return out


def clean_category(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str)


def future_numeric_cols(phys_feats: bool) -> list[str]:
    cols = list(BASE_FUTURE_NUMERIC_COLS)
    if phys_feats:
        cols.extend(PHYS_FEATURE_COLS)
    return cols


def prediction_csv_path(out_dir: Path, strategy: str, station: str, seed: int, split_name: str) -> Path:
    prefix = "val_predictions" if split_name == "val" else "predictions"
    return out_dir / f"{prefix}_{strategy}_{station}_seed{seed}.csv"


def numpy_saturation_vapor_pressure(temp_c: np.ndarray) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=float)
    out = np.full(temp.shape, np.nan, dtype=float)
    valid = np.isfinite(temp) & (temp > -237.3)
    out[np.isfinite(temp) & ~valid] = 0.0
    out[valid] = 0.6108 * np.exp(17.27 * temp[valid] / (temp[valid] + 237.3))
    return out


def torch_saturation_vapor_pressure(temp_c: torch.Tensor) -> torch.Tensor:
    raw = 0.6108 * torch.exp(17.27 * temp_c / (temp_c + 237.3))
    return torch.where(temp_c <= -237.3, torch.zeros_like(raw), raw)


def torch_fao56_pm_daily_rs(
    tmax: torch.Tensor,
    tmin: torch.Tensor,
    ra: torch.Tensor,
    altitude_m: torch.Tensor,
    rs: torch.Tensor,
    ea: torch.Tensor,
    u2: torch.Tensor,
) -> torch.Tensor:
    """Differentiable FAO-56 daily PM layer matching fao56.fao56_pm_daily_rs."""
    ea_tmax = torch_saturation_vapor_pressure(tmax)
    ea_tmin = torch_saturation_vapor_pressure(tmin)
    es = (ea_tmax + ea_tmin) / 2.0
    tmean = (tmax + tmin) / 2.0

    rns = 0.77 * rs
    rso = (0.75 + 2e-5 * altitude_m) * ra
    safe_ratio = torch.where(rso != 0, rs / rso, torch.zeros_like(rs))
    rnl = 4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0
    rnl = rnl * (0.34 - 0.14 * torch.sqrt(torch.clamp(ea, min=0.0))) * (1.35 * safe_ratio - 0.35)
    rn = rns - rnl

    pressure = 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26
    gamma = 0.000665 * pressure
    slope = 4098.0 * torch_saturation_vapor_pressure(tmean) / ((tmean + 237.3) ** 2)
    denominator = slope + gamma * (1.0 + 0.34 * u2)
    numerator = 0.408 * slope * rn + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    return torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))


def make_vocab(
    stations: list[str],
    data_dir: Path,
    station_clean_years: dict[str, set[int]] | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    weather_set: set[str] = set()
    wind_set: set[str] = set()
    for station in stations:
        fc = pd.read_csv(
            data_dir / "forecast_domain" / f"forecast_modeling_rows_{station}.csv",
            usecols=["origin_split", "origin_date", "weather_text", "wind_text"],
        )
        train = fc["origin_split"].astype(str) == "train"
        if station_clean_years is not None and station in station_clean_years:
            years = station_clean_years[station]
            origin_year = pd.to_datetime(fc["origin_date"], errors="coerce").dt.year
            train = train & origin_year.isin(years)
        weather_set.update(clean_category(fc.loc[train, "weather_text"]).unique().tolist())
        wind_set.update(clean_category(fc.loc[train, "wind_text"]).unique().tolist())
    weather_vocab = {token: i + 1 for i, token in enumerate(sorted(weather_set))}
    wind_vocab = {token: i + 1 for i, token in enumerate(sorted(wind_set))}
    return weather_vocab, wind_vocab


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def annual_daylight_hours(year: int, latitude: float) -> float:
    start = date(year, 1, 1)
    total = 0.0
    for offset in range(366 if is_leap_year(year) else 365):
        day = start + pd.Timedelta(days=offset)
        total += extraterrestrial_radiation(day, latitude)[1]
    return float(total)


def blaney_criddle_daily_pet(dates: pd.Series, latitude: float, tmean: pd.Series) -> np.ndarray:
    out = np.full(len(dates), np.nan, dtype=float)
    daylight_cache: dict[int, float] = {}
    for idx, (row_date, temp) in enumerate(zip(pd.to_datetime(dates), pd.to_numeric(tmean, errors="coerce"))):
        if not np.isfinite(temp):
            continue
        row_day = row_date.date()
        if row_day.year not in daylight_cache:
            daylight_cache[row_day.year] = annual_daylight_hours(row_day.year, latitude)
        _, daylight = extraterrestrial_radiation(row_day, latitude)
        p_day = 100.0 * daylight / daylight_cache[row_day.year] if daylight_cache[row_day.year] > 0 else np.nan
        value = p_day * (0.46 * float(temp) + 8.13)
        out[idx] = max(0.0, float(value)) if np.isfinite(value) else np.nan
    return out


def train_only_ea_params(history: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    train = history[history["split"].astype(str) == "train"].copy()
    if train.empty:
        raise RuntimeError(f"No train history rows for station {meta['station_code']}")
    assert train["date"].max().strftime("%Y-%m-%d") <= TRAIN_END

    for column in ["rain", "t_mean_filled", "tmax_filled", "tmin_filled"]:
        train[column] = pd.to_numeric(train[column], errors="coerce")
    train["year"] = train["date"].dt.year
    tmean = train["t_mean_filled"].copy()
    tmean = tmean.fillna((train["tmax_filled"] + train["tmin_filled"]) / 2.0)
    train["pet_bc"] = blaney_criddle_daily_pet(train["date"], float(meta["latitude"]), tmean)

    rain_year = train.dropna(subset=["rain"]).groupby("year")["rain"].agg(["sum", "count"])
    rain_year = rain_year.rename(columns={"sum": "rain_sum", "count": "rain_days"})
    pet_year = train.dropna(subset=["pet_bc"]).groupby("year")["pet_bc"].agg(["sum", "count"])
    pet_year = pet_year.rename(columns={"sum": "pet_bc_sum", "count": "pet_bc_days"})
    annual = rain_year.join(pet_year, how="inner")
    complete = annual[(annual["rain_days"] >= 330) & (annual["pet_bc_days"] >= 330) & (annual["pet_bc_sum"] > 0)].copy()
    if complete.empty:
        complete = annual[(annual["rain_sum"].notna()) & (annual["pet_bc_sum"] > 0)].copy()
    if complete.empty:
        raise RuntimeError(f"Cannot compute train-only AI/PET_BC for station {meta['station_code']}")

    annual_rain_mean = float(complete["rain_sum"].mean())
    annual_pet_bc_mean = float(complete["pet_bc_sum"].mean())
    ai = annual_rain_mean / annual_pet_bc_mean if annual_pet_bc_mean > 0 else float("nan")
    if not np.isfinite(ai) or ai <= 0:
        raise RuntimeError(f"Invalid train-only AI for station {meta['station_code']}: {ai}")
    at = -2.87 * math.log(ai) + 1.47
    return {
        "station_code": str(meta["station_code"]),
        "ai_source": "train annual precipitation / train annual PET_BC",
        "pet_bc_formula": "daily PET_BC=max(0, p_daylight_percent*(0.46*Tmean+8.13)); annual sum by calendar year",
        "train_end": TRAIN_END,
        "ai": float(ai),
        "aT": float(at),
        "annual_rain_mean_mm": annual_rain_mean,
        "annual_pet_bc_mean_mm": annual_pet_bc_mean,
        "annual_years_used": int(len(complete)),
        "annual_year_min": int(complete.index.min()),
        "annual_year_max": int(complete.index.max()),
    }


def qiu_ea_from_tmin(tmin: np.ndarray, at: float) -> np.ndarray:
    dew_proxy = np.asarray(tmin, dtype=float) - float(at)
    return np.array([saturation_vapor_pressure(value) for value in dew_proxy], dtype=float)


def compute_station_physical_payloads(
    station: str,
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models = calibrate_station(history, meta)
    m3 = models["M3"]
    ea_params = train_only_ea_params(history, meta)

    tmp = forecast.copy()
    tmp["target_date"] = pd.to_datetime(tmp["target_date"])
    for column in ["tmax_input", "tmin_input"]:
        tmp[column] = pd.to_numeric(tmp[column], errors="coerce")
    latitude = float(meta["latitude"])
    target_dates = [item.date() for item in tmp["target_date"]]
    ra = np.array([extraterrestrial_radiation(item, latitude)[0] for item in target_dates], dtype=float)
    tmax = tmp["tmax_input"].to_numpy(dtype=float)
    tmin = tmp["tmin_input"].to_numpy(dtype=float)
    tmean = (tmax + tmin) / 2.0
    valid_temp = np.isfinite(tmax) & np.isfinite(tmin) & (tmax >= tmin)

    rs_m3 = np.full(len(tmp), np.nan, dtype=float)
    if valid_temp.any():
        raw = design_matrix("M3", tmax[valid_temp], tmin[valid_temp], tmean[valid_temp], ra[valid_temp]) @ np.asarray(
            m3["coefficients_ordered"],
            dtype=float,
        )
        rs_m3[valid_temp] = np.clip(raw, 0.0, ra[valid_temp])
    ea_phys = qiu_ea_from_tmin(tmin, float(ea_params["aT"]))

    train_mask = tmp["origin_split"].astype(str) == "train"
    rs_train_mean = float(np.nanmean(rs_m3[train_mask])) if np.isfinite(rs_m3[train_mask]).any() else float("nan")
    if not np.isfinite(rs_train_mean):
        hist_train = history[history["split"].astype(str) == "train"].copy()
        rs_train_mean = float(pd.to_numeric(hist_train["rs"], errors="coerce").mean())
    if not np.isfinite(rs_train_mean):
        raise RuntimeError(f"Cannot compute train-only rs_m3 fill mean for station {station}")

    ea_train_mean = float(np.nanmean(ea_phys[train_mask])) if np.isfinite(ea_phys[train_mask]).any() else float("nan")
    if not np.isfinite(ea_train_mean):
        hist_train = history[history["split"].astype(str) == "train"].copy()
        tmin_train = pd.to_numeric(hist_train["tmin_filled"], errors="coerce").to_numpy(dtype=float)
        ea_train_mean = float(np.nanmean(qiu_ea_from_tmin(tmin_train, float(ea_params["aT"]))))
    if not np.isfinite(ea_train_mean):
        raise RuntimeError(f"Cannot compute train-only ea_phys fill mean for station {station}")

    m3_payload = {
        "station_code": str(station),
        "formula": m3["formula"],
        "columns": m3["columns"],
        "coefficients": m3["coefficients"],
        "coefficients_ordered": m3["coefficients_ordered"],
        "metrics": m3["metrics"],
        "calibration_rows": m3["calibration_rows"],
        "train_date_min": m3["train_date_min"],
        "train_date_max": m3["train_date_max"],
        "rs_clip": "[0, Ra]",
        "rs_m3_train_fill_mean": rs_train_mean,
    }
    ea_payload = {
        **ea_params,
        "ea_phys_train_fill_mean": ea_train_mean,
        "ea_formula": "ea=saturation_vapor_pressure(tmin_forecast-aT)",
    }
    return m3_payload, ea_payload


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_physical_caches(
    stations: list[str],
    data_dir: Path,
    meta: dict[str, dict[str, Any]],
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m3_path = out_dir / "m3_coeffs.json"
    ea_path = out_dir / "ea_params.json"
    print(f"[phys] calibrating M3 Rs and Qiu ea params for {len(stations)} stations", flush=True)
    m3_cache: dict[str, Any] = {}
    ea_cache: dict[str, Any] = {}
    started = time.time()
    for idx, station in enumerate(stations, start=1):
        history = pd.read_csv(data_dir / f"clean_history_{station}.csv")
        history["date"] = pd.to_datetime(history["date"])
        forecast = pd.read_csv(data_dir / "forecast_domain" / f"forecast_modeling_rows_{station}.csv")
        m3_payload, ea_payload = compute_station_physical_payloads(station, forecast, history, meta[station])
        m3_cache[station] = m3_payload
        ea_cache[station] = ea_payload
        print(
            f"[phys] {idx}/{len(stations)} station={station} "
            f"M3_R2={float(m3_payload['metrics']['r2']):.3f} aT={float(ea_payload['aT']):.3f}",
            flush=True,
        )
    write_json(m3_path, m3_cache)
    write_json(ea_path, ea_cache)
    print(f"[phys] wrote {m3_path} and {ea_path} elapsed={(time.time() - started):.1f}s", flush=True)
    return m3_cache, ea_cache


def load_physical_caches(phys_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    m3_path = phys_dir / "m3_coeffs.json"
    ea_path = phys_dir / "ea_params.json"
    if not m3_path.exists() or not ea_path.exists():
        raise FileNotFoundError(f"Missing physical cache files under {phys_dir}")
    return (
        json.loads(m3_path.read_text(encoding="utf-8")),
        json.loads(ea_path.read_text(encoding="utf-8")),
    )


def apply_physical_forecast_features(
    forecast: pd.DataFrame,
    meta: dict[str, Any],
    m3_payload: dict[str, Any],
    ea_payload: dict[str, Any],
) -> pd.DataFrame:
    out = forecast.copy()
    out["target_date"] = pd.to_datetime(out["target_date"])
    for column in ["tmax_input", "tmin_input"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    latitude = float(meta["latitude"])
    target_dates = [item.date() for item in out["target_date"]]
    ra = np.array([extraterrestrial_radiation(item, latitude)[0] for item in target_dates], dtype=float)
    tmax = out["tmax_input"].to_numpy(dtype=float)
    tmin = out["tmin_input"].to_numpy(dtype=float)
    tmean = (tmax + tmin) / 2.0
    valid_temp = np.isfinite(tmax) & np.isfinite(tmin) & (tmax >= tmin)

    rs_m3 = np.full(len(out), np.nan, dtype=float)
    if valid_temp.any():
        raw = design_matrix(
            "M3",
            tmax[valid_temp],
            tmin[valid_temp],
            tmean[valid_temp],
            ra[valid_temp],
        ) @ np.asarray(m3_payload["coefficients_ordered"], dtype=float)
        rs_m3[valid_temp] = np.clip(raw, 0.0, ra[valid_temp])
    ea_phys = qiu_ea_from_tmin(tmin, float(ea_payload["aT"]))

    rs_fill = float(m3_payload["rs_m3_train_fill_mean"])
    ea_fill = float(ea_payload["ea_phys_train_fill_mean"])
    out["rs_m3"] = np.where(np.isfinite(rs_m3), rs_m3, rs_fill)
    out["ea_phys"] = np.where(np.isfinite(ea_phys), ea_phys, ea_fill)
    return out


def observed_physical_drivers(history: pd.DataFrame) -> pd.DataFrame:
    """Observed Rs, ea, and 2 m wind by date for physical-head auxiliary loss."""
    obs = history[["date", "rs", "tmax", "tmin", "rh", "wind10"]].copy()
    obs["date"] = pd.to_datetime(obs["date"])
    for column in ["rs", "tmax", "tmin", "rh", "wind10"]:
        obs[column] = pd.to_numeric(obs[column], errors="coerce")

    tmax = obs["tmax"].to_numpy(dtype=float)
    tmin = obs["tmin"].to_numpy(dtype=float)
    rh = obs["rh"].to_numpy(dtype=float)
    wind10 = obs["wind10"].to_numpy(dtype=float)
    valid_temp = np.isfinite(tmax) & np.isfinite(tmin) & (tmax >= tmin)

    es_obs = np.full(len(obs), np.nan, dtype=float)
    es_obs[valid_temp] = (
        numpy_saturation_vapor_pressure(tmax[valid_temp])
        + numpy_saturation_vapor_pressure(tmin[valid_temp])
    ) / 2.0
    ea_obs = np.where(valid_temp & np.isfinite(rh), es_obs * rh / 100.0, np.nan)
    u2_obs = np.full(len(obs), np.nan, dtype=float)
    valid_wind = np.isfinite(wind10)
    if valid_wind.any():
        u2_obs[valid_wind] = np.array([wind10_to_wind2(value) for value in wind10[valid_wind]], dtype=float)

    out = pd.DataFrame(
        {
            "date": obs["date"],
            "rs_obs": obs["rs"].to_numpy(dtype=float),
            "ea_obs": ea_obs,
            "u2_obs": u2_obs,
        }
    )
    return out.set_index("date")


def run_alignment_check(
    stations: list[str],
    data_dir: Path,
    out_dir: Path,
    min_corr: float = 0.85,
    station_clean_years: dict[str, set[int]] | None = None,
    required_splits: tuple[str, ...] = ("train", "val", "test"),
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for station in stations:
        fc = pd.read_csv(
            data_dir / "forecast_domain" / f"forecast_modeling_rows_{station}.csv",
            usecols=["origin_split", "origin_date", "lead", "target_date", "input_source", "tmax_input"],
        )
        fc = fc[fc["lead"].astype(int) == 1].copy()
        fc = fc[fc["input_source"].astype(str) == "forecast"].copy()
        fc["origin_date"] = pd.to_datetime(fc["origin_date"])
        fc["target_date"] = pd.to_datetime(fc["target_date"])
        fc["tmax_input"] = pd.to_numeric(fc["tmax_input"], errors="coerce")
        hist = pd.read_csv(data_dir / f"clean_history_{station}.csv", usecols=["date", "tmax"])
        hist["target_date"] = pd.to_datetime(hist["date"])
        hist["tmax_obs"] = pd.to_numeric(hist["tmax"], errors="coerce")
        merged = fc.merge(hist[["target_date", "tmax_obs"]], on="target_date", how="left")
        station_row: dict[str, Any] = {"station": station}
        pass_station = True
        for split_name in ["train", "val", "test"]:
            part = merged[merged["origin_split"].astype(str) == split_name].dropna(subset=["tmax_input", "tmax_obs"])
            if split_name == "train" and station_clean_years is not None and station in station_clean_years:
                part = part[part["origin_date"].dt.year.isin(station_clean_years[station])]
            if split_name == "test":
                part = part[
                    (part["origin_date"] >= pd.Timestamp(TEST_START))
                    & (part["origin_date"] <= pd.Timestamp(TEST_END))
                ]
            n = int(len(part))
            if n > 1 and part["tmax_input"].std() > 0 and part["tmax_obs"].std() > 0:
                corr = float(part["tmax_input"].corr(part["tmax_obs"]))
            else:
                corr = float("nan")
            station_row[f"{split_name}_n"] = n
            station_row[f"{split_name}_corr"] = corr
            if split_name in required_splits and ((not np.isfinite(corr)) or corr < min_corr):
                pass_station = False
        station_row["min_corr"] = float(
            np.nanmin([station_row["train_corr"], station_row["val_corr"], station_row["test_corr"]])
        )
        station_row["pass"] = bool(pass_station)
        station_row["drop_reason"] = "" if pass_station else f"lead1_tmax_corr_below_{min_corr}"
        rows.append(station_row)

    check = pd.DataFrame(rows)
    check.to_csv(out_dir / "alignment_check.csv", index=False)
    failed = check[~check["pass"]]
    if not failed.empty:
        print("!!! ALIGNMENT WARNING: dropping failed stations !!!", flush=True)
        print(failed[["station", "train_corr", "val_corr", "test_corr", "drop_reason"]].to_string(index=False), flush=True)
    passed = check.loc[check["pass"], "station"].astype(str).tolist()
    if not passed:
        raise RuntimeError("Alignment check dropped all stations.")
    print(
        f"[alignment] pass={len(passed)}/{len(stations)} "
        f"min_corr={check['min_corr'].min():.3f} required_splits={','.join(required_splits)} "
        f"output={out_dir / 'alignment_check.csv'}",
        flush=True,
    )
    return passed, check


def build_samples(
    stations: list[str],
    data_dir: Path,
    meta: dict[str, dict[str, Any]],
    input_size: int,
    station_to_id: dict[str, int],
    vocab_stations: list[str] | None = None,
    forecast_only: bool = False,
    phys_feats: bool = False,
    phys_head: bool = False,
    phys_dir: Path | None = None,
    station_clean_years: dict[str, set[int]] | None = None,
    station_to_zone_id: dict[str, int] | None = None,
    absolute_target: bool = False,
    drop_rh: bool = False,
    enforce_label_boundaries: bool = False,
) -> dict[str, Any]:
    """Build origin-level tensors using train-defined vocabularies and optional train-only physical features.

    When ``absolute_target`` is True the per-station climatology is forced to zero,
    so the network target ``yanom`` becomes the absolute ET0 and the reconstruction
    ``yhat = pred + clim`` reduces to ``yhat = pred``. This supports an ablation that
    removes the climatology-anomaly decomposition entirely (no historical climatology
    used anywhere), i.e. a model deployable at a station with zero historical data.
    """
    vocab_stations = vocab_stations or stations
    weather_vocab, wind_vocab = make_vocab(vocab_stations, data_dir, station_clean_years=station_clean_years)
    m3_cache: dict[str, Any] = {}
    ea_cache: dict[str, Any] = {}
    if phys_feats:
        if phys_dir is None:
            raise ValueError("phys_dir is required when phys_feats=True")
        m3_cache, ea_cache = load_physical_caches(phys_dir)

    sample: dict[str, list[Any]] = {
        "past": [],
        "fnum": [],
        "fwea": [],
        "fwind": [],
        "station_id": [],
        "zone_id": [],
        "yanom": [],
        "clim": [],
        "split": [],
        "station": [],
        "origin_date": [],
        "target_date": [],
    }
    if phys_head:
        sample["phys_raw"] = []
        sample["phys_obs"] = []
    required_leads = list(range(1, HORIZON + 1))
    rejected_label_boundaries = 0

    for station in stations:
        if station not in meta:
            raise KeyError(f"Station {station} missing from station metadata")
        if station not in station_to_id:
            raise KeyError(f"Station {station} missing from station_to_id")
        if station_to_zone_id is not None and station not in station_to_zone_id:
            raise KeyError(f"Station {station} missing from station_to_zone_id")

        fc = pd.read_csv(data_dir / "forecast_domain" / f"forecast_modeling_rows_{station}.csv")
        if phys_feats:
            fc = apply_physical_forecast_features(fc, meta[station], m3_cache[station], ea_cache[station])
        hist = pd.read_csv(data_dir / f"clean_history_{station}.csv")
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date").reset_index(drop=True)
        hist["et0_anomaly"] = pd.to_numeric(hist["et0_anomaly"], errors="coerce").fillna(0.0)
        for col in BASE_PAST_COLS:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")
        hist_phys = observed_physical_drivers(hist) if phys_head else None

        by_date = {date: i for i, date in enumerate(hist["date"])}
        past_matrix = hist[BASE_PAST_COLS].to_numpy(dtype=np.float32)
        static = meta[station]["static"]
        latitude = float(meta[station]["latitude"])
        altitude = float(meta[station]["altitude_m"])

        for origin_date, group in fc.groupby("origin_date", sort=True):
            group = group.sort_values("lead").copy()
            group["lead"] = group["lead"].astype(int)
            if group["lead"].astype(int).tolist() != required_leads:
                continue
            split_name = str(group["origin_split"].iloc[0])
            if enforce_label_boundaries and not sequence_within_split(group["target_date"].tolist(), split_name):
                rejected_label_boundaries += 1
                continue
            origin_ts = pd.Timestamp(origin_date)
            if split_name == "train" and station_clean_years is not None and station in station_clean_years:
                origin_year = origin_ts.year
                if origin_year not in station_clean_years[station]:
                    continue
            if split_name == "test" and not (pd.Timestamp(TEST_START) <= origin_ts <= pd.Timestamp(TEST_END)):
                continue

            if forecast_only:
                past = np.zeros((1, 0), dtype=np.float32)
            else:
                idx = by_date.get(pd.Timestamp(origin_date))
                if idx is None or idx - input_size + 1 < 0:
                    continue
                past = past_matrix[idx - input_size + 1 : idx + 1]
                if past.shape != (input_size, len(BASE_PAST_COLS)) or not np.isfinite(past).all():
                    continue

            tmax = pd.to_numeric(group["tmax_input"], errors="coerce").to_numpy(float)
            tmin = pd.to_numeric(group["tmin_input"], errors="coerce").to_numpy(float)
            rh = pd.to_numeric(group["rh_input"], errors="coerce").to_numpy(float)
            rain = pd.to_numeric(group["rain_signal_input"], errors="coerce").to_numpy(float)
            clim = pd.to_numeric(group["climatology_et0"], errors="coerce").to_numpy(float)
            if absolute_target:
                clim = np.zeros_like(clim)
            y = pd.to_numeric(group["y"], errors="coerce").to_numpy(float)
            target_ts = pd.to_datetime(group["target_date"])
            target_index = pd.DatetimeIndex(target_ts)
            doy = target_ts.dt.dayofyear.to_numpy(float)
            dtr = tmax - tmin
            core_cols = [tmax, tmin, dtr]
            if not drop_rh:
                core_cols.append(rh)  # rh is a climatological term, NOT in the public forecast
            core_cols.extend(
                [
                    rain,
                    np.sin(2 * np.pi * doy / 365.25),
                    np.cos(2 * np.pi * doy / 365.25),
                ]
            )
            future_core = np.stack(core_cols, axis=1)
            future_parts = [future_core, np.tile(static, (HORIZON, 1))]
            if phys_feats:
                future_parts.append(
                    np.stack(
                        [
                            pd.to_numeric(group["rs_m3"], errors="coerce").to_numpy(float),
                            pd.to_numeric(group["ea_phys"], errors="coerce").to_numpy(float),
                        ],
                        axis=1,
                    )
                )
            future_num = np.concatenate(future_parts, axis=1)
            if not (np.isfinite(future_num).all() and np.isfinite(y).all() and np.isfinite(clim).all()):
                continue

            if phys_head:
                target_dates = [item.date() for item in target_index]
                ra = np.array([extraterrestrial_radiation(item, latitude)[0] for item in target_dates], dtype=float)
                phys_raw = np.stack(
                    [
                        tmax,
                        tmin,
                        ra,
                        np.full(HORIZON, altitude, dtype=float),
                    ],
                    axis=1,
                )
                if not np.isfinite(phys_raw).all():
                    continue
                assert hist_phys is not None
                phys_obs = hist_phys.reindex(target_index)[PHYS_OBS_COLS].to_numpy(dtype=np.float32)

            sample["past"].append(past)
            sample["fnum"].append(future_num.astype(np.float32))
            sample["fwea"].append(
                clean_category(group["weather_text"])
                .map(lambda token: weather_vocab.get(token, 0))
                .to_numpy(np.int64)
            )
            sample["fwind"].append(
                clean_category(group["wind_text"])
                .map(lambda token: wind_vocab.get(token, 0))
                .to_numpy(np.int64)
            )
            sample["station_id"].append(station_to_id[station])
            sample["zone_id"].append(0 if station_to_zone_id is None else station_to_zone_id[station])
            sample["yanom"].append((y - clim).astype(np.float32))
            sample["clim"].append(clim.astype(np.float32))
            sample["split"].append(split_name)
            sample["station"].append(station)
            sample["origin_date"].append(str(origin_date))
            sample["target_date"].append(group["target_date"].astype(str).to_numpy())
            if phys_head:
                sample["phys_raw"].append(phys_raw.astype(np.float32))
                sample["phys_obs"].append(phys_obs.astype(np.float32))

    if not sample["past"]:
        raise RuntimeError(f"No samples built for stations={stations}, input_size={input_size}")

    out: dict[str, Any] = {}
    for key, value in sample.items():
        out[key] = np.array(value)
    out["weather_n"] = len(weather_vocab) + 1
    out["wind_n"] = len(wind_vocab) + 1
    out["weather_vocab"] = weather_vocab
    out["wind_vocab"] = wind_vocab
    out["station_to_id"] = station_to_id
    out["station_to_zone_id"] = station_to_zone_id or {}
    out["past_cols"] = [] if forecast_only else BASE_PAST_COLS
    out["future_numeric_cols"] = [c for c in future_numeric_cols(phys_feats) if not (drop_rh and c == "rh")]
    out["horizon"] = HORIZON
    out["rejected_label_boundary_origins"] = rejected_label_boundaries
    out["forecast_only"] = bool(forecast_only)
    out["phys_feats"] = bool(phys_feats)
    out["phys_head"] = bool(phys_head)
    if phys_head:
        out["phys_raw_cols"] = PHYS_RAW_COLS
        out["phys_obs_cols"] = PHYS_OBS_COLS
    return out


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        dim = x.shape[-1]
        if dim == 0:
            self.mean = np.zeros(0, dtype=np.float32)
            self.std = np.ones(0, dtype=np.float32)
            return self
        flat = x.reshape(-1, dim)
        self.mean = np.nanmean(flat, axis=0).astype(np.float32)
        self.std = np.nanstd(flat, axis=0).astype(np.float32)
        self.std[~np.isfinite(self.std)] = 1.0
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.shape[-1] == 0:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class PanelE2ENet(nn.Module):
    def __init__(
        self,
        past_dim: int,
        future_num_dim: int,
        weather_n: int,
        wind_n: int,
        hidden: int,
        horizon: int,
        station_n: int = 0,
        emb_dim: int = 8,
        station_emb_dim: int = 8,
        zone_n: int = 0,
        zone_emb_dim: int = 4,
        forecast_only: bool = False,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        future_encoder: str = "transformer",
    ):
        super().__init__()
        self.horizon = horizon
        self.forecast_only = bool(forecast_only)
        self.num_layers = int(num_layers)
        self.nhead = int(nhead)
        self.dropout = float(dropout)
        self.future_encoder = str(future_encoder)
        self.station_emb = nn.Embedding(station_n, station_emb_dim) if station_n > 0 else None
        self.zone_emb = nn.Embedding(zone_n, zone_emb_dim) if zone_n > 0 else None
        extra_station = station_emb_dim if self.station_emb is not None else 0
        extra_zone = zone_emb_dim if self.zone_emb is not None else 0
        if self.forecast_only:
            self.cnn = None
            self.gru = None
        else:
            self.cnn = nn.Sequential(
                nn.Conv1d(past_dim, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.gru = nn.GRU(64, hidden, num_layers=2, batch_first=True)
        self.weather_emb = nn.Embedding(weather_n, emb_dim, padding_idx=0)
        self.wind_emb = nn.Embedding(wind_n, emb_dim, padding_idx=0)
        self.future_proj = nn.Linear(future_num_dim + 2 * emb_dim + extra_station + extra_zone, hidden)
        self.pos = nn.Parameter(torch.randn(1, horizon, hidden) * 0.02)
        encoder = nn.TransformerEncoderLayer(
            hidden,
            nhead=self.nhead,
            dim_feedforward=max(256, 2 * hidden),
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=self.num_layers)
        if self.future_encoder == "cnn_gru_transformer":
            # CNN-GRU-Transformer future encoder over the lead (horizon) sequence.
            self.fcnn = nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.fgru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
            self.fgru_proj = nn.Linear(2 * hidden, hidden)
        else:
            self.fcnn = None
            self.fgru = None
            self.fgru_proj = None
        head_dim = hidden if self.forecast_only else 2 * hidden
        self.head = nn.Sequential(
            nn.Linear(head_dim, max(64, hidden // 2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(max(64, hidden // 2), 1),
        )

    def fused_features(self, past, future_num, future_weather, future_wind, station_id, zone_id=None):
        parts = [future_num, self.weather_emb(future_weather), self.wind_emb(future_wind)]
        if self.station_emb is not None:
            station = self.station_emb(station_id).unsqueeze(1).expand(-1, future_num.shape[1], -1)
            parts.append(station)
        if self.zone_emb is not None:
            if zone_id is None:
                raise ValueError("zone_id is required when zone embeddings are enabled")
            zone = self.zone_emb(zone_id).unsqueeze(1).expand(-1, future_num.shape[1], -1)
            parts.append(zone)
        future = torch.cat(parts, dim=-1)
        proj = self.future_proj(future) + self.pos[:, : future.shape[1], :]
        if self.future_encoder == "cnn_gru_transformer":
            cg = self.fcnn(proj.transpose(1, 2)).transpose(1, 2)
            cg, _ = self.fgru(cg)
            proj = proj + self.fgru_proj(cg)
        future_state = self.transformer(proj)
        if self.forecast_only:
            fused = future_state
        else:
            hist_seq = self.cnn(past.transpose(1, 2)).transpose(1, 2)
            _, hist_hidden = self.gru(hist_seq)
            hist_state = hist_hidden[-1]
            fused = torch.cat([hist_state.unsqueeze(1).expand(-1, future.shape[1], -1), future_state], dim=-1)
        return fused

    def forward(self, past, future_num, future_weather, future_wind, station_id, zone_id=None):
        fused = self.fused_features(past, future_num, future_weather, future_wind, station_id, zone_id)
        return self.head(fused).squeeze(-1)


class PanelPMHeadNet(PanelE2ENet):
    def __init__(
        self,
        past_dim: int,
        future_num_dim: int,
        weather_n: int,
        wind_n: int,
        hidden: int,
        horizon: int,
        station_n: int = 0,
        emb_dim: int = 8,
        station_emb_dim: int = 8,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(
            past_dim=past_dim,
            future_num_dim=future_num_dim,
            weather_n=weather_n,
            wind_n=wind_n,
            hidden=hidden,
            horizon=horizon,
            station_n=station_n,
            emb_dim=emb_dim,
            station_emb_dim=station_emb_dim,
            zone_n=0,
            forecast_only=True,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout,
        )
        self.head = nn.Identity()
        self.rs_head = nn.Linear(hidden, 1)
        self.ea_head = nn.Linear(hidden, 1)
        self.u2_head = nn.Linear(hidden, 1)

    def forward(
        self,
        past,
        future_num,
        future_weather,
        future_wind,
        station_id,
        zone_id=None,
        phys_raw=None,
        return_components: bool = False,
    ):
        if phys_raw is None:
            raise ValueError("phys_raw is required for PanelPMHeadNet")
        fused = self.fused_features(past, future_num, future_weather, future_wind, station_id, None)
        raw_rs = self.rs_head(fused).squeeze(-1)
        raw_ea = self.ea_head(fused).squeeze(-1)
        raw_u2 = self.u2_head(fused).squeeze(-1)

        tmax = phys_raw[..., 0]
        tmin = phys_raw[..., 1]
        ra = phys_raw[..., 2]
        altitude = phys_raw[..., 3]
        es = (torch_saturation_vapor_pressure(tmax) + torch_saturation_vapor_pressure(tmin)) / 2.0
        rso = (0.75 + 2e-5 * altitude) * ra
        rs = rso * torch.sigmoid(raw_rs)
        ea = es * torch.sigmoid(raw_ea)
        u2 = F.softplus(raw_u2)
        et0 = torch_fao56_pm_daily_rs(tmax, tmin, ra, altitude, rs, ea, u2)
        if return_components:
            return {"et0": et0, "rs": rs, "ea": ea, "u2": u2}
        return et0


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return {col: float("nan") for col in METRIC_COLS}
    yy = y[mask]
    pp = yhat[mask]
    err = pp - yy
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    nse = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if len(yy) > 1 and np.std(yy) > 0 and np.std(pp) > 0:
        r2 = float(np.corrcoef(yy, pp)[0, 1] ** 2)
    else:
        r2 = float("nan")
    mean_et0 = float(np.mean(yy))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(err))),
        "r2": r2,
        "nse": nse,
        "bias": float(np.mean(err)),
        "mean_et0": mean_et0,
        "nrmse": float(rmse / mean_et0) if mean_et0 else float("nan"),
    }


def split_mask(samples: dict[str, Any], split_name: str, stations: list[str]) -> np.ndarray:
    return (samples["split"] == split_name) & np.isin(samples["station"], stations)


def pack_tensors(
    samples: dict[str, Any],
    mask: np.ndarray,
    past_std: Standardizer,
    future_std: Standardizer,
    y_mean: float,
    y_std: float,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(past_std.transform(samples["past"][mask]), dtype=torch.float32),
        torch.tensor(future_std.transform(samples["fnum"][mask]), dtype=torch.float32),
        torch.tensor(samples["fwea"][mask], dtype=torch.long),
        torch.tensor(samples["fwind"][mask], dtype=torch.long),
        torch.tensor(samples["station_id"][mask], dtype=torch.long),
        torch.tensor(samples.get("zone_id", np.zeros(len(samples["station_id"]), dtype=np.int64))[mask], dtype=torch.long),
        torch.tensor(((samples["yanom"][mask] - y_mean) / y_std).astype(np.float32), dtype=torch.float32),
    )


def pack_phys_tensors(
    samples: dict[str, Any],
    mask: np.ndarray,
    past_std: Standardizer,
    future_std: Standardizer,
) -> tuple[torch.Tensor, ...]:
    y_abs = (samples["yanom"][mask].astype(np.float32) + samples["clim"][mask].astype(np.float32)).astype(np.float32)
    return (
        torch.tensor(past_std.transform(samples["past"][mask]), dtype=torch.float32),
        torch.tensor(future_std.transform(samples["fnum"][mask]), dtype=torch.float32),
        torch.tensor(samples["fwea"][mask], dtype=torch.long),
        torch.tensor(samples["fwind"][mask], dtype=torch.long),
        torch.tensor(samples["station_id"][mask], dtype=torch.long),
        torch.tensor(samples.get("zone_id", np.zeros(len(samples["station_id"]), dtype=np.int64))[mask], dtype=torch.long),
        torch.tensor(samples["phys_raw"][mask], dtype=torch.float32),
        torch.tensor(samples["phys_obs"][mask], dtype=torch.float32),
        torch.tensor(y_abs, dtype=torch.float32),
    )


def train_phys_scales(samples: dict[str, Any], mask: np.ndarray) -> dict[str, float]:
    obs = samples["phys_obs"][mask].reshape(-1, len(PHYS_OBS_COLS))
    out: dict[str, float] = {}
    for idx, name in enumerate(PHYS_OBS_COLS):
        values = obs[:, idx]
        finite = values[np.isfinite(values)]
        scale = float(np.nanstd(finite)) if len(finite) else float("nan")
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.nanmean(np.abs(finite))) if len(finite) else float("nan")
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        out[name] = scale
    return out


def _masked_scaled_aux_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    scale: float,
    smooth: bool = False,
) -> torch.Tensor:
    mask = torch.isfinite(pred) & torch.isfinite(obs)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    diff = (pred[mask] - obs[mask]) / float(scale)
    if smooth:
        return F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="mean")
    return torch.mean(torch.abs(diff))


def physical_head_loss(
    outputs: dict[str, torch.Tensor],
    y_abs: torch.Tensor,
    phys_obs: torch.Tensor,
    scales: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    et0_loss = torch.mean(torch.abs(outputs["et0"] - y_abs))
    rs_loss = _masked_scaled_aux_loss(outputs["rs"], phys_obs[..., 0], scales["rs_obs"], smooth=True)
    ea_loss = _masked_scaled_aux_loss(outputs["ea"], phys_obs[..., 1], scales["ea_obs"])
    u2_loss = _masked_scaled_aux_loss(outputs["u2"], phys_obs[..., 2], scales["u2_obs"])
    total = et0_loss + 0.1 * rs_loss + 0.05 * ea_loss + 0.05 * u2_loss
    return total, {
        "et0_l1": float(et0_loss.detach().cpu()),
        "rs_smooth_l1_scaled": float(rs_loss.detach().cpu()),
        "ea_l1_scaled": float(ea_loss.detach().cpu()),
        "u2_l1_scaled": float(u2_loss.detach().cpu()),
    }


def pm_sanity_check(samples: dict[str, Any], train_mask: np.ndarray, seed: int, rows: int = 2000) -> dict[str, Any]:
    if "phys_raw" not in samples or "phys_obs" not in samples:
        raise RuntimeError("PM sanity check requires phys_raw and phys_obs sample fields.")
    train_idx = np.flatnonzero(train_mask)
    raw = samples["phys_raw"][train_idx]
    obs = samples["phys_obs"][train_idx]
    lat_idx = samples["future_numeric_cols"].index("latitude")
    lat = samples["fnum"][train_idx, :, lat_idx]
    valid = np.isfinite(raw).all(axis=2) & np.isfinite(obs).all(axis=2) & np.isfinite(lat)
    flat = np.argwhere(valid)
    if len(flat) < rows:
        raise RuntimeError(f"PM sanity needs at least {rows} finite train lead rows, found {len(flat)}")

    rng = np.random.default_rng(seed + SEED_OFFSET)
    chosen = flat[rng.choice(len(flat), size=rows, replace=False)]
    sample_pos = chosen[:, 0]
    lead_pos = chosen[:, 1]
    raw_sel = raw[sample_pos, lead_pos].astype(np.float64)
    obs_sel = obs[sample_pos, lead_pos].astype(np.float64)
    lat_sel = lat[sample_pos, lead_pos].astype(np.float64)
    target_dates = samples["target_date"][train_idx][sample_pos, lead_pos]

    raw_t = torch.tensor(raw_sel, dtype=torch.float64)
    obs_t = torch.tensor(obs_sel, dtype=torch.float64)
    with torch.no_grad():
        torch_pred = torch_fao56_pm_daily_rs(
            raw_t[:, 0],
            raw_t[:, 1],
            raw_t[:, 2],
            raw_t[:, 3],
            obs_t[:, 0],
            obs_t[:, 1],
            obs_t[:, 2],
        ).cpu().numpy()

    reference = np.empty(rows, dtype=np.float64)
    for idx in range(rows):
        tmax = float(raw_sel[idx, 0])
        tmin = float(raw_sel[idx, 1])
        reference[idx] = fao56_pm_daily_rs(
            pd.Timestamp(target_dates[idx]).date(),
            float(lat_sel[idx]),
            float(raw_sel[idx, 3]),
            (tmax + tmin) / 2.0,
            tmax,
            tmin,
            float(obs_sel[idx, 1]),
            float(obs_sel[idx, 2]),
            float(obs_sel[idx, 0]),
        )
    max_abs_delta = float(np.max(np.abs(torch_pred - reference)))
    passed = bool(max_abs_delta < 1e-3)
    print(
        f"[pm-sanity] rows={rows} finite_train_rows={len(flat)} "
        f"max_abs_delta={max_abs_delta:.8g} threshold=0.001 pass={passed}",
        flush=True,
    )
    if not passed:
        raise RuntimeError(f"PM sanity failed: max_abs_delta={max_abs_delta}")
    return {"rows": int(rows), "finite_train_rows": int(len(flat)), "max_abs_delta": max_abs_delta, "pass": passed}


def ytrue_and_clim(samples: dict[str, Any], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clim = samples["clim"][mask].astype(np.float32)
    return samples["yanom"][mask].astype(np.float32) + clim, clim


def pred_anomaly(model: nn.Module, tensors: tuple[torch.Tensor, ...], y_mean: float, y_std: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(*tensors[:-1]).cpu().numpy()
    return pred * y_std + y_mean


def pred_absolute_phys(model: nn.Module, tensors: tuple[torch.Tensor, ...]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(*tensors[:6], phys_raw=tensors[6]).cpu().numpy()
    return pred


def rmse_for_arrays(ytrue: np.ndarray, pred_anom: np.ndarray, clim: np.ndarray) -> float:
    yhat = np.clip(pred_anom + clim, 0, None)
    return metrics(ytrue.ravel(), yhat.ravel())["rmse"]


def rmse_for_absolute_arrays(ytrue: np.ndarray, pred_abs: np.ndarray) -> float:
    yhat = np.clip(pred_abs, 0, None)
    return metrics(ytrue.ravel(), yhat.ravel())["rmse"]


def lead_sort(value: Any) -> int:
    text = str(value)
    if text.isdigit():
        return int(text)
    return {"1-7": 101}.get(text, 999)


def score_prediction_frame(station: str, strategy: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lead in range(1, HORIZON + 1):
        part = frame[frame["lead"].astype(int) == lead]
        rows.append({"station": station, "strategy": strategy, "lead": str(lead), "n": int(len(part)), **metrics(part["y"], part["yhat"])})
    rows.append({"station": station, "strategy": strategy, "lead": "1-7", "n": int(len(frame)), **metrics(frame["y"], frame["yhat"])})
    return rows


def write_prediction_table(
    path: Path,
    samples: dict[str, Any],
    mask: np.ndarray,
    pred: np.ndarray,
    split_name: str,
) -> None:
    ytrue, clim = ytrue_and_clim(samples, mask)
    yhat = np.clip(pred + clim, 0, None)
    stations = samples["station"][mask]
    origins = samples["origin_date"][mask]
    targets = samples["target_date"][mask]
    rows: list[dict[str, Any]] = []
    for i in range(yhat.shape[0]):
        for j in range(yhat.shape[1]):
            rows.append(
                {
                    "station": str(stations[i]),
                    "origin_date": str(origins[i]),
                    "target_date": str(targets[i, j]),
                    "lead": int(j + 1),
                    "origin_split": split_name,
                    "y": float(ytrue[i, j]),
                    "yhat": float(yhat[i, j]),
                    "climatology": float(clim[i, j]),
                    "pred_anomaly": float(pred[i, j]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_absolute_prediction_table(
    path: Path,
    samples: dict[str, Any],
    mask: np.ndarray,
    pred_abs: np.ndarray,
    split_name: str,
) -> None:
    ytrue, clim = ytrue_and_clim(samples, mask)
    yhat = np.clip(pred_abs, 0, None)
    stations = samples["station"][mask]
    origins = samples["origin_date"][mask]
    targets = samples["target_date"][mask]
    rows: list[dict[str, Any]] = []
    for i in range(yhat.shape[0]):
        for j in range(yhat.shape[1]):
            rows.append(
                {
                    "station": str(stations[i]),
                    "origin_date": str(origins[i]),
                    "target_date": str(targets[i, j]),
                    "lead": int(j + 1),
                    "origin_split": split_name,
                    "y": float(ytrue[i, j]),
                    "yhat": float(yhat[i, j]),
                    "climatology": float(clim[i, j]),
                    "pred_anomaly": float(yhat[i, j] - clim[i, j]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


@dataclass
class TrainJob:
    strategy: str
    station: str
    seed: int
    train_stations: list[str]
    eval_stations: list[str]
    vocab_stations: list[str]
    norm_stations: list[str]
    station_to_id: dict[str, int]
    use_station_emb: bool
    input_size: int
    hidden: int
    epochs: int
    patience: int
    batch_size: int
    lr: float
    worker_threads: int
    out_dir: str
    forecast_only: bool = False
    phys_feats: bool = False
    phys_head: bool = False
    phys_dir: str | None = None
    zone_to_id: dict[str, int] | None = None
    station_to_zone_id: dict[str, int] | None = None
    use_zone_emb: bool = False
    station_clean_years: dict[str, set[int]] | None = None
    loss_name: str = "l1"
    scheduler_name: str = "cosine"
    num_layers: int = 2
    nhead: int = 4
    dropout: float = 0.1
    warmup_epochs: int = 0
    skip_prediction_csv: bool = False
    future_encoder: str = "transformer"
    absolute_target: bool = False
    train_splits: tuple[str, ...] | None = None
    val_split: str = "val"
    drop_rh: bool = False

    # Explicitly separate model-selection locations from final test locations.
    validation_stations: list[str] | None = None
    spatial_selection_guard: bool = False


def run_train_job(job: TrainJob) -> dict[str, Any]:
    validation_stations = resolve_validation_stations(job)
    set_torch_worker_threads(job.worker_threads)
    set_seed(job.seed)
    data_dir = DATA_DIR
    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if job.phys_head and not job.forecast_only:
        raise ValueError("phys_head requires forecast_only=True")
    if job.phys_head and job.use_zone_emb:
        raise ValueError("phys_head does not use zone embeddings")
    meta = station_meta(ROOT / "data" / "db" / "rice_rl_data.db")
    needed = set(job.train_stations) | set(job.eval_stations) | set(job.norm_stations) | set(validation_stations)
    context_stations = [station for station in PANEL_STATIONS if station in needed]
    samples = build_samples(
        context_stations,
        data_dir,
        meta,
        input_size=job.input_size,
        station_to_id=job.station_to_id,
        vocab_stations=job.vocab_stations,
        forecast_only=job.forecast_only,
        phys_feats=job.phys_feats,
        phys_head=job.phys_head,
        phys_dir=Path(job.phys_dir) if job.phys_dir else None,
        station_clean_years=job.station_clean_years,
        station_to_zone_id=job.station_to_zone_id if job.use_zone_emb else None,
        absolute_target=job.absolute_target,
        drop_rh=job.drop_rh,
        enforce_label_boundaries=True,
    )

    train_split_names = job.train_splits if job.train_splits else ("train",)
    train = np.zeros(len(samples["split"]), dtype=bool)
    for _split_name in train_split_names:
        train = train | split_mask(samples, _split_name, job.train_stations)
    val = split_mask(samples, job.val_split, validation_stations)
    test = split_mask(samples, "test", job.eval_stations)
    # Production "all-years" mode fits on train+val and uses test as the early-stop
    # monitor; norm statistics stay fit on the same splits used for fitting (train).
    norm = np.zeros(len(samples["split"]), dtype=bool)
    for _split_name in train_split_names:
        norm = norm | split_mask(samples, _split_name, job.norm_stations)
    if not train.any() or not val.any() or not test.any() or not norm.any():
        raise RuntimeError(
            f"{job.strategy}/{job.station}/seed{job.seed}: empty masks "
            f"train={train.sum()} val={val.sum()} test={test.sum()} norm={norm.sum()}"
        )

    past_std = Standardizer().fit(samples["past"][norm])
    future_std = Standardizer().fit(samples["fnum"][norm])
    y_mean = float(samples["yanom"][norm].mean())
    y_std = float(samples["yanom"][norm].std())
    if not math.isfinite(y_std) or y_std < 1e-6:
        y_std = 1.0

    pm_sanity: dict[str, Any] | None = None
    phys_scales: dict[str, float] | None = None
    if job.phys_head:
        pm_sanity = pm_sanity_check(samples, train, seed=job.seed)
        phys_scales = train_phys_scales(samples, train)
        tensors_train = pack_phys_tensors(samples, train, past_std, future_std)
        tensors_val = pack_phys_tensors(samples, val, past_std, future_std)
        print(
            f"[phys-head] train_scales rs={phys_scales['rs_obs']:.5g} "
            f"ea={phys_scales['ea_obs']:.5g} u2={phys_scales['u2_obs']:.5g}",
            flush=True,
        )
    else:
        tensors_train = pack_tensors(samples, train, past_std, future_std, y_mean, y_std)
        tensors_val = pack_tensors(samples, val, past_std, future_std, y_mean, y_std)
    val_ytrue, val_clim = ytrue_and_clim(samples, val)

    if job.phys_head:
        model = PanelPMHeadNet(
            past_dim=samples["past"].shape[-1],
            future_num_dim=samples["fnum"].shape[-1],
            weather_n=int(samples["weather_n"]),
            wind_n=int(samples["wind_n"]),
            hidden=job.hidden,
            horizon=int(samples["horizon"]),
            station_n=len(job.station_to_id) if job.use_station_emb else 0,
            num_layers=job.num_layers,
            nhead=job.nhead,
            dropout=job.dropout,
        )
        loss_fn = None
    else:
        model = PanelE2ENet(
            past_dim=samples["past"].shape[-1],
            future_num_dim=samples["fnum"].shape[-1],
            weather_n=int(samples["weather_n"]),
            wind_n=int(samples["wind_n"]),
            hidden=job.hidden,
            horizon=int(samples["horizon"]),
            station_n=len(job.station_to_id) if job.use_station_emb else 0,
            zone_n=len(job.zone_to_id or {}) if job.use_zone_emb else 0,
            forecast_only=job.forecast_only,
            num_layers=job.num_layers,
            nhead=job.nhead,
            dropout=job.dropout,
            future_encoder=job.future_encoder,
        )

        loss_key = job.loss_name.lower()
        if loss_key == "mse":
            loss_fn = nn.MSELoss()
        elif loss_key == "l1":
            loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_name={job.loss_name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=job.lr, weight_decay=1e-5)
    scheduler_key = job.scheduler_name.lower()
    if scheduler_key == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, job.patience // 3),
            min_lr=job.lr * 0.01,
        )
    elif scheduler_key == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=job.epochs, eta_min=job.lr * 0.05)
    elif scheduler_key == "cosine_warmup":
        warmup_epochs = max(0, min(int(job.warmup_epochs), int(job.epochs)))
        cosine_epochs = max(1, int(job.epochs) - warmup_epochs)
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=0.0,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(job.epochs)),
                eta_min=0.0,
            )
    else:
        raise ValueError(f"Unknown scheduler_name={job.scheduler_name}")
    loader = DataLoader(
        TensorDataset(*tensors_train),
        batch_size=job.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(job.seed + SEED_OFFSET),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    bad = 0
    epochs_ran = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(job.epochs):
        model.train()
        train_loss_total = 0.0
        train_loss_count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if job.phys_head:
                assert phys_scales is not None
                outputs = model(*batch[:6], phys_raw=batch[6], return_components=True)
                loss, _ = physical_head_loss(outputs, batch[-1], batch[7], phys_scales)
            else:
                assert loss_fn is not None
                loss = loss_fn(model(*batch[:-1]), batch[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            batch_n = int(batch[0].shape[0])
            train_loss_total += float(loss.detach().cpu().item()) * batch_n
            train_loss_count += batch_n
        train_loss = train_loss_total / max(1, train_loss_count)

        model.eval()
        with torch.no_grad():
            if job.phys_head:
                assert phys_scales is not None
                val_outputs = model(*tensors_val[:6], phys_raw=tensors_val[6], return_components=True)
                val_loss_tensor, _ = physical_head_loss(val_outputs, tensors_val[-1], tensors_val[7], phys_scales)
                val_loss = float(val_loss_tensor.item())
                val_pred_abs = val_outputs["et0"].cpu().numpy()
                val_rmse = rmse_for_absolute_arrays(val_ytrue, val_pred_abs)
            else:
                assert loss_fn is not None
                val_pred_std = model(*tensors_val[:-1])
                val_loss = float(loss_fn(val_pred_std, tensors_val[-1]).item())
                val_pred = val_pred_std.cpu().numpy() * y_std + y_mean
                val_rmse = rmse_for_arrays(val_ytrue, val_pred, val_clim)
        if scheduler_key == "plateau":
            scheduler.step(val_rmse)
        else:
            scheduler.step()
        epochs_ran = epoch + 1
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_rmse": float(val_rmse),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if os.environ.get("PANEL_DL_VERBOSE"):
            print(
                f"[epoch] {epoch + 1}/{job.epochs} train_loss={train_loss:.4f} "
                f"val_rmse={val_rmse:.4f} best={best_val_rmse:.4f}@{best_epoch} bad={bad}",
                flush=True,
            )
        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= job.patience:
                break

    if best_state is None:
        raise RuntimeError(f"{job.strategy}/{job.station}/seed{job.seed}: no best model state")
    model.load_state_dict(best_state)

    checkpoint_path = out_dir / f"checkpoint_{job.strategy}_{job.station}_seed{job.seed}.pt"
    torch.save(
        {
            "model_state": best_state,
            "job": asdict(job),
            "standardizers": {"past": past_std.to_json(), "future": future_std.to_json()},
            "target_scale": {"mean": y_mean, "std": y_std},
            "vocab_sizes": {"weather": int(samples["weather_n"]), "wind": int(samples["wind_n"])},
            "vocabularies": {"weather": samples["weather_vocab"], "wind": samples["wind_vocab"]},
            "selection_protocol": {
                "validation_stations": validation_stations,
                "spatial_selection_guard": job.spatial_selection_guard,
                "rejected_label_boundary_origins": samples.get("rejected_label_boundary_origins", 0),
                "scope": "station membership only; HPO and feature provenance require separate audit",
            },
            "past_cols": samples["past_cols"],
            "future_numeric_cols": samples["future_numeric_cols"],
            "forecast_only": bool(job.forecast_only),
            "phys_feats": bool(job.phys_feats),
            "phys_head": bool(job.phys_head),
            "phys_raw_cols": samples.get("phys_raw_cols", []),
            "phys_obs_cols": samples.get("phys_obs_cols", []),
            "phys_scales": phys_scales or {},
            "pm_sanity": pm_sanity or {},
            "use_zone_emb": bool(job.use_zone_emb),
            "zone_to_id": job.zone_to_id or {},
            "station_to_zone_id": job.station_to_zone_id or {},
            "loss_name": job.loss_name,
            "scheduler_name": job.scheduler_name,
            "num_layers": job.num_layers,
            "nhead": job.nhead,
            "dropout": job.dropout,
            "warmup_epochs": job.warmup_epochs,
        },
        checkpoint_path,
    )

    prediction_paths: list[str] = []
    val_prediction_paths: list[str] = []
    test_rmse_by_station: dict[str, float] = {}
    test_rmse_1_7_by_station: dict[str, float] = {}
    for station in sorted(set(job.eval_stations) | set(validation_stations)):
        for split_name, split_values, sink in [
            ("val", val, val_prediction_paths),
            ("test", test, prediction_paths),
        ]:
            station_mask = split_values & (samples["station"] == station)
            if not station_mask.any():
                continue
            split_ytrue, split_clim = ytrue_and_clim(samples, station_mask)
            if job.phys_head:
                tensors_split = pack_phys_tensors(samples, station_mask, past_std, future_std)
                split_pred_abs = pred_absolute_phys(model, tensors_split)
                if not job.skip_prediction_csv:
                    pred_path = prediction_csv_path(out_dir, job.strategy, station, job.seed, split_name)
                    write_absolute_prediction_table(pred_path, samples, station_mask, split_pred_abs, split_name=split_name)
            else:
                tensors_split = pack_tensors(samples, station_mask, past_std, future_std, y_mean, y_std)
                split_pred = pred_anomaly(model, tensors_split, y_mean, y_std)
                if not job.skip_prediction_csv:
                    pred_path = prediction_csv_path(out_dir, job.strategy, station, job.seed, split_name)
                    write_prediction_table(pred_path, samples, station_mask, split_pred, split_name=split_name)
            if not job.skip_prediction_csv:
                sink.append(str(pred_path))
            if split_name == "test":
                if job.phys_head:
                    test_rmse_by_station[station] = rmse_for_absolute_arrays(split_ytrue, split_pred_abs)
                    split_yhat = np.clip(split_pred_abs, 0, None)
                else:
                    test_rmse_by_station[station] = rmse_for_arrays(split_ytrue, split_pred, split_clim)
                    split_yhat = np.clip(split_pred + split_clim, 0, None)
                lead_stop = min(7, split_ytrue.shape[1])
                test_rmse_1_7_by_station[station] = metrics(
                    split_ytrue[:, :lead_stop].ravel(),
                    split_yhat[:, :lead_stop].ravel(),
                )["rmse"]

    return {
        "strategy": job.strategy,
        "station": job.station,
        "seed": job.seed,
        "input_size": job.input_size,
        "forecast_only": bool(job.forecast_only),
        "phys_feats": bool(job.phys_feats),
        "phys_head": bool(job.phys_head),
        "pm_sanity": pm_sanity or {},
        "phys_scales": phys_scales or {},
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "history": history,
        "train_origins": int(train.sum()),
        "val_origins": int(val.sum()),
        "test_origins": int(test.sum()),
        "norm_origins": int(norm.sum()),
        "checkpoint": str(checkpoint_path),
        "prediction_paths": prediction_paths,
        "val_prediction_paths": val_prediction_paths,
        "test_rmse_by_station": test_rmse_by_station,
        "test_rmse_1_7_by_station": test_rmse_1_7_by_station,
    }


def run_jobs(jobs: list[TrainJob], max_workers: int, stage: str) -> list[dict[str, Any]]:
    if not jobs:
        return []
    print(f"\n[{stage}] launching {len(jobs)} jobs with max_workers={max_workers}", flush=True)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_train_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            rmse_txt = ", ".join(f"{k}:{v:.4f}" for k, v in result["test_rmse_by_station"].items())
            print(
                f"[{stage}] done {job.strategy} station={job.station} seed={job.seed} "
                f"best_epoch={result['best_epoch']} epochs={result['epochs_ran']} "
                f"val_rmse={result['best_val_rmse']:.4f} test_rmse={rmse_txt}",
                flush=True,
            )
    return sorted(results, key=lambda r: (r["strategy"], r["station"], r["seed"]))


def read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station": str, "origin_date": str, "target_date": str})
    df["lead"] = df["lead"].astype(int)
    return df.sort_values(KEY_COLS).reset_index(drop=True)


def ensemble_prediction(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("No prediction paths supplied")
    frames = [read_prediction(path) for path in paths]
    base = frames[0][KEY_COLS + ["origin_split", "y", "climatology"]].copy()
    preds = []
    for frame in frames:
        if not frame[KEY_COLS].equals(base[KEY_COLS]):
            raise RuntimeError(f"Prediction keys do not align for ensemble: {paths}")
        preds.append(frame["pred_anomaly"].to_numpy(float))
    base["pred_anomaly"] = np.mean(np.stack(preds, axis=0), axis=0)
    base["yhat"] = np.clip(base["pred_anomaly"] + base["climatology"], 0, None)
    return base


def write_ensemble_and_metrics(
    out_dir: Path,
    strategy: str,
    stations: list[str],
    seeds: list[int],
    split_name: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    ensembles: dict[str, pd.DataFrame] = {}
    for station in stations:
        paths = [prediction_csv_path(out_dir, strategy, station, seed, split_name) for seed in seeds]
        ens = ensemble_prediction(paths)
        ens_path = out_dir / f"ensemble_{split_name}_predictions_{strategy}_{station}.csv"
        ens.to_csv(ens_path, index=False)
        ensembles[station] = ens
        if split_name == "test":
            rows.extend(score_prediction_frame(station, strategy, ens))
    return rows, ensembles


def test_mean_et0_lookup(stations: list[str], data_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for station in stations:
        fc = pd.read_csv(
            data_dir / "forecast_domain" / f"forecast_modeling_rows_{station}.csv",
            usecols=["origin_split", "target_split", "lead", "y"],
        )
        fc = fc[
            (fc["origin_split"].astype(str) == "test")
            & (fc["target_split"].astype(str) == "test")
            & (fc["lead"].astype(int).between(1, HORIZON))
        ].copy()
        fc["y"] = pd.to_numeric(fc["y"], errors="coerce")
        for lead, group in fc.groupby("lead", sort=True):
            y = group["y"].dropna().to_numpy(float)
            mean_et0 = float(np.mean(y)) if len(y) else float("nan")
            rows.append(
                {
                    "station": station,
                    "lead": str(int(lead)),
                    "mean_et0": mean_et0,
                }
            )
        y_all = fc["y"].dropna().to_numpy(float)
        rows.append({"station": station, "lead": "1-7", "mean_et0": float(np.mean(y_all)) if len(y_all) else float("nan")})
    return pd.DataFrame(rows)


def physical_baseline_rows(baseline_dir: Path, stations: list[str], data_dir: Path) -> pd.DataFrame:
    by_lead_path = baseline_dir / "et0_baseline_by_lead.csv"
    summary_path = baseline_dir / "et0_baseline_summary.csv"
    if not by_lead_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing physical baseline outputs under {baseline_dir}")

    by_lead = pd.read_csv(by_lead_path, dtype={"station_code": str, "lead": str})
    summary = pd.read_csv(summary_path, dtype={"station_code": str})
    by_lead = by_lead[by_lead["method"].isin(["PMTM3", "PMF", "HS", "climatology"])].copy()
    summary = summary[summary["method"].isin(["PMTM3", "PMF", "HS", "climatology"])].copy()
    summary["lead"] = "1-7"
    summary = summary.drop(columns=["lead_group"])
    df = pd.concat([by_lead, summary], ignore_index=True)
    df = df[df["station_code"].isin(stations)].copy()
    df = df.rename(columns={"station_code": "station", "method": "strategy", "ef": "nse"})
    keep = ["station", "strategy", "lead", "n", "rmse", "mae", "r2", "nse", "bias"]
    df = df[keep]
    means = test_mean_et0_lookup(stations, data_dir)
    df = df.merge(means, on=["station", "lead"], how="left")
    df["nrmse"] = pd.to_numeric(df["rmse"], errors="coerce") / pd.to_numeric(df["mean_et0"], errors="coerce")
    return df


def apply_global_bias_correction(
    out_dir: Path,
    strategy: str,
    stations: list[str],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, val_ensembles = write_ensemble_and_metrics(out_dir, strategy, stations, seeds, split_name="val")
    _, test_ensembles = write_ensemble_and_metrics(out_dir, strategy, stations, seeds, split_name="test")
    val_all = pd.concat(val_ensembles.values(), ignore_index=True)
    global_delta = float((val_all["yhat"].to_numpy(float) - val_all["y"].to_numpy(float)).mean())
    corrected_strategy = f"{strategy}_global_bc"
    rows: list[dict[str, Any]] = []
    for station, frame in test_ensembles.items():
        corrected = frame.copy()
        corrected["yhat"] = np.clip(corrected["yhat"].astype(float) - global_delta, 0, None)
        corrected["pred_anomaly"] = corrected["yhat"].astype(float) - corrected["climatology"].astype(float)
        corrected.to_csv(out_dir / f"ensemble_test_predictions_{corrected_strategy}_{station}.csv", index=False)
        rows.extend(score_prediction_frame(station, corrected_strategy, corrected))
    payload = {
        "strategy": strategy,
        "corrected_strategy": corrected_strategy,
        "source_split": "val",
        "source_rows": int(len(val_all)),
        "global_delta_yhat_minus_y": global_delta,
    }
    write_json(out_dir / "global_bias_correction.json", payload)
    print(
        f"[bc] best_u={strategy} val_global_delta={global_delta:+.5f} "
        f"corrected_strategy={corrected_strategy}",
        flush=True,
    )
    return rows, payload


def build_panel_summary(master: pd.DataFrame, dl_strategies: list[str]) -> pd.DataFrame:
    one7 = master[master["lead"].astype(str) == "1-7"].copy()
    rows: list[dict[str, Any]] = []
    s_lookup = one7[one7["strategy"] == "S_single"].set_index("station")["rmse"].to_dict()
    best_lookup: dict[str, str] = {}
    best_rmse_lookup: dict[str, float] = {}
    for station, group in one7[one7["strategy"].isin(dl_strategies)].groupby("station"):
        best = group.sort_values("rmse").iloc[0]
        best_lookup[station] = str(best["strategy"])
        best_rmse_lookup[station] = float(best["rmse"])

    for _, row in one7.iterrows():
        station = str(row["station"])
        s_rmse = s_lookup.get(station, np.nan)
        delta = float(row["rmse"] - s_rmse) if np.isfinite(s_rmse) else np.nan
        improve_pct = float((s_rmse - row["rmse"]) / s_rmse * 100.0) if np.isfinite(s_rmse) and s_rmse else np.nan
        rows.append(
            {
                "row_type": "station",
                "station": station,
                "strategy": row["strategy"],
                "station_count": 1,
                **{col: row[col] for col in METRIC_COLS},
                "best_dl_strategy": best_lookup.get(station),
                "best_dl_rmse": best_rmse_lookup.get(station),
                "rmse_delta_vs_S_single": delta,
                "rmse_improvement_pct_vs_S_single": improve_pct,
            }
        )

    for strategy, group in one7.groupby("strategy", sort=False):
        metric_means = group[METRIC_COLS].mean(numeric_only=True)
        s_panel = one7[one7["strategy"] == "S_single"]["rmse"].mean()
        delta = float(metric_means["rmse"] - s_panel) if np.isfinite(s_panel) else np.nan
        improve_pct = float((s_panel - metric_means["rmse"]) / s_panel * 100.0) if np.isfinite(s_panel) and s_panel else np.nan
        rows.append(
            {
                "row_type": "panel_mean",
                "station": "PANEL_MEAN",
                "strategy": strategy,
                "station_count": int(group["station"].nunique()),
                **{col: metric_means[col] for col in METRIC_COLS},
                "best_dl_strategy": "",
                "best_dl_rmse": np.nan,
                "rmse_delta_vs_S_single": delta,
                "rmse_improvement_pct_vs_S_single": improve_pct,
            }
        )
    return pd.DataFrame(rows)


def build_ablation_2x2(master: pd.DataFrame) -> pd.DataFrame:
    one7 = master[(master["lead"].astype(str) == "1-7") & (master["strategy"].isin(U_STRATEGY_ORDER))].copy()
    panel_rows: list[dict[str, Any]] = []
    station_rows: list[pd.DataFrame] = []
    for strategy in U_STRATEGY_ORDER:
        group = one7[one7["strategy"] == strategy]
        flags = U_STRATEGIES[strategy]
        means = group[METRIC_COLS].mean(numeric_only=True)
        panel_rows.append(
            {
                "row_type": "panel_mean",
                "station": "PANEL_MEAN",
                "strategy": strategy,
                "forecast_only": bool(flags["forecast_only"]),
                "phys_feats": bool(flags["phys_feats"]),
                "station_count": int(group["station"].nunique()),
                **{col: means[col] for col in METRIC_COLS},
            }
        )
        part = group.copy()
        part["row_type"] = "station"
        part["forecast_only"] = bool(flags["forecast_only"])
        part["phys_feats"] = bool(flags["phys_feats"])
        station_rows.append(part[["row_type", "station", "strategy", "forecast_only", "phys_feats", "n", *METRIC_COLS]])

    out = pd.concat([pd.DataFrame(panel_rows), *station_rows], ignore_index=True)
    panel = out[out["row_type"] == "panel_mean"].set_index("strategy")
    if all(strategy in panel.index for strategy in U_STRATEGY_ORDER):
        out["delta_vs_U_hist_nophys"] = out["rmse"] - float(panel.loc["U_hist_nophys", "rmse"])
        comparisons = {
            "U_FO_nophys": "U_hist_nophys",
            "U_hist_phys": "U_hist_nophys",
            "U_FO_phys": "U_hist_phys",
        }
        out["delta_vs_matched_reference"] = np.nan
        for strategy, ref in comparisons.items():
            mask = (out["row_type"] == "panel_mean") & (out["strategy"] == strategy)
            out.loc[mask, "delta_vs_matched_reference"] = float(panel.loc[strategy, "rmse"] - panel.loc[ref, "rmse"])
    return out


def fmt(x: Any, nd: int = 3) -> str:
    try:
        if pd.isna(x):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def markdown_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "(empty)"
    text_df = df[cols].copy() if cols else df.copy()
    if max_rows:
        text_df = text_df.head(max_rows)
    for col in text_df.columns:
        if pd.api.types.is_numeric_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: fmt(x) if pd.notna(x) else "nan")
        else:
            text_df[col] = text_df[col].astype(str)
    headers = [str(col) for col in text_df.columns]
    rows = text_df.astype(str).values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def comparison_delta(master: pd.DataFrame, left: str, right: str, station: str | None = None) -> float:
    one7 = master[master["lead"].astype(str) == "1-7"].copy()
    if station is not None:
        one7 = one7[one7["station"].astype(str) == station]
    left_rmse = float(one7[one7["strategy"] == left]["rmse"].mean())
    right_rmse = float(one7[one7["strategy"] == right]["rmse"].mean())
    return left_rmse - right_rmse


def make_report(
    master: pd.DataFrame,
    summary: pd.DataFrame,
    ablation: pd.DataFrame,
    alignment: pd.DataFrame,
    bc_payload: dict[str, Any] | None,
    baseline_available: bool,
    out_dir: Path,
) -> str:
    one7 = master[master["lead"].astype(str) == "1-7"].copy()
    panel = summary[summary["row_type"] == "panel_mean"].set_index("strategy")
    station_summary = summary[summary["row_type"] == "station"].set_index(["station", "strategy"])
    dl_strategies = ["S_single", *U_STRATEGY_ORDER]
    if bc_payload:
        dl_strategies.append(str(bc_payload["corrected_strategy"]))
    best_u = one7[one7["strategy"].isin(U_STRATEGY_ORDER)].groupby("strategy")["rmse"].mean().sort_values().index[0]
    best_dl = one7[one7["strategy"].isin(dl_strategies)].groupby("strategy")["rmse"].mean().sort_values().index[0]

    phys_bits = []
    for method in ["PMTM3", "PMF"]:
        if baseline_available and method in panel.index:
            best_rows = one7[one7["strategy"] == best_dl].set_index("station")
            method_rows = one7[one7["strategy"] == method].set_index("station")
            common = best_rows.index.intersection(method_rows.index)
            count = int((best_rows.loc[common, "rmse"] < method_rows.loc[common, "rmse"]).sum())
            gain = float(method_rows.loc[common, "rmse"].mean() - best_rows.loc[common, "rmse"].mean())
            phys_bits.append(f"{method}: {count}/{len(common)} stations beat, panel gain={gain:+.3f}")
    phys_line = "; ".join(phys_bits) if phys_bits else "Physical baseline not available."

    phys_station = one7[one7["strategy"].isin(["U_hist_nophys", "U_hist_phys"])].pivot(
        index="station", columns="strategy", values="rmse"
    )
    phys_station["hist_phys_gain"] = phys_station["U_hist_nophys"] - phys_station["U_hist_phys"]
    top_phys = phys_station.sort_values("hist_phys_gain", ascending=False).head(10).reset_index()

    s = panel.loc["S_single"]
    u = panel.loc[best_u]
    w_s = station_summary.loc[(WUHAN, "S_single")]
    w_u = station_summary.loc[(WUHAN, best_u)]
    fo_delta_panel = comparison_delta(master, "U_FO_nophys", "U_hist_nophys")
    fo_delta_wuhan = comparison_delta(master, "U_FO_nophys", "U_hist_nophys", WUHAN)
    phys_delta_panel = comparison_delta(master, "U_hist_phys", "U_hist_nophys")
    phys_delta_wuhan = comparison_delta(master, "U_hist_phys", "U_hist_nophys", WUHAN)

    lines = [
        "# Clean SMZ Panel DL Report",
        "",
        "Leakage guard: station screening uses `results/integrity/clean_smz_stations.txt`; alignment check is lead-1 forecast Tmax vs observed Tmax by split; vocabularies, normalizers, anomaly scaling, M3 coefficients, PET_BC AI/aT, and physical-feature fill means are train-only. Validation is used only for early stopping and the optional global bias correction.",
        "",
        "## Alignment Check",
        "",
        f"Passed stations: {int(alignment['pass'].sum())}/{len(alignment)}; minimum split correlation across stations={fmt(alignment['min_corr'].min())}.",
        "",
        "## Panel Mean 1-7",
        "",
        markdown_table(
            summary[summary["row_type"] == "panel_mean"].sort_values("rmse"),
            ["strategy", "station_count", "rmse", "mae", "r2", "nse", "bias", "mean_et0", "nrmse", "rmse_delta_vs_S_single"],
        ),
        "",
        "## U Pooled 2x2",
        "",
        markdown_table(
            ablation[ablation["row_type"] == "panel_mean"].sort_values("strategy"),
            ["strategy", "forecast_only", "phys_feats", "station_count", "rmse", "mae", "r2", "nse", "bias", "delta_vs_U_hist_nophys", "delta_vs_matched_reference"],
        ),
        "",
        "## Physical Feature Station Gains",
        "",
        "Positive gain means `U_hist_phys` beat `U_hist_nophys` by RMSE.",
        "",
        markdown_table(top_phys, ["station", "U_hist_nophys", "U_hist_phys", "hist_phys_gain"]),
        "",
        "## Four Points",
        "",
        (
            f"1. 干净SMZ真实精度: S_single panel 1-7 RMSE/MAE/R2/NSE="
            f"{fmt(s['rmse'])}/{fmt(s['mae'])}/{fmt(s['r2'])}/{fmt(s['nse'])}; "
            f"best U ({best_u})={fmt(u['rmse'])}/{fmt(u['mae'])}/{fmt(u['r2'])}/{fmt(u['nse'])}. "
            f"Wuhan S_single={fmt(w_s['rmse'])}/{fmt(w_s['mae'])}/{fmt(w_s['r2'])}/{fmt(w_s['nse'])}; "
            f"Wuhan best U={fmt(w_u['rmse'])}/{fmt(w_u['mae'])}/{fmt(w_u['r2'])}/{fmt(w_u['nse'])}. "
            f"vs old misaligned 0.914/0.836: S delta={fmt(s['rmse'] - OLD_MISALIGNED_S_SINGLE)}, "
            f"best-U delta={fmt(u['rmse'] - OLD_MISALIGNED_U_POOLED)}."
        ),
        (
            f"2. 仅预报 vs 含历史: U_FO_nophys - U_hist_nophys panel RMSE delta={fmt(fo_delta_panel)}; "
            f"Wuhan delta={fmt(fo_delta_wuhan)}. Positive means removing history worsened RMSE."
        ),
        (
            f"3. 物理Rs/ea特征: U_hist_phys - U_hist_nophys panel delta={fmt(phys_delta_panel)}; "
            f"Wuhan delta={fmt(phys_delta_wuhan)}. Negative means the physical prior helped."
        ),
        f"4. vs物理基线: {phys_line}. Main model by panel RMSE: {best_dl}.",
        "",
    ]
    if bc_payload:
        bc_strategy = str(bc_payload["corrected_strategy"])
        bc = panel.loc[bc_strategy]
        lines.extend(
            [
                "## Validation Global Bias Correction",
                "",
                f"Best U before correction: {bc_payload['strategy']}; validation global delta yhat-y={fmt(bc_payload['global_delta_yhat_minus_y'], 5)}; corrected panel RMSE={fmt(bc['rmse'])}.",
                "",
            ]
        )
    text = "\n".join(lines)
    (out_dir / "smz_clean_report.md").write_text(text + "\n", encoding="utf-8")
    return text


def print_final_four_points(master: pd.DataFrame, summary: pd.DataFrame, bc_payload: dict[str, Any] | None) -> None:
    one7 = master[master["lead"].astype(str) == "1-7"].copy()
    panel = summary[summary["row_type"] == "panel_mean"].set_index("strategy")
    station = summary[summary["row_type"] == "station"].set_index(["station", "strategy"])
    best_u = one7[one7["strategy"].isin(U_STRATEGY_ORDER)].groupby("strategy")["rmse"].mean().sort_values().index[0]
    dl_candidates = ["S_single", *U_STRATEGY_ORDER]
    if bc_payload:
        dl_candidates.append(str(bc_payload["corrected_strategy"]))
    best_dl = one7[one7["strategy"].isin(dl_candidates)].groupby("strategy")["rmse"].mean().sort_values().index[0]

    s = panel.loc["S_single"]
    u = panel.loc[best_u]
    w_s = station.loc[(WUHAN, "S_single")]
    w_u = station.loc[(WUHAN, best_u)]
    fo_delta_panel = comparison_delta(master, "U_FO_nophys", "U_hist_nophys")
    fo_delta_wuhan = comparison_delta(master, "U_FO_nophys", "U_hist_nophys", WUHAN)
    phys_delta_panel = comparison_delta(master, "U_hist_phys", "U_hist_nophys")
    phys_delta_wuhan = comparison_delta(master, "U_hist_phys", "U_hist_nophys", WUHAN)

    print("\n4点结论", flush=True)
    print(
        f"1. 干净SMZ: S_single panel RMSE/MAE/R2/NSE={fmt(s['rmse'])}/{fmt(s['mae'])}/{fmt(s['r2'])}/{fmt(s['nse'])}; "
        f"best U {best_u}={fmt(u['rmse'])}/{fmt(u['mae'])}/{fmt(u['r2'])}/{fmt(u['nse'])}. "
        f"Wuhan S={fmt(w_s['rmse'])}, best U={fmt(w_u['rmse'])}; "
        f"vs old 0.914/0.836 deltas={fmt(s['rmse'] - OLD_MISALIGNED_S_SINGLE)}/{fmt(u['rmse'] - OLD_MISALIGNED_U_POOLED)}.",
        flush=True,
    )
    print(
        f"2. 仅预报 vs 含历史: U_FO_nophys - U_hist_nophys panel={fmt(fo_delta_panel)} mm/d, "
        f"Wuhan={fmt(fo_delta_wuhan)} mm/d.",
        flush=True,
    )
    print(
        f"3. 物理Rs/ea: U_hist_phys - U_hist_nophys panel={fmt(phys_delta_panel)} mm/d, "
        f"Wuhan={fmt(phys_delta_wuhan)} mm/d.",
        flush=True,
    )
    phys_bits = []
    for method in ["PMTM3", "PMF"]:
        if method in one7["strategy"].unique():
            best_rows = one7[one7["strategy"] == best_dl].set_index("station")
            method_rows = one7[one7["strategy"] == method].set_index("station")
            common = best_rows.index.intersection(method_rows.index)
            count = int((best_rows.loc[common, "rmse"] < method_rows.loc[common, "rmse"]).sum())
            gain = float(method_rows.loc[common, "rmse"].mean() - best_rows.loc[common, "rmse"].mean())
            phys_bits.append(f"{method} {count}/{len(common)}, gain={gain:+.3f}")
    print(
        f"4. vs物理基线: {'; '.join(phys_bits) if phys_bits else 'baseline missing'}; 主模型={best_dl} panel RMSE={fmt(panel.loc[best_dl, 'rmse'])}.",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clean-SMZ 71-station panel DL 2x2 ablation.")
    parser.add_argument("--out-dir", default="results/smz_clean")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pooled-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-s-single", action="store_true")
    parser.add_argument("--skip-u", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    set_torch_worker_threads(8)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = DATA_DIR
    seeds = parse_csv_list(args.seeds, int)
    meta = station_meta(ROOT / "data" / "db" / "rice_rl_data.db")

    active_stations, alignment = run_alignment_check(PANEL_STATIONS, data_dir, out_dir)
    station_to_id = {station: i for i, station in enumerate(active_stations)}

    if not args.aggregate_only and not args.skip_training:
        if not args.skip_u:
            if any(cfg["phys_feats"] for cfg in U_STRATEGIES.values()):
                ensure_physical_caches(active_stations, data_dir, meta, out_dir)

        seed_rows: list[dict[str, Any]] = []
        if not args.skip_s_single:
            s_jobs = [
                TrainJob(
                    strategy="S_single",
                    station=station,
                    seed=seed,
                    train_stations=[station],
                    eval_stations=[station],
                    vocab_stations=[station],
                    norm_stations=[station],
                    station_to_id={station: 0},
                    use_station_emb=False,
                    input_size=90,
                    hidden=args.hidden,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    worker_threads=args.worker_threads,
                    out_dir=str(out_dir),
                    forecast_only=False,
                    phys_feats=False,
                    phys_dir=None,
                )
                for station in active_stations
                for seed in seeds
            ]
            seed_rows.extend(run_jobs(s_jobs, args.jobs, "S_single"))

        if not args.skip_u:
            u_jobs: list[TrainJob] = []
            for strategy, flags in U_STRATEGIES.items():
                for seed in seeds:
                    u_jobs.append(
                        TrainJob(
                            strategy=strategy,
                            station="PANEL",
                            seed=seed,
                            train_stations=active_stations,
                            eval_stations=active_stations,
                            vocab_stations=active_stations,
                            norm_stations=active_stations,
                            station_to_id=station_to_id,
                            use_station_emb=True,
                            input_size=90,
                            hidden=args.hidden,
                            epochs=args.epochs,
                            patience=args.patience,
                            batch_size=args.pooled_batch_size,
                            lr=args.lr,
                            worker_threads=args.worker_threads,
                            out_dir=str(out_dir),
                            forecast_only=bool(flags["forecast_only"]),
                            phys_feats=bool(flags["phys_feats"]),
                            phys_dir=str(out_dir) if flags["phys_feats"] else None,
                        )
                    )
            seed_rows.extend(run_jobs(u_jobs, min(args.jobs, len(u_jobs)), "U_pooled_2x2"))

        if seed_rows:
            pd.DataFrame(seed_rows).to_csv(out_dir / "seed_training_summary.csv", index=False)

    all_metric_rows: list[dict[str, Any]] = []
    dl_strategies = ["S_single", *U_STRATEGY_ORDER]
    for strategy in dl_strategies:
        try:
            rows, _ = write_ensemble_and_metrics(out_dir, strategy, active_stations, seeds, split_name="test")
        except FileNotFoundError:
            if args.aggregate_only or args.skip_training:
                raise
            continue
        all_metric_rows.extend(rows)

    baseline_available = True
    try:
        physical = physical_baseline_rows(out_dir / "baseline_pmt", active_stations, data_dir)
        all_metric_rows.extend(physical.to_dict("records"))
    except FileNotFoundError as exc:
        baseline_available = False
        print(f"[baseline] skip physical baseline: {exc}", flush=True)

    master_pre_bc = pd.DataFrame(all_metric_rows)
    master_pre_bc["station"] = master_pre_bc["station"].astype(str)
    master_pre_bc["lead"] = master_pre_bc["lead"].astype(str)
    best_u = (
        master_pre_bc[
            (master_pre_bc["lead"].astype(str) == "1-7")
            & (master_pre_bc["strategy"].isin(U_STRATEGY_ORDER))
        ]
        .groupby("strategy")["rmse"]
        .mean()
        .sort_values()
        .index[0]
    )
    bc_rows, bc_payload = apply_global_bias_correction(out_dir, best_u, active_stations, seeds)
    all_metric_rows.extend(bc_rows)

    master = pd.DataFrame(all_metric_rows)
    master["station"] = master["station"].astype(str)
    master["lead"] = master["lead"].astype(str)
    master["_lead_sort"] = master["lead"].map(lead_sort)
    master = master.sort_values(["station", "strategy", "_lead_sort"]).drop(columns="_lead_sort").reset_index(drop=True)
    master.to_csv(out_dir / "master_metrics.csv", index=False)

    dl_strategies_with_bc = ["S_single", *U_STRATEGY_ORDER, str(bc_payload["corrected_strategy"])]
    summary = build_panel_summary(master, dl_strategies_with_bc)
    summary.to_csv(out_dir / "panel_summary.csv", index=False)
    ablation = build_ablation_2x2(master)
    ablation.to_csv(out_dir / "ablation_2x2.csv", index=False)
    make_report(master, summary, ablation, alignment, bc_payload, baseline_available, out_dir)
    print_final_four_points(master, summary, bc_payload)
    print(f"\nWrote outputs under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
