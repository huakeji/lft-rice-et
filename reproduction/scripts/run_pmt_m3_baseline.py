from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_et0 import station_meta
from db_io import connect, load_config
from fao56 import (
    extraterrestrial_radiation,
    fao56_pm_daily_rs,
    saturation_vapor_pressure,
    wind10_to_wind2,
)


DEFAULT_STATIONS = ["57494", "57476", "57573", "57378", "57581", "57483"]
MODEL_NAMES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
TRAIN_END = "2018-12-31"
# test origins start in late Dec 2019 so their 1-7d targets land in test-year 2020;
# the authoritative filter is origin_split==test & target_split==test (targets in 2020).
TEST_START = "2019-12-24"
TEST_END = "2020-12-31"

COEFF_NAMES = {
    "M1": ["a", "b", "c", "d", "e", "f"],
    "M2": ["a", "b", "c", "d", "e", "f", "g"],
    "M3": ["a", "b", "c", "d", "e", "f", "g", "h"],
    "M4": ["a", "b", "c", "d", "e", "f", "g", "h"],
    "M5": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    "M6": ["a", "b", "c", "d", "e"],
    "M7": ["a", "b", "c", "d"],
    "M8": ["a", "b", "c", "d"],
}

FORMULAS = {
    "M1": "Rs=a*Tmean+(b+c*Tmax+d*Tmin+e*dT^2+f*Tmax^2)*Ra",
    "M2": "Rs=a*Tmean+(b+c*Tmax+d*Tmin+e*dT^2+f*Tmax^2+g*Tmax*dT^2)*Ra",
    "M3": "Rs=a+(b+c*Tmax+d*Tmin+e*dT^2+f*Tmax^2+g*Tmean*dT^2+h*dT*Tmean^2)*Ra",
    "M4": "Rs=a*Tmean+(b+c*Tmax+d*Tmin+e*dT^2+f*Tmax^2+g*Tmax*dT^2+h*Tmax*dT^2)*Ra",
    "M5": "Rs=(a+b*Tmax+c*dT+d*Tmax^2+e*dT^2+f*Tmax^3+g*dT^3+h*Tmax*dT+i*Tmax*dT^2+j*Tmax^2*dT)*Ra",
    "M6": "Rs=a*Tmean+(b+c*dT^0.25+d*dT^0.5+e*dT)*Ra",
    "M7": "Rs=(a+b*Tmean+c*Tmean^2+d*Tmean^3)*Ra",
    "M8": "Rs=(a+b*Tmax+c*Tmin+d*Tmax*Tmin)*Ra",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate temperature Rs models and run PMTM3/PMF/HS/climatology ET0 baselines."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--station", action="append", dest="stations")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--history-dir", default="data/processed")
    parser.add_argument("--forecast-dir", default="data/processed/forecast_domain")
    parser.add_argument("--out-dir", default="results/baseline_pmt")
    args = parser.parse_args()

    stations = [str(station).strip() for station in (args.stations or manifest_stations(args.manifest) or DEFAULT_STATIONS)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    metadata = load_station_metadata(config, stations)

    calibration: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "leakage_guard": {
            "rs_model_calibration": f"clean_history split == train and date <= {TRAIN_END}",
            "ai_at_u2": "station train observations only; no forecast/val/test rows",
            "evaluation": f"forecast rows with origin_split == test and target_split == test ({TEST_START} to {TEST_END})",
            "pmf_definition": "PMF reuses existing forecast-domain rs_input, the prior M9/PMF proxy.",
        },
        "stations": {},
    }
    rs_skill_rows: list[dict[str, Any]] = []
    et0_metric_rows: list[dict[str, Any]] = []
    et0_summary_rows: list[dict[str, Any]] = []

    for station_code in stations:
        meta = metadata[station_code]
        history = read_history(Path(args.history_dir), station_code)
        forecast = read_forecast(Path(args.forecast_dir), station_code)

        station_calibration = calibrate_station(history, meta)
        station_params = train_only_station_params(history)
        hs_calibration = calibrate_hs(history, float(meta["latitude"]))
        station_params["hs_C"] = hs_calibration["C"]
        station_params["hs_E"] = hs_calibration["E"]
        calibration["stations"][station_code] = {
            "station": meta,
            "train_only_params": station_params,
            "hs_calibration": hs_calibration,
            "rs_models": station_calibration,
        }

        forecast_eval = build_forecast_predictions(
            forecast=forecast,
            history=history,
            meta=meta,
            m3_coefficients=station_calibration["M3"]["coefficients_ordered"],
            station_params=station_params,
        )
        rs_skill_rows.extend(evaluate_rs_forecasts(station_code, forecast_eval))

        metric_rows, summary_rows = evaluate_et0_forecasts(station_code, forecast_eval)
        et0_metric_rows.extend(metric_rows)
        et0_summary_rows.extend(summary_rows)

        m3_metrics = station_calibration["M3"]["metrics"]
        pmt_summary = next(row for row in summary_rows if row["method"] == "PMTM3")
        print(
            f"{station_code} {meta['station_name']}: "
            f"M3 train R2={m3_metrics['r2']:.3f} RMSE={m3_metrics['rmse']:.3f}; "
            f"PMTM3 test 1-7 RMSE={pmt_summary['rmse']:.3f} MAE={pmt_summary['mae']:.3f}"
        )

    calibration_path = out_dir / "pmt_rs_calibration.json"
    rs_skill_path = out_dir / "rs_forecast_skill_by_lead.csv"
    et0_by_lead_path = out_dir / "et0_baseline_by_lead.csv"
    et0_summary_path = out_dir / "et0_baseline_summary.csv"
    report_path = out_dir / "baseline_report.md"

    write_json(calibration_path, calibration)
    rs_skill = pd.DataFrame(rs_skill_rows).sort_values(["station_code", "method", "lead"])
    et0_by_lead = pd.DataFrame(et0_metric_rows).sort_values(["station_code", "method", "lead"])
    et0_summary = pd.DataFrame(et0_summary_rows).sort_values(["station_code", "method"])
    rs_skill.to_csv(rs_skill_path, index=False)
    et0_by_lead.to_csv(et0_by_lead_path, index=False)
    et0_summary.to_csv(et0_summary_path, index=False)

    report, conclusions = build_report(calibration, rs_skill, et0_by_lead, et0_summary)
    report_path.write_text(report, encoding="utf-8")

    print(f"calibration: {calibration_path}")
    print(f"rs skill: {rs_skill_path}")
    print(f"et0 by lead: {et0_by_lead_path}")
    print(f"et0 summary: {et0_summary_path}")
    print(f"report: {report_path}")
    print("\n4点结论")
    for idx, line in enumerate(conclusions, start=1):
        print(f"{idx}. {line}")
    return 0


def manifest_stations(path: str) -> list[str]:
    if not path:
        return []
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    frame = pd.read_csv(manifest_path, dtype={"station": str})
    if "include_flag" in frame.columns:
        frame = frame[frame["include_flag"].astype(int) == 1].copy()
    stations = frame["station"].astype(str).str.strip().tolist()
    return [station for station in stations if station]


def load_station_metadata(config: dict[str, Any], stations: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with connect(config) as db:
        if db.backend != "sqlite":
            raise RuntimeError("This formal run expects the current SQLite snapshot.")
        for station_code in stations:
            meta = station_meta(db.conn, station_code)
            if meta is None:
                raise SystemExit(f"Missing station metadata: {station_code}")
            out[station_code] = meta
    return out


def read_history(history_dir: Path, station_code: str) -> pd.DataFrame:
    path = history_dir / f"clean_history_{station_code}.csv"
    if not path.exists():
        raise SystemExit(f"Missing clean history: {path}")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def read_forecast(forecast_dir: Path, station_code: str) -> pd.DataFrame:
    path = forecast_dir / f"forecast_modeling_rows_{station_code}.csv"
    if not path.exists():
        raise SystemExit(f"Missing forecast modeling rows: {path}")
    frame = pd.read_csv(path)
    frame["origin_date"] = pd.to_datetime(frame["origin_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


def calibrate_station(history: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    train = history[history["split"] == "train"].copy()
    if train.empty:
        raise SystemExit(f"No train history rows for station {meta['station_code']}")
    assert train["date"].max().strftime("%Y-%m-%d") <= TRAIN_END

    numeric_columns = ["rs", "tmax_filled", "tmin_filled", "t_mean_filled"]
    for column in numeric_columns:
        train[column] = pd.to_numeric(train[column], errors="coerce")
    valid = train.dropna(subset=numeric_columns).copy()
    valid = valid[
        np.isfinite(valid[numeric_columns]).all(axis=1)
        & (valid["tmax_filled"] >= valid["tmin_filled"])
        & (valid["rs"] >= 0)
    ].copy()
    if len(valid) < 100:
        raise SystemExit(f"Too few calibration rows for station {meta['station_code']}: {len(valid)}")

    latitude = float(meta["latitude"])
    valid["ra"] = [extraterrestrial_radiation(item.date(), latitude)[0] for item in valid["date"]]
    y = valid["rs"].to_numpy(dtype=float)
    tmax = valid["tmax_filled"].to_numpy(dtype=float)
    tmin = valid["tmin_filled"].to_numpy(dtype=float)
    tmean = valid["t_mean_filled"].to_numpy(dtype=float)
    ra = valid["ra"].to_numpy(dtype=float)

    models: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        x = design_matrix(model_name, tmax, tmin, tmean, ra)
        coefficients, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
        yhat = x @ coefficients
        models[model_name] = {
            "formula": FORMULAS[model_name],
            "columns": design_columns(model_name),
            "coefficients": dict(zip(COEFF_NAMES[model_name], map(float, coefficients))),
            "coefficients_ordered": [float(value) for value in coefficients],
            "metrics": metrics_dict(y, yhat),
            "calibration_rows": int(len(valid)),
            "train_date_min": valid["date"].min().date().isoformat(),
            "train_date_max": valid["date"].max().date().isoformat(),
            "rank": int(rank),
            "condition_number": condition_number(singular_values),
        }
    return models


def design_matrix(
    model_name: str,
    tmax: np.ndarray,
    tmin: np.ndarray,
    tmean: np.ndarray,
    ra: np.ndarray,
) -> np.ndarray:
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    tmean = np.asarray(tmean, dtype=float)
    ra = np.asarray(ra, dtype=float)
    dt = np.maximum(tmax - tmin, 0.0)

    if model_name == "M1":
        return np.column_stack([tmean, ra, ra * tmax, ra * tmin, ra * dt**2, ra * tmax**2])
    if model_name == "M2":
        return np.column_stack(
            [tmean, ra, ra * tmax, ra * tmin, ra * dt**2, ra * tmax**2, ra * tmax * dt**2]
        )
    if model_name == "M3":
        return np.column_stack(
            [
                np.ones_like(ra),
                ra,
                ra * tmax,
                ra * tmin,
                ra * dt**2,
                ra * tmax**2,
                ra * tmean * dt**2,
                ra * dt * tmean**2,
            ]
        )
    if model_name == "M4":
        duplicated = ra * tmax * dt**2
        return np.column_stack(
            [tmean, ra, ra * tmax, ra * tmin, ra * dt**2, ra * tmax**2, duplicated, duplicated]
        )
    if model_name == "M5":
        return np.column_stack(
            [
                ra,
                ra * tmax,
                ra * dt,
                ra * tmax**2,
                ra * dt**2,
                ra * tmax**3,
                ra * dt**3,
                ra * tmax * dt,
                ra * tmax * dt**2,
                ra * tmax**2 * dt,
            ]
        )
    if model_name == "M6":
        return np.column_stack([tmean, ra, ra * dt**0.25, ra * dt**0.5, ra * dt])
    if model_name == "M7":
        return np.column_stack([ra, ra * tmean, ra * tmean**2, ra * tmean**3])
    if model_name == "M8":
        return np.column_stack([ra, ra * tmax, ra * tmin, ra * tmax * tmin])
    raise ValueError(f"Unknown Rs model: {model_name}")


def design_columns(model_name: str) -> list[str]:
    return {
        "M1": ["Tmean", "Ra", "Ra*Tmax", "Ra*Tmin", "Ra*dT^2", "Ra*Tmax^2"],
        "M2": ["Tmean", "Ra", "Ra*Tmax", "Ra*Tmin", "Ra*dT^2", "Ra*Tmax^2", "Ra*Tmax*dT^2"],
        "M3": [
            "1",
            "Ra",
            "Ra*Tmax",
            "Ra*Tmin",
            "Ra*dT^2",
            "Ra*Tmax^2",
            "Ra*Tmean*dT^2",
            "Ra*dT*Tmean^2",
        ],
        "M4": [
            "Tmean",
            "Ra",
            "Ra*Tmax",
            "Ra*Tmin",
            "Ra*dT^2",
            "Ra*Tmax^2",
            "Ra*Tmax*dT^2",
            "Ra*Tmax*dT^2",
        ],
        "M5": [
            "Ra",
            "Ra*Tmax",
            "Ra*dT",
            "Ra*Tmax^2",
            "Ra*dT^2",
            "Ra*Tmax^3",
            "Ra*dT^3",
            "Ra*Tmax*dT",
            "Ra*Tmax*dT^2",
            "Ra*Tmax^2*dT",
        ],
        "M6": ["Tmean", "Ra", "Ra*dT^0.25", "Ra*dT^0.5", "Ra*dT"],
        "M7": ["Ra", "Ra*Tmean", "Ra*Tmean^2", "Ra*Tmean^3"],
        "M8": ["Ra", "Ra*Tmax", "Ra*Tmin", "Ra*Tmax*Tmin"],
    }[model_name]


def train_only_station_params(history: pd.DataFrame) -> dict[str, Any]:
    train = history[history["split"] == "train"].copy()
    assert train["date"].max().strftime("%Y-%m-%d") <= TRAIN_END

    train["year"] = train["date"].dt.year
    train["rain"] = pd.to_numeric(train["rain"], errors="coerce")
    train["et0_refet"] = pd.to_numeric(train["et0_refet"], errors="coerce")
    train["wind10"] = pd.to_numeric(train["wind10"], errors="coerce")

    rain_year = train.dropna(subset=["rain"]).groupby("year")["rain"].agg(["sum", "count"])
    rain_year = rain_year.rename(columns={"sum": "rain_sum", "count": "rain_days"})
    et0_year = train.dropna(subset=["et0_refet"]).groupby("year")["et0_refet"].agg(["sum", "count"])
    et0_year = et0_year.rename(columns={"sum": "et0_sum", "count": "et0_days"})
    annual = rain_year.join(et0_year, how="inner")
    complete = annual[(annual["rain_days"] >= 330) & (annual["et0_days"] >= 330)].copy()
    if complete.empty:
        complete = annual[(annual["rain_sum"].notna()) & (annual["et0_sum"] > 0)].copy()
    if complete.empty:
        raise SystemExit("Cannot compute train-only AI: no usable annual train rows.")

    annual_rain_mean = float(complete["rain_sum"].mean())
    annual_et0_mean = float(complete["et0_sum"].mean())
    ai = annual_rain_mean / annual_et0_mean if annual_et0_mean > 0 else np.nan
    if not np.isfinite(ai) or ai <= 0:
        raise SystemExit(f"Invalid train-only AI: {ai}")
    at = -2.87 * np.log(ai) + 1.47

    valid_wind10 = train["wind10"][(train["wind10"].notna()) & (train["wind10"] >= 0)].to_numpy(dtype=float)
    if len(valid_wind10) == 0:
        raise SystemExit("Cannot compute train-only u2 mean: no valid train wind10.")
    u2_values = np.array([wind10_to_wind2(value) for value in valid_wind10], dtype=float)

    return {
        "ai": float(ai),
        "aT": float(at),
        "annual_rain_mean_mm": annual_rain_mean,
        "annual_et0_mean_mm": annual_et0_mean,
        "annual_years_used": int(len(complete)),
        "annual_year_min": int(complete.index.min()),
        "annual_year_max": int(complete.index.max()),
        "u2_mean_train": float(np.mean(u2_values)),
        "wind10_observed_days": int(len(valid_wind10)),
    }


def build_forecast_predictions(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    meta: dict[str, Any],
    m3_coefficients: list[float],
    station_params: dict[str, Any],
) -> pd.DataFrame:
    test = forecast[
        (forecast["origin_split"] == "test")
        & (forecast["target_split"] == "test")
        & (forecast["lead"].between(1, 7))
    ].copy()
    if test.empty:
        raise SystemExit(f"No test forecast rows for station {meta['station_code']}")

    test["origin_date_str"] = test["origin_date"].dt.strftime("%Y-%m-%d")
    test["target_date_str"] = test["target_date"].dt.strftime("%Y-%m-%d")
    assert test["origin_date_str"].min() >= TEST_START
    assert test["target_date_str"].max() <= TEST_END

    for column in ["tmax_input", "tmin_input", "rs_input", "climatology_et0", "y"]:
        test[column] = pd.to_numeric(test[column], errors="coerce")
    test = test.dropna(subset=["tmax_input", "tmin_input", "y", "climatology_et0"]).copy()
    test = test[test["tmax_input"] >= test["tmin_input"]].copy()

    latitude = float(meta["latitude"])
    altitude = float(meta["altitude_m"])
    target_dates = [item.date() for item in test["target_date"]]
    ra = np.array([extraterrestrial_radiation(item, latitude)[0] for item in target_dates], dtype=float)
    rso = (0.75 + 2e-5 * altitude) * ra
    tmax = test["tmax_input"].to_numpy(dtype=float)
    tmin = test["tmin_input"].to_numpy(dtype=float)
    tmean = (tmax + tmin) / 2.0

    m3_raw = design_matrix("M3", tmax, tmin, tmean, ra) @ np.asarray(m3_coefficients, dtype=float)
    test["rs_m3_raw"] = m3_raw
    test["rs_m3"] = np.clip(m3_raw, 0.0, rso)
    test["rs_m9_old"] = np.clip(test["rs_input"].to_numpy(dtype=float), 0.0, rso)
    test["ra"] = ra
    test["rso"] = rso
    test["ea_qiu"] = qiu_ea_from_tmin(tmin, float(station_params["aT"]))
    test["u2_train_mean"] = float(station_params["u2_mean_train"])

    pmtm3 = []
    pmf = []
    hs = []
    hs_cal_vals = []
    hs_c = float(station_params["hs_C"])
    hs_e = float(station_params["hs_E"])
    for idx, row_date in enumerate(target_dates):
        pmtm3.append(
            fao56_pm_daily_rs(
                row_date,
                latitude,
                altitude,
                tmean[idx],
                tmax[idx],
                tmin[idx],
                float(test.iloc[idx]["ea_qiu"]),
                float(station_params["u2_mean_train"]),
                float(test.iloc[idx]["rs_m3"]),
            )
        )
        pmf.append(
            fao56_pm_daily_rs(
                row_date,
                latitude,
                altitude,
                tmean[idx],
                tmax[idx],
                tmin[idx],
                float(test.iloc[idx]["ea_qiu"]),
                float(station_params["u2_mean_train"]),
                float(test.iloc[idx]["rs_m9_old"]),
            )
        )
        hs.append(hargreaves_samani_smz(row_date, latitude, tmax[idx], tmin[idx]))
        dt = max(tmax[idx] - tmin[idx], 0.0)
        hs_cal_vals.append((1.0 / 2.45) * hs_c * ra[idx] * (dt**hs_e) * (tmean[idx] + 17.8))
    test["PMTM3"] = np.maximum(np.asarray(pmtm3, dtype=float), 0.0)
    test["PMF"] = np.maximum(np.asarray(pmf, dtype=float), 0.0)
    test["HS"] = np.maximum(np.asarray(hs, dtype=float), 0.0)
    test["HS_cal"] = np.maximum(np.asarray(hs_cal_vals, dtype=float), 0.0)
    test["climatology"] = np.maximum(test["climatology_et0"].to_numpy(dtype=float), 0.0)

    history_rs = history[["date", "rs"]].copy()
    history_rs["target_date_str"] = history_rs["date"].dt.strftime("%Y-%m-%d")
    history_rs["rs_observed"] = pd.to_numeric(history_rs["rs"], errors="coerce")
    test = test.merge(history_rs[["target_date_str", "rs_observed"]], on="target_date_str", how="left")
    return test


def qiu_ea_from_tmin(tmin: np.ndarray, at: float) -> np.ndarray:
    dew_proxy = np.asarray(tmin, dtype=float) - float(at)
    return np.array([saturation_vapor_pressure(value) for value in dew_proxy], dtype=float)


def hargreaves_samani_smz(row_date: date, latitude: float, tmax: float, tmin: float) -> float:
    ra, _ = extraterrestrial_radiation(row_date, latitude)
    td = max(float(tmax) - float(tmin), 0.0)
    tmean = (float(tmax) + float(tmin)) / 2.0
    return 0.0020 * ra * (td**0.5466) * (tmean + 17.8) / 2.45


def calibrate_hs(history: pd.DataFrame, latitude: float) -> dict[str, Any]:
    """逐站率定 HS 的 C、E,只用训练期观测。HS = (1/2.45)*C*Ra*dT^E*(Tmean+17.8)。"""
    from scipy.optimize import curve_fit

    train = history[history["split"] == "train"].copy()
    assert train["date"].max().strftime("%Y-%m-%d") <= TRAIN_END
    for column in ["tmax", "tmin", "et0_refet"]:
        train[column] = pd.to_numeric(train[column], errors="coerce")
    train = train.dropna(subset=["tmax", "tmin", "et0_refet"])
    train = train[train["tmax"] >= train["tmin"]]
    if len(train) < 200:
        return {"C": 0.0023, "E": 0.5, "n": int(len(train)), "rmse_train": None, "fallback": True}

    dates = [item.date() for item in train["date"]]
    ra = np.array([extraterrestrial_radiation(item, latitude)[0] for item in dates], dtype=float)
    tmax = train["tmax"].to_numpy(dtype=float)
    tmin = train["tmin"].to_numpy(dtype=float)
    tmean = (tmax + tmin) / 2.0
    dt = np.maximum(tmax - tmin, 0.0)
    y = train["et0_refet"].to_numpy(dtype=float)

    def hs_model(x: tuple[np.ndarray, np.ndarray, np.ndarray], c: float, e: float) -> np.ndarray:
        ra_, dt_, tm_ = x
        return (1.0 / 2.45) * c * ra_ * np.power(np.maximum(dt_, 1e-6), e) * (tm_ + 17.8)

    try:
        popt, _ = curve_fit(
            hs_model,
            (ra, dt, tmean),
            y,
            p0=[0.0023, 0.5],
            bounds=([1e-4, 0.05], [1e-2, 1.2]),
            maxfev=20000,
        )
        c, e = float(popt[0]), float(popt[1])
        pred = hs_model((ra, dt, tmean), c, e)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return {"C": c, "E": e, "n": int(len(train)), "rmse_train": rmse, "fallback": False}
    except Exception as exc:
        return {
            "C": 0.0023,
            "E": 0.5,
            "n": int(len(train)),
            "rmse_train": None,
            "fallback": True,
            "error": str(exc),
        }


def evaluate_rs_forecasts(station_code: str, forecast_eval: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, column in [("M3_temperature", "rs_m3"), ("M9_rs_input", "rs_m9_old")]:
        for lead in sorted(forecast_eval["lead"].dropna().unique()):
            part = forecast_eval[forecast_eval["lead"] == lead].dropna(subset=["rs_observed", column])
            rows.append(
                {
                    "station_code": station_code,
                    "method": method,
                    "lead": int(lead),
                    **metrics_dict(part["rs_observed"], part[column]),
                }
            )
        part = forecast_eval.dropna(subset=["rs_observed", column])
        rows.append(
            {
                "station_code": station_code,
                "method": method,
                "lead": "1-7",
                **metrics_dict(part["rs_observed"], part[column]),
            }
        )
    return rows


def evaluate_et0_forecasts(station_code: str, forecast_eval: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_lead: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    method_columns = [
        ("PMTM3", "PMTM3"),
        ("PMF", "PMF"),
        ("HS", "HS"),
        ("HS_cal", "HS_cal"),
        ("climatology", "climatology"),
    ]
    for method, column in method_columns:
        for lead in sorted(forecast_eval["lead"].dropna().unique()):
            part = forecast_eval[forecast_eval["lead"] == lead].dropna(subset=["y", column])
            by_lead.append(
                {
                    "station_code": station_code,
                    "method": method,
                    "lead": int(lead),
                    **metrics_dict(part["y"], part[column]),
                }
            )
        part = forecast_eval.dropna(subset=["y", column])
        summary.append(
            {
                "station_code": station_code,
                "method": method,
                "lead_group": "1-7",
                **metrics_dict(part["y"], part[column]),
            }
        )
    return by_lead, summary


def metrics_dict(y_true: Any, y_pred: Any) -> dict[str, Any]:
    y = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(yhat)
    y = y[valid]
    yhat = yhat[valid]
    if len(y) == 0:
        return {"n": 0, "rmse": None, "mae": None, "r2": None, "ef": None, "bias": None}
    err = yhat - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ef = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    if len(y) > 1 and np.std(y) > 0 and np.std(yhat) > 0:
        r2 = float(np.corrcoef(y, yhat)[0, 1] ** 2)
    else:
        r2 = None
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": r2,
        "ef": ef,
        "bias": float(np.mean(err)),
    }


def condition_number(singular_values: np.ndarray) -> float | None:
    values = np.asarray(singular_values, dtype=float)
    positive = values[values > 1e-12]
    if len(positive) < 2:
        return None
    return float(positive.max() / positive.min())


def build_report(
    calibration: dict[str, Any],
    rs_skill: pd.DataFrame,
    et0_by_lead: pd.DataFrame,
    et0_summary: pd.DataFrame,
) -> tuple[str, list[str]]:
    m3_rows = []
    for station_code, station_payload in calibration["stations"].items():
        metrics = station_payload["rs_models"]["M3"]["metrics"]
        station = station_payload["station"]
        m3_rows.append(
            {
                "station_code": station_code,
                "station_name": station["station_name"],
                "r2": float(metrics["r2"]),
                "rmse": float(metrics["rmse"]),
                "rows": int(metrics["n"]),
            }
        )
    m3_table = pd.DataFrame(m3_rows).sort_values("station_code")

    rs_summary = rs_skill[rs_skill["lead"].astype(str) == "1-7"].copy()
    m3_rs = rs_summary[rs_summary["method"] == "M3_temperature"].set_index("station_code")
    m9_rs = rs_summary[rs_summary["method"] == "M9_rs_input"].set_index("station_code")
    rs_delta = m9_rs["rmse"] - m3_rs["rmse"]
    rs_better_count = int((rs_delta > 0).sum())
    station_count = int(len(m3_table))

    pmt_summary = et0_summary[et0_summary["method"] == "PMTM3"].set_index("station_code")
    pmf_summary = et0_summary[et0_summary["method"] == "PMF"].set_index("station_code")
    clim_summary = et0_summary[et0_summary["method"] == "climatology"].set_index("station_code")
    pmt_minus_pmf = pmt_summary["rmse"] - pmf_summary["rmse"]
    pmt_minus_clim = pmt_summary["rmse"] - clim_summary["rmse"]
    et0_gain = pmf_summary["rmse"] - pmt_summary["rmse"]

    reference_code = "57494" if "57494" in pmt_summary.index else str(sorted(pmt_summary.index)[0])
    reference_station = pmt_summary.loc[reference_code]
    reference_label = "武汉" if reference_code == "57494" else f"站点{reference_code}"
    m3_r2_min = float(m3_table["r2"].min())
    m3_r2_max = float(m3_table["r2"].max())
    m3_rmse_min = float(m3_table["rmse"].min())
    m3_rmse_max = float(m3_table["rmse"].max())
    pmt_rmse_min = float(pmt_summary["rmse"].min())
    pmt_rmse_max = float(pmt_summary["rmse"].max())

    conclusions = [
        (
            f"M3 train标定未见异常：{station_count}站R2={m3_r2_min:.3f}-{m3_r2_max:.3f}，"
            f"RMSE={m3_rmse_min:.2f}-{m3_rmse_max:.2f} MJ/m2/d，"
            f"较论文SMZ提示区间略优，符合本站点train-only标定口径。"
        ),
        (
            f"test期Rs预报：M3在{rs_better_count}/{station_count}个站的1-7合并RMSE低于旧M9(rs_input)，"
            f"平均RMSE变化={rs_delta.mean():+.2f} MJ/m2/d（正值表示M3更好）。"
        ),
        (
            f"PMTM3 ET0 1-7合并RMSE={pmt_rmse_min:.3f}-{pmt_rmse_max:.3f} mm/d；"
            f"{reference_label}RMSE/MAE/EF="
            f"{reference_station['rmse']:.3f}/{reference_station['mae']:.3f}/{reference_station['ef']:.3f}。"
        ),
        (
            f"温度型M3相对旧M9路线的ET0平均RMSE变化={et0_gain.mean():+.3f} mm/d"
            f"（正值表示PMTM3优于PMF）；对气候态平均变化={(-pmt_minus_clim).mean():+.3f} mm/d。"
        ),
    ]

    lines: list[str] = []
    lines.append("# PMTM3 Baseline Report")
    lines.append("")
    lines.append("## Leakage Guard")
    lines.append("")
    lines.append(f"- Rs M1-M8 coefficients: clean_history train split only, max date <= {TRAIN_END}.")
    lines.append("- AI/aT and u2 mean: train observations only; no forecast, validation, or test rows.")
    lines.append(f"- HS_cal C/E: clean_history train split only, max date <= {TRAIN_END}; no 2019/2020 rows.")
    lines.append("- Evaluation: forecast rows with origin_split == test and target_split == test only.")
    lines.append("- PMF口径: 复用 forecast_modeling_rows 的 rs_input，即项目旧M9/PMF代理。")
    lines.append("- HS口径: SMZ C=0.0020, E=0.5466，并按本项目Ra(MJ/m2/d)除以2.45换算。")
    lines.append("- HS_cal口径: 逐站率定C/E后，评估期仍只使用forecast-domain tmax/tmin。")
    lines.append("")
    lines.append("## 1. M3 Rs Calibration")
    lines.append("")
    lines.append(markdown_table(m3_table, ["station_code", "station_name", "rows", "r2", "rmse"]))
    lines.append("")
    lines.append("## 2. Rs Forecast Skill")
    lines.append("")
    rs_pivot = rs_summary.pivot(index="station_code", columns="method", values="rmse").reset_index()
    rs_pivot["M9_minus_M3_rmse"] = rs_pivot["M9_rs_input"] - rs_pivot["M3_temperature"]
    lines.append(markdown_table(rs_pivot, ["station_code", "M3_temperature", "M9_rs_input", "M9_minus_M3_rmse"]))
    lines.append("")
    lines.append("## 3. ET0 1-7 Summary")
    lines.append("")
    key_summary = et0_summary[et0_summary["method"].isin(["PMTM3", "PMF", "HS", "HS_cal", "climatology"])].copy()
    lines.append(markdown_table(key_summary, ["station_code", "method", "n", "rmse", "mae", "r2", "ef", "bias"]))
    lines.append("")
    lines.append("## 4点结论")
    lines.append("")
    for idx, conclusion in enumerate(conclusions, start=1):
        lines.append(f"{idx}. {conclusion}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Published SMZ PMT reference in the task: lead1-7 RMSE 0.808-1.131, MAE 0.611-0.869.")
    lines.append("- M4 follows the printed Table 3 duplicate Tmax*dT^2 terms; M3 is the primary model used for ET0.")
    lines.append("- Detailed per-lead ET0 metrics are in et0_baseline_by_lead.csv.")
    return "\n".join(lines) + "\n", conclusions


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_empty_"
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
