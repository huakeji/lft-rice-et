from __future__ import annotations

from datetime import date
from math import acos, cos, exp, log, pi, pow, sin, sqrt, tan


def latitude_to_radians(latitude: float) -> float:
    return float(latitude) * pi / 180.0


def wind10_to_wind2(wind10: float) -> float:
    return float(wind10) * 4.87 / log(67.8 * 10.0 - 5.42)


def saturation_vapor_pressure(temp_c: float) -> float:
    if temp_c <= -237.3:
        return 0.0
    return 0.6108 * exp(17.27 * temp_c / (temp_c + 237.3))


def extraterrestrial_radiation(row_date: date, latitude: float) -> tuple[float, float]:
    doy = row_date.timetuple().tm_yday
    lat_rad = latitude_to_radians(latitude)
    delta = 0.409 * sin(2.0 * pi * doy / 365.0 - 1.39)
    ws = acos(-tan(lat_rad) * tan(delta))
    daylight_hours = 24.0 * ws / pi
    dr = 1.0 + 0.033 * cos(2.0 * pi * doy / 365.0)
    ra = (24.0 * 60.0 / pi) * 0.0820 * dr * (
        ws * sin(lat_rad) * sin(delta) + cos(lat_rad) * cos(delta) * sin(ws)
    )
    return ra, daylight_hours


def sunshine_to_rs(row_date: date, latitude: float, sunshine_hours: float) -> float:
    ra, daylight_hours = extraterrestrial_radiation(row_date, latitude)
    ratio = 0.0 if daylight_hours == 0 else max(0.0, min(float(sunshine_hours) / daylight_hours, 1.0))
    return (0.25 + 0.50 * ratio) * ra


def hargreaves_samani(row_date: date, latitude: float, tmax: float, tmin: float) -> float:
    ra, _ = extraterrestrial_radiation(row_date, latitude)
    tmean = (float(tmax) + float(tmin)) / 2.0
    td = max(float(tmax) - float(tmin), 0.0)
    return 0.0023 * ra * (tmean + 17.8) * sqrt(td) / 2.45


def fao56_pm_daily(
    row_date: date,
    latitude: float,
    altitude_m: float,
    tmean: float,
    tmax: float,
    tmin: float,
    rh_mean: float,
    wind10: float,
    sunshine_hours: float,
) -> float:
    # Local fallback used for validation if refet is unavailable.
    doy = row_date.timetuple().tm_yday
    lat_rad = latitude_to_radians(latitude)
    ea_tmax = saturation_vapor_pressure(tmax)
    ea_tmin = saturation_vapor_pressure(tmin)
    es = (ea_tmax + ea_tmin) / 2.0
    ea = es * float(rh_mean) / 100.0
    ra, daylight_hours = extraterrestrial_radiation(row_date, latitude)
    ratio = 0.0 if daylight_hours == 0 else max(0.0, min(float(sunshine_hours) / daylight_hours, 1.0))
    rs = (0.25 + 0.50 * ratio) * ra
    rns = (1.0 - 0.23) * rs
    rso = (0.75 + 2e-5 * float(altitude_m)) * ra
    rnl = 4.903e-9 * (pow(float(tmax) + 273.16, 4) + pow(float(tmin) + 273.16, 4)) / 2.0
    rnl *= (0.34 - 0.14 * sqrt(max(ea, 0.0))) * (1.35 * _safe_div(rs, rso) - 0.35)
    rn = rns - rnl

    pressure = 101.3 * pow((293.0 - 0.0065 * float(altitude_m)) / 293.0, 5.26)
    gamma = 0.000665 * pressure
    slope = 4098.0 * saturation_vapor_pressure(tmean) / pow(float(tmean) + 237.3, 2)
    u2 = wind10_to_wind2(wind10)
    denominator = slope + gamma * (1.0 + 0.34 * u2)
    if denominator == 0:
        return 0.0
    return (
        0.408 * slope * rn
        + gamma * (900.0 / (float(tmean) + 273.0)) * u2 * (es - ea)
    ) / denominator


def fao56_pm_daily_rs(
    row_date: date,
    latitude: float,
    altitude_m: float,
    tmean: float,
    tmax: float,
    tmin: float,
    ea: float,
    u2: float,
    rs: float,
) -> float:
    """FAO-56 daily PM with externally supplied Rs, ea, and 2 m wind speed."""
    ea_tmax = saturation_vapor_pressure(tmax)
    ea_tmin = saturation_vapor_pressure(tmin)
    es = (ea_tmax + ea_tmin) / 2.0
    ra, _ = extraterrestrial_radiation(row_date, latitude)
    rns = (1.0 - 0.23) * float(rs)
    rso = (0.75 + 2e-5 * float(altitude_m)) * ra
    rnl = 4.903e-9 * (pow(float(tmax) + 273.16, 4) + pow(float(tmin) + 273.16, 4)) / 2.0
    rnl *= (0.34 - 0.14 * sqrt(max(float(ea), 0.0))) * (1.35 * _safe_div(float(rs), rso) - 0.35)
    rn = rns - rnl

    pressure = 101.3 * pow((293.0 - 0.0065 * float(altitude_m)) / 293.0, 5.26)
    gamma = 0.000665 * pressure
    slope = 4098.0 * saturation_vapor_pressure(tmean) / pow(float(tmean) + 237.3, 2)
    denominator = slope + gamma * (1.0 + 0.34 * float(u2))
    if denominator == 0:
        return 0.0
    return (
        0.408 * slope * rn
        + gamma * (900.0 / (float(tmean) + 273.0)) * float(u2) * (es - float(ea))
    ) / denominator


def _safe_div(left: float, right: float) -> float:
    return 0.0 if right == 0 else left / right
