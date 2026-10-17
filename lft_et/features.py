"""Build model inputs from Chinese public city weather forecasts (weather.com.cn style messages).

A message issued on date O lists, for keys "1".."15", the weather phrase, the daily maximum and minimum
temperature and the wind force text; key "1" is the issue day itself (valid date O), key k is O+k-1.
For an issue date O the model uses the first seven keys of issue O and, for each of the six previous
issues O-j (j = 1..6), what that issue said about the same valid dates (key k+j).

Array layout (identical to the arrays used in training):
  x[7, 36]: tmax, tmin, dtr, rain_level, season_sin, season_cos, lat, lon, alt, lead,
            then for j = 1..6: tmax_j, tmin_j, rain_j, avail_j, then spread_tmax, spread_tmin
  c[7, 15]: weather, wind, zone, then for j = 1..6: weather_j, wind_j
Earlier issues that are missing or invalid are filled with the current issue and flagged avail_j = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

N_LAGS = 6
EMPTY = ("", "none", "nan", "0")


def split_weather_text(text: str) -> list[str]:
    normalized = text.replace("转", "|").replace("到", "|").replace("/", "|").replace("-", "|")
    return [part.strip() for part in normalized.split("|") if part.strip()]


def rain_level(text: str) -> float:
    """Ordinal precipitation level parsed from the weather phrase (0 none, 1.5 snow, 2-6 light to extreme rain)."""
    if not text:
        return float("nan")
    levels = []
    for token in split_weather_text(text):
        if "特大暴雨" in token:
            levels.append(6.0)
        elif "大暴雨" in token:
            levels.append(5.0)
        elif "暴雨" in token:
            levels.append(4.5)
        elif "大雨" in token:
            levels.append(4.0)
        elif "中雨" in token:
            levels.append(3.0)
        elif "小雨" in token or "阵雨" in token or "雨" in token:
            levels.append(2.0)
        elif "雪" in token:
            levels.append(1.5)
        else:
            levels.append(0.0)
    return float(max(levels)) if levels else float("nan")


def temperature(value) -> float:
    """Forecast temperature in deg C; values outside [-50, 50] (archive sentinels) are treated as missing."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if -50 <= v <= 50 else float("nan")


@dataclass
class Message:
    """One public forecast issue: dicts keyed "1".."15"."""
    weather: dict
    high: dict
    low: dict
    wind: dict

    def day(self, key: int):
        k = str(key)
        return (str(self.weather.get(k) or "").strip(), temperature(self.high.get(k)),
                temperature(self.low.get(k)), str(self.wind.get(k) or "").strip())


@dataclass
class Station:
    station: str
    lat: float   # decimal degrees
    lon: float   # decimal degrees
    alt: float   # m a.s.l.
    zone: str    # climate-zone token of the training network (see data/stations.csv)


def build_sequence(station: Station, issue: date, messages: dict[date, Message]):
    """Return (x, c) for the issue date, or None if the current issue is incomplete for the seven days."""
    cur = messages.get(issue)
    if cur is None:
        return None
    x = np.zeros((7, 10 + 4 * N_LAGS + 2), dtype="float32")
    c = np.empty((7, 3 + 2 * N_LAGS), dtype=object)
    for k in range(1, 8):
        wt, hi, lo, wn = cur.day(k)
        if wt.lower() in EMPTY or not (math.isfinite(hi) and math.isfinite(lo)) or hi < lo:
            return None
        doy = (issue + timedelta(days=k - 1)).timetuple().tm_yday
        x[k - 1, :10] = [hi, lo, hi - lo, rain_level(wt), math.sin(2 * math.pi * doy / 365.0),
                         math.cos(2 * math.pi * doy / 365.0), station.lat, station.lon, station.alt, float(k)]
        c[k - 1, :3] = [wt, wn, str(station.zone)]
    hs, ls = [x[:, 0].copy()], [x[:, 1].copy()]
    for j in range(1, N_LAGS + 1):
        prev = messages.get(issue - timedelta(days=j))
        h_j = np.full(7, np.nan)
        l_j = np.full(7, np.nan)
        for k in range(1, 8):
            col = 10 + 4 * (j - 1)
            if prev is not None:
                wt, hi, lo, wn = prev.day(k + j)
            else:
                wt, hi, lo, wn = "", float("nan"), float("nan"), ""
            ok = math.isfinite(hi) and math.isfinite(lo) and hi >= lo and wt.lower() not in EMPTY
            if ok:
                x[k - 1, col:col + 4] = [hi, lo, rain_level(wt), 1.0]
                c[k - 1, 3 + 2 * (j - 1)] = wt
                c[k - 1, 4 + 2 * (j - 1)] = wn if wn != "" else c[k - 1, 1]
                h_j[k - 1], l_j[k - 1] = hi, lo
            else:
                x[k - 1, col:col + 4] = [x[k - 1, 0], x[k - 1, 1], x[k - 1, 3], 0.0]
                c[k - 1, 3 + 2 * (j - 1)] = c[k - 1, 0]
                c[k - 1, 4 + 2 * (j - 1)] = c[k - 1, 1]
        hs.append(h_j)
        ls.append(l_j)
    with np.errstate(all="ignore"):
        x[:, -2] = np.nanstd(np.stack(hs), 0)
        x[:, -1] = np.nanstd(np.stack(ls), 0)
    return x, c.astype(str)
