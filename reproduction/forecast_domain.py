from __future__ import annotations

import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from db_io import connect, load_config


DEFAULT_START = "2012-01-01"
DEFAULT_END = "2020-12-31"
FORECAST_LEADS = list(range(1, 16))
MMD_FEATURES = ["tmax", "tmin", "rain_signal"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse city-level forecast JSON and quantify forecast-vs-history domain shift."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--station", required=True)
    parser.add_argument("--city")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out-dir", default="data/processed/forecast_domain")
    parser.add_argument("--report-dir", default="reports/domain_shift")
    parser.add_argument("--mmd-sample", type=int, default=3000)
    args = parser.parse_args()

    station_code = normalize_station(args.station)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    with connect(config) as db:
        if db.backend != "sqlite":
            raise RuntimeError("forecast_domain.py currently expects the SQLite project database.")
        city = args.city or station_city(db.conn, station_code)
        if not city:
            raise SystemExit(f"No mapped forecast city found for station {station_code}")
        forecast = parse_forecast_city(db.conn, station_code, city, args.start, args.end)
        history = station_history_features(db.conn, station_code, args.start, args.end)

    joined = forecast.merge(
        history,
        on=["station_code", "target_date"],
        how="inner",
        suffixes=("_forecast", "_history"),
    )
    joined_complete = joined[joined["forecast_complete"]].copy()

    summary = summarize_domain_shift(
        forecast,
        history,
        joined_complete,
        station_code,
        city,
        args.start,
        args.end,
        mmd_sample=args.mmd_sample,
    )

    forecast_path = out_dir / f"forecast_long_{station_code}.csv"
    joined_path = out_dir / f"forecast_history_joined_{station_code}.csv"
    summary_path = report_dir / f"domain_shift_{station_code}.summary.json"
    lead_metrics_path = report_dir / f"domain_shift_{station_code}.by_lead.csv"
    histogram_path = report_dir / f"domain_shift_{station_code}.histograms.png"

    forecast.to_csv(forecast_path, index=False)
    joined_complete.to_csv(joined_path, index=False)
    write_csv(lead_metrics_path, summary["by_lead"])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_histograms(joined_complete, histogram_path)

    print(f"station: {station_code}")
    print(f"city: {city}")
    print(f"forecast rows: {len(forecast)}")
    print(f"joined complete rows: {len(joined_complete)}")
    print(f"forecast: {forecast_path}")
    print(f"joined: {joined_path}")
    print(f"summary: {summary_path}")
    print(f"by_lead: {lead_metrics_path}")
    print(f"histograms: {histogram_path}")
    print(
        "overall MMD2 "
        f"[Tmax,Tmin,rain_proxy]: {summary['overall']['mmd2_tmax_tmin_rain_signal']:.6f}"
    )
    return 0


def normalize_station(value: str) -> str:
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def station_city(conn, station_code: str) -> str | None:
    row = conn.execute(
        """
        SELECT city
        FROM city_station_mapping
        WHERE station_code = ?
        ORDER BY CASE WHEN city = (SELECT station_name FROM stations WHERE station_code = ?) THEN 0 ELSE 1 END,
                 city
        LIMIT 1
        """,
        (station_code, station_code),
    ).fetchone()
    return None if row is None else str(row["city"])


def parse_forecast_city(conn, station_code: str, city: str, start: str, end: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT city, date, day_weather_json, day_low_json, day_high_json, day_wind_json, source_file
        FROM forecast_raw_daily
        WHERE city = ?
          AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        (city, start, end),
    ).fetchall()

    parsed: list[dict[str, Any]] = []
    for raw in rows:
        init_date = pd.to_datetime(raw["date"]).date()
        weather = parse_json_map(raw["day_weather_json"])
        low = parse_json_map(raw["day_low_json"])
        high = parse_json_map(raw["day_high_json"])
        wind = parse_json_map(raw["day_wind_json"])
        for panel_index in FORECAST_LEADS:
            key = str(panel_index)
            # Archived pairing convention: lead = panel_index - 1.
            # Earlier lag-correlation checks do not establish original issue/
            # valid-time semantics. Retained for reproducibility, pending source
            # confirmation; index 1 cannot yet be certified as a same-day nowcast.
            lead = panel_index - 1
            if lead < 1:
                continue  # omit index 1 under the archived convention
            tmin = to_float(low.get(key))
            tmax = to_float(high.get(key))
            weather_text = clean_text(weather.get(key))
            wind_text = clean_text(wind.get(key))
            target_date = init_date + timedelta(days=lead)
            parsed.append(
                {
                    "station_code": station_code,
                    "city": city,
                    "init_date": init_date.isoformat(),
                    "lead": lead,
                    "target_date": target_date.isoformat(),
                    "weather_text": weather_text,
                    "wind_text": wind_text,
                    "tmin": tmin,
                    "tmax": tmax,
                    "rain_signal": weather_to_rain_signal(weather_text),
                    "forecast_complete": (
                        weather_text != ""
                        and wind_text != ""
                        and tmin is not None
                        and tmax is not None
                        and tmax >= tmin
                    ),
                    "source_file": raw["source_file"],
                }
            )
    return pd.DataFrame(parsed)


def station_history_features(conn, station_code: str, start: str, end: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT station_code, date AS target_date, tmax, tmin, rain
        FROM historical_weather_daily
        WHERE station_code = ?
          AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        (station_code, start, end),
    ).fetchall()
    df = pd.DataFrame([dict(row) for row in rows])
    if df.empty:
        return pd.DataFrame(columns=["station_code", "target_date", "tmax", "tmin", "rain", "rain_signal"])
    for column in ["tmax", "tmin", "rain"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[df["tmax"].notna() & df["tmin"].notna() & (df["tmax"] < df["tmin"]), ["tmax", "tmin"]] = np.nan
    df.loc[df["rain"] < 0, "rain"] = np.nan
    df["rain_signal"] = df["rain"].map(history_rain_to_signal)
    return df


def summarize_domain_shift(
    forecast: pd.DataFrame,
    history: pd.DataFrame,
    joined: pd.DataFrame,
    station_code: str,
    city: str,
    start: str,
    end: str,
    mmd_sample: int,
) -> dict[str, Any]:
    by_lead = []
    for lead in FORECAST_LEADS:
        part = joined[joined["lead"] == lead]
        by_lead.append(lead_summary(part, lead, mmd_sample))

    overall = lead_summary(joined, "all", mmd_sample)
    return {
        "station_code": station_code,
        "city": city,
        "period": {"start": start, "end": end},
        "feature_note": (
            "Forecast table has no numeric precipitation probability. "
            "MMD uses [Tmax, Tmin, rain_signal], where forecast rain_signal is parsed "
            "from weather text and historical rain_signal is derived from daily rain amount."
        ),
        "forecast_rows_long": int(len(forecast)),
        "forecast_complete_rows": int(forecast["forecast_complete"].sum()) if len(forecast) else 0,
        "history_rows": int(len(history)),
        "joined_complete_rows": int(len(joined)),
        "overall": overall,
        "by_lead": by_lead,
        "weather_top20": value_counts(forecast["weather_text"], n=20),
        "wind_top20": value_counts(forecast["wind_text"], n=20),
    }


def lead_summary(part: pd.DataFrame, lead: int | str, mmd_sample: int) -> dict[str, Any]:
    aligned = part.dropna(
        subset=[
            "tmax_forecast",
            "tmin_forecast",
            "rain_signal_forecast",
            "tmax_history",
            "tmin_history",
            "rain_signal_history",
        ]
    ).copy()
    if aligned.empty:
        return {
            "lead": lead,
            "n": 0,
            "mmd2_tmax_tmin_rain_signal": None,
            "mae_tmax": None,
            "mae_tmin": None,
            "rain_signal_mae": None,
        }

    x = aligned[["tmax_history", "tmin_history", "rain_signal_history"]].to_numpy(dtype=float)
    y = aligned[["tmax_forecast", "tmin_forecast", "rain_signal_forecast"]].to_numpy(dtype=float)
    return {
        "lead": lead,
        "n": int(len(aligned)),
        "mmd2_tmax_tmin_rain_signal": float(rbf_mmd2(x, y, max_samples=mmd_sample)),
        "mae_tmax": float(np.mean(np.abs(aligned["tmax_forecast"] - aligned["tmax_history"]))),
        "mae_tmin": float(np.mean(np.abs(aligned["tmin_forecast"] - aligned["tmin_history"]))),
        "bias_tmax": float(np.mean(aligned["tmax_forecast"] - aligned["tmax_history"])),
        "bias_tmin": float(np.mean(aligned["tmin_forecast"] - aligned["tmin_history"])),
        "rain_signal_mae": float(
            np.mean(np.abs(aligned["rain_signal_forecast"] - aligned["rain_signal_history"]))
        ),
        "forecast_complete_rate_in_joined": float(part["forecast_complete"].mean()) if len(part) else 0.0,
    }


def rbf_mmd2(x: np.ndarray, y: np.ndarray, max_samples: int = 3000, seed: int = 42) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    if len(x) > max_samples:
        idx = rng.choice(len(x), size=max_samples, replace=False)
        x = x[idx]
    if len(y) > max_samples:
        idx = rng.choice(len(y), size=max_samples, replace=False)
        y = y[idx]

    combined = np.vstack([x, y])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    y = (y - mean) / std

    gamma = median_gamma(np.vstack([x, y]), sample=min(1200, len(x) + len(y)), rng=rng)
    k_xx = rbf_kernel_mean(x, x, gamma)
    k_yy = rbf_kernel_mean(y, y, gamma)
    k_xy = rbf_kernel_mean(x, y, gamma)
    return max(0.0, k_xx + k_yy - 2.0 * k_xy)


def median_gamma(values: np.ndarray, sample: int, rng: np.random.Generator) -> float:
    if len(values) > sample:
        values = values[rng.choice(len(values), size=sample, replace=False)]
    diff = values[:, None, :] - values[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    tri = dist2[np.triu_indices_from(dist2, k=1)]
    median = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
    return 1.0 / (2.0 * median)


def rbf_kernel_mean(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    total = 0.0
    chunk = 500
    count = 0
    for start in range(0, len(x), chunk):
        block = x[start : start + chunk]
        diff = block[:, None, :] - y[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        total += float(np.exp(-gamma * dist2).sum())
        count += block.shape[0] * len(y)
    return total / count if count else 0.0


def weather_to_rain_signal(text: str) -> float:
    if not text:
        return np.nan
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
    return float(max(levels)) if levels else np.nan


def history_rain_to_signal(value: Any) -> float:
    rain = to_float(value)
    if rain is None:
        return np.nan
    if rain <= 0:
        return 0.0
    if rain < 10:
        return 2.0
    if rain < 25:
        return 3.0
    if rain < 50:
        return 4.0
    if rain < 100:
        return 4.5
    if rain < 250:
        return 5.0
    return 6.0


def split_weather_text(text: str) -> list[str]:
    normalized = text.replace("转", "|").replace("到", "|").replace("/", "|").replace("-", "|")
    return [part.strip() for part in normalized.split("|") if part.strip()]


def parse_json_map(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def clean_text(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text in ("None", "nan", "0") else text


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def value_counts(series: pd.Series, n: int = 20) -> list[dict[str, Any]]:
    counts = series.fillna("").replace("", "<missing>").value_counts().head(n)
    return [{"value": str(value), "count": int(count)} for value, count in counts.items()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_histograms(joined: pd.DataFrame, path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.text((40, 20), "Forecast vs historical feature distributions", fill=(20, 20, 20, 255))

    panels = [
        ("Tmax", "tmax_history", "tmax_forecast", (40, 70, 370, 320)),
        ("Tmin", "tmin_history", "tmin_forecast", (430, 70, 760, 320)),
        ("Rain signal", "rain_signal_history", "rain_signal_forecast", (820, 70, 1150, 320)),
        ("Lead-1 Tmax error", "lead1_tmax_error", "", (40, 410, 370, 660)),
        ("Lead-7 Tmax error", "lead7_tmax_error", "", (430, 410, 760, 660)),
        ("Lead-15 Tmax error", "lead15_tmax_error", "", (820, 410, 1150, 660)),
    ]
    for title, hist_col, fcst_col, box in panels:
        if hist_col.startswith("lead"):
            lead = int(hist_col.replace("lead", "").replace("_tmax_error", ""))
            part = joined[joined["lead"] == lead]
            values = (part["tmax_forecast"] - part["tmax_history"]).dropna().to_numpy(dtype=float)
            draw_single_hist(draw, box, title, values, (-15, 15), (214, 39, 40, 150))
        else:
            hist_values = joined[hist_col].dropna().to_numpy(dtype=float)
            fcst_values = joined[fcst_col].dropna().to_numpy(dtype=float)
            value_range = combined_range(hist_values, fcst_values)
            draw_dual_hist(draw, box, title, hist_values, fcst_values, value_range)
    image.save(path)


def combined_range(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([left, right]) if len(left) and len(right) else np.array([0.0, 1.0])
    lo = float(np.nanpercentile(values, 1))
    hi = float(np.nanpercentile(values, 99))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def draw_dual_hist(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    history: np.ndarray,
    forecast: np.ndarray,
    value_range: tuple[float, float],
) -> None:
    draw_panel(draw, box, title)
    hist_counts, edges = np.histogram(history, bins=18, range=value_range)
    fcst_counts, _ = np.histogram(forecast, bins=18, range=value_range)
    max_count = max(int(hist_counts.max() if len(hist_counts) else 0), int(fcst_counts.max() if len(fcst_counts) else 0), 1)
    left, top, right, bottom = box
    inner_left, inner_top, inner_right, inner_bottom = left + 35, top + 35, right - 15, bottom - 35
    bar_w = (inner_right - inner_left) / len(hist_counts)
    for idx, (h_count, f_count) in enumerate(zip(hist_counts, fcst_counts)):
        x0 = inner_left + idx * bar_w
        x1 = x0 + bar_w * 0.42
        x2 = x0 + bar_w * 0.45
        x3 = x0 + bar_w * 0.87
        yh = inner_bottom - (h_count / max_count) * (inner_bottom - inner_top)
        yf = inner_bottom - (f_count / max_count) * (inner_bottom - inner_top)
        draw.rectangle([(x0, yh), (x1, inner_bottom)], fill=(31, 119, 180, 125))
        draw.rectangle([(x2, yf), (x3, inner_bottom)], fill=(255, 127, 14, 125))
    draw.text((inner_left, bottom - 25), "blue=history orange=forecast", fill=(60, 60, 60, 255))


def draw_single_hist(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    values: np.ndarray,
    value_range: tuple[float, float],
    color: tuple[int, int, int, int],
) -> None:
    draw_panel(draw, box, title)
    counts, _ = np.histogram(values, bins=18, range=value_range)
    max_count = max(int(counts.max() if len(counts) else 0), 1)
    left, top, right, bottom = box
    inner_left, inner_top, inner_right, inner_bottom = left + 35, top + 35, right - 15, bottom - 35
    bar_w = (inner_right - inner_left) / len(counts)
    for idx, count in enumerate(counts):
        x0 = inner_left + idx * bar_w
        x1 = x0 + bar_w * 0.8
        y = inner_bottom - (count / max_count) * (inner_bottom - inner_top)
        draw.rectangle([(x0, y), (x1, inner_bottom)], fill=color)
    draw.text((inner_left, bottom - 25), "forecast - history", fill=(60, 60, 60, 255))


def draw_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    left, top, right, bottom = box
    draw.rectangle([(left, top), (right, bottom)], outline=(90, 90, 90, 255), width=2)
    draw.text((left + 10, top + 10), title, fill=(20, 20, 20, 255))


if __name__ == "__main__":
    raise SystemExit(main())
