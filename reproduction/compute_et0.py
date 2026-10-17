from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from db_io import connect, load_config, station_metadata_to_decimal
from fao56 import (
    hargreaves_samani,
    saturation_vapor_pressure,
    sunshine_to_rs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute daily FAO-56/ASCE ET0 for one station.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--station", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    station_code = normalize_station(args.station)
    out_dir = Path(args.out_dir)
    if any((out_dir / name).exists() for name in (
        f"et0_{station_code}.csv", f"et0_{station_code}.summary.json", f"et0_{station_code}.hs_scatter.png"
    )):
        raise FileExistsError("Use a new output directory: corrected reference targets must not overwrite legacy outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    with connect(config) as db:
        if db.backend != "sqlite":
            raise RuntimeError("compute_et0.py currently expects the SQLite project database.")
        meta = station_meta(db.conn, station_code)
        if meta is None:
            raise SystemExit(f"No station metadata found for station {station_code}")
        df = station_history(db.conn, station_code)

    if df.empty:
        raise SystemExit(f"No history rows found for station {station_code}")

    df = compute_et0(df, meta)
    summary = summarize(df, station_code, meta)

    csv_path = out_dir / f"et0_{station_code}.csv"
    summary_path = out_dir / f"et0_{station_code}.summary.json"
    scatter_path = out_dir / f"et0_{station_code}.hs_scatter.png"

    df.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_hs_scatter(df, scatter_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"hs_scatter: {scatter_path}")
    return 0


def normalize_station(value: str) -> str:
    value = str(value).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def _ensure_refet():
    try:
        import refet

        return refet
    except ImportError:
        deps = Path(__file__).resolve().parent / ".deps"
        if deps.exists():
            sys.path.insert(0, str(deps))
        import refet

        return refet


def station_meta(conn, station_code: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT station_code, province, station_name, longitude, latitude, altitude_m
        FROM stations
        WHERE station_code = ?
        """,
        (station_code,),
    ).fetchone()
    return None if row is None else station_metadata_to_decimal(row)


def station_history(conn, station_code: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT
            station_code,
            date,
            t_mean,
            tmax,
            tmin,
            rh,
            rain,
            wind10,
            sunshine_hours,
            et0 AS et0_db,
            source
        FROM historical_weather_daily
        WHERE station_code = ?
        ORDER BY date
        """,
        (station_code,),
    ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def compute_et0(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    refet = _ensure_refet()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["doy"] = out["date"].dt.dayofyear

    for col in ["t_mean", "tmax", "tmin", "rh", "rain", "wind10", "sunshine_hours", "et0_db"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    valid = (
        out["tmax"].notna()
        & out["tmin"].notna()
        & out["rh"].notna()
        & out["wind10"].notna()
        & out["sunshine_hours"].notna()
        & (out["tmax"] >= out["tmin"])
        & out["rh"].between(0, 100)
        & (out["wind10"] >= 0)
        & (out["sunshine_hours"] >= 0)
        & np.isfinite(out[["tmax", "tmin", "rh", "wind10", "sunshine_hours"]].to_numpy(dtype=float)).all(axis=1)
    )

    out["es"] = (out["tmax"].map(saturation_vapor_pressure) + out["tmin"].map(saturation_vapor_pressure)) / 2.0
    out["ea"] = out["es"] * out["rh"] / 100.0
    latitude = float(meta["latitude"])
    altitude = float(meta["altitude_m"])

    out["rs"] = [
        sunshine_to_rs(row_date.date(), latitude, n)
        for row_date, n in zip(out["date"], out["sunshine_hours"])
    ]
    out["et0_hs"] = [
        hargreaves_samani(row_date.date(), latitude, tmax, tmin)
        for row_date, tmax, tmin in zip(out["date"], out["tmax"], out["tmin"])
    ]
    out["et0_refet"] = pd.NA

    v = out.loc[valid]
    if not v.empty:
        eto = refet.Daily(
            tmin=v["tmin"].to_numpy(dtype=float),
            tmax=v["tmax"].to_numpy(dtype=float),
            ea=v["ea"].to_numpy(dtype=float),
            rs=v["rs"].to_numpy(dtype=float),
            uz=v["wind10"].to_numpy(dtype=float),
            zw=10.0,
            elev=altitude,
            # refet.Daily accepts degrees, converting them internally to radians.
            lat=latitude,
            input_units={"lat": "deg"},
            doy=v["doy"].to_numpy(dtype=int),
            method="asce",
        ).eto()
        out.loc[valid, "et0_refet"] = eto

    out["quality_valid_physical"] = valid
    out.attrs["reference_definition"] = {
        "version": "asce_daily_short_latitude_degrees_v2",
        "refet_version": refet.__version__,
        "method": "asce",
        "surface": "short reference grass; computed target, not measured actual ET",
        "latitude_input_unit": "degrees",
        "wind_input_height_m": 10.0,
        "tmean": "(tmax+tmin)/2; observed t_mean not used by refet.Daily",
        "ea": "mean saturation pressure at Tmax/Tmin multiplied by RH/100",
        "rs": "Angstrom-Prescott a=0.25 b=0.50; sunshine fraction clipped to [0,1]",
        "negative_target_policy": "preserve signed output; do not silently clip reference labels",
    }
    out["date"] = out["date"].dt.date.astype(str)
    return out


def summarize(df: pd.DataFrame, station_code: str, meta: dict[str, Any]) -> dict[str, Any]:
    valid = df[df["quality_valid_physical"] & df["et0_refet"].notna()].copy()
    valid["et0_refet"] = pd.to_numeric(valid["et0_refet"], errors="coerce")
    valid = valid.dropna(subset=["et0_refet", "et0_hs", "et0_db"])

    return {
        "reference_definition": df.attrs.get("reference_definition", {}),
        "station": meta,
        "station_code": station_code,
        "rows_total": int(len(df)),
        "rows_valid_physical": int(df["quality_valid_physical"].sum()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "et0_refet": describe(valid["et0_refet"]),
        "et0_db": describe(valid["et0_db"]),
        "et0_hs": describe(valid["et0_hs"]),
        "compare_refet_vs_db": compare(valid["et0_refet"], valid["et0_db"]),
        "compare_refet_vs_hs": compare(valid["et0_refet"], valid["et0_hs"]),
    }


def describe(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "mean": float(series.mean()),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def compare(left: pd.Series, right: pd.Series) -> dict[str, float]:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    err = aligned["left"] - aligned["right"]
    ss_res = float((err**2).sum())
    ss_tot = float(((aligned["right"] - aligned["right"].mean()) ** 2).sum())
    return {
        "n": int(len(aligned)),
        "mae": float(err.abs().mean()),
        "rmse": float((err**2).mean() ** 0.5),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot else 0.0,
        "corr2": float(aligned["left"].corr(aligned["right"]) ** 2),
    }


def plot_hs_scatter(df: pd.DataFrame, path: Path) -> None:
    from PIL import Image, ImageDraw

    valid = df[df["quality_valid_physical"] & df["et0_refet"].notna()].copy()
    valid["et0_refet"] = pd.to_numeric(valid["et0_refet"], errors="coerce")
    valid = valid.dropna(subset=["et0_refet", "et0_hs"])

    sample = valid.sample(n=min(8000, len(valid)), random_state=42) if len(valid) else valid
    width, height = 1000, 850
    left, top, right, bottom = 95, 55, 955, 755
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")

    lim_max = max(float(valid["et0_hs"].max()), float(valid["et0_refet"].max()), 1.0)
    lim_max = round(lim_max + 0.5, 1)

    def xy(x: float, y: float) -> tuple[int, int]:
        px = left + int((float(x) / lim_max) * (right - left))
        py = bottom - int((float(y) / lim_max) * (bottom - top))
        return px, py

    for i in range(6):
        value = lim_max * i / 5.0
        x0, y0 = xy(value, value)
        draw.line([(x0, top), (x0, bottom)], fill=(220, 220, 220, 255), width=1)
        draw.line([(left, y0), (right, y0)], fill=(220, 220, 220, 255), width=1)
        draw.text((x0 - 10, bottom + 10), f"{value:.1f}", fill=(60, 60, 60, 255))
        draw.text((left - 55, y0 - 7), f"{value:.1f}", fill=(60, 60, 60, 255))

    draw.rectangle([(left, top), (right, bottom)], outline=(80, 80, 80, 255), width=2)
    draw.line([xy(0, 0), xy(lim_max, lim_max)], fill=(192, 57, 43, 255), width=3)

    for x, y in zip(sample["et0_hs"], sample["et0_refet"]):
        px, py = xy(float(x), float(y))
        draw.ellipse([(px - 2, py - 2), (px + 2, py + 2)], fill=(30, 110, 180, 70))

    draw.text((left, 20), "ET0 sanity check: HS vs FAO-56 refet", fill=(20, 20, 20, 255))
    draw.text((350, height - 55), "Hargreaves-Samani ET0 (mm/d)", fill=(20, 20, 20, 255))
    draw.text((15, 25), "FAO-56 refet ET0 (mm/d)", fill=(20, 20, 20, 255))
    image.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
