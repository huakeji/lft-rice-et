"""Command-line interface.

    python -m lft_et.cli --input examples/example_messages.json --output forecast.csv

Input: a JSON object or list of objects
    {"station": "57494",                      # a station of data/stations.csv, or an object with
                                              # station, lat, lon, alt (decimal degrees, m) and zone
     "issue_date": "2020-07-15",
     "messages": [{"date": "2020-07-15", "weather": {"1": "多云", ...}, "high": {"1": 34, ...},
                   "low": {"1": 26, ...}, "wind": {"1": "3-4级", ...}}, ...]}   # issue date and up to 6 earlier issues
Output: one row per station, issue and lead with ETo (ensemble, LFT, GRU) and quantiles; with --kc, also Kc,
the rice-season flag and standard-condition rice ETc (mm d-1).
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from .features import Message, Station, build_sequence
from .predict import Forecaster, load_stations


def parse(obj: dict, stations: dict[str, Station]):
    s = obj["station"]
    if isinstance(s, dict):
        st = Station(str(s["station"]), float(s["lat"]), float(s["lon"]), float(s["alt"]), str(s.get("zone", "unknown")))
    else:
        st = stations[str(s)]
    issue = date.fromisoformat(obj["issue_date"])
    msgs = {date.fromisoformat(m["date"]): Message(m.get("weather", {}), m.get("high", {}), m.get("low", {}), m.get("wind", {}))
            for m in obj["messages"]}
    return st, issue, msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="forecast.csv")
    ap.add_argument("--no-quantiles", action="store_true")
    ap.add_argument("--kc", default=None, help="CSV with columns station, month, Kc, rice_season_month")
    a = ap.parse_args()
    data = json.load(open(a.input, encoding="utf-8"))
    data = data if isinstance(data, list) else [data]
    stations = load_stations()
    fc = Forecaster(kc_path=a.kc, quantiles=not a.no_quantiles)
    items = []
    for obj in data:
        st, issue, msgs = parse(obj, stations)
        seq = build_sequence(st, issue, msgs)
        if seq is None:
            print(f"skipped {st.station} {issue}: incomplete current message")
            continue
        items.append((st, issue, seq))
    out = fc.forecast_many(items)
    out.to_csv(a.output, index=False)
    print(out.to_string(index=False) if len(out) <= 30 else f"wrote {len(out)} rows to {a.output}")


if __name__ == "__main__":
    main()
