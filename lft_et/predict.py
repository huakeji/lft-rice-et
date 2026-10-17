"""Seven-day ETo and rice ETc forecasts from public weather messages (LFT+GRU ensemble, quantile LFT)."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import Message, Station, build_sequence
from .models import load_family

PKG = Path(__file__).resolve().parents[1]
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def load_stations(path: str | Path = PKG / "data/stations.csv") -> dict[str, Station]:
    s = pd.read_csv(path, dtype={"station": str, "climate_zone": str})
    return {r.station: Station(r.station, float(r.lat), float(r.lon), float(r.alt), r.climate_zone) for r in s.itertuples()}


class Forecaster:
    """Loads the three-seed LFT and GRU ensembles, the quantile LFT and the station-by-month crop coefficients."""

    def __init__(self, model_dir: str | Path = PKG / "model", kc_path: str | Path | None = None,
                 quantiles: bool = True, threads: int = 4):
        torch.set_num_threads(threads)
        self.families = {f: load_family(model_dir, f) for f in (["lft", "gru"] + (["lftq"] if quantiles else []))}
        self.kc = None
        if kc_path is not None and Path(kc_path).exists():
            k = pd.read_csv(kc_path, dtype={"station": str})
            self.kc = {(r.station, int(r.month)): (float(r.Kc), int(r.rice_season_month)) for r in k.itertuples()}

    def _run(self, family: str, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        enc, meta, nets = self.families[family]
        num = torch.from_numpy(enc.numeric(x))
        ids = torch.from_numpy(enc.ids(c))
        with torch.no_grad():
            out = np.mean([net(num, ids).numpy() for net in nets], 0)
        return out * enc.target_scale + enc.target_mean

    def predict_arrays(self, x: np.ndarray, c: np.ndarray) -> dict:
        """x [N,7,36], c [N,7,15] -> ETo (mm d-1) for leads 1..7: ensemble, LFT, GRU and quantiles [N,7,7]."""
        lft = self._run("lft", x, c)
        gru = self._run("gru", x[:, :, :10], c[:, :, :3])
        out = {"eto": np.clip((lft + gru) / 2, 0, None), "eto_lft": lft, "eto_gru": gru}
        if "lftq" in self.families:
            q = np.sort(self._run("lftq", x, c), axis=-1)   # enforce non-crossing quantiles
            out["eto_quantiles"] = np.clip(q, 0, None)
        enc = self.families["lft"][0]
        out["unknown_text_share"] = enc.unknown_share(c)
        return out

    def forecast(self, station: Station, issue: date, messages: dict[date, Message]) -> pd.DataFrame:
        seq = build_sequence(station, issue, messages)
        if seq is None:
            raise ValueError(f"issue {issue} of station {station.station} lacks a complete seven-day message")
        return self.forecast_many([(station, issue, seq)])

    def forecast_many(self, items) -> pd.DataFrame:
        """items: iterable of (Station, issue date, (x, c)) -> long table, one row per station, issue and lead."""
        items = list(items)
        if not items:
            return pd.DataFrame()
        x = np.stack([s[2][0] for s in items])
        c = np.stack([s[2][1] for s in items])
        p = self.predict_arrays(x, c)
        rows = []
        for n, (st, issue, _) in enumerate(items):
            for k in range(7):
                valid = issue + timedelta(days=k)
                r = dict(station=st.station, issue_date=issue.isoformat(), lead=k + 1, valid_date=valid.isoformat(),
                         eto=round(float(p["eto"][n, k]), 3), eto_lft=round(float(p["eto_lft"][n, k]), 3), eto_gru=round(float(p["eto_gru"][n, k]), 3))
                if "eto_quantiles" in p:
                    for qi, q in enumerate(QUANTILES):
                        r[f"eto_q{int(q * 100):02d}"] = round(float(p["eto_quantiles"][n, k, qi]), 3)
                if self.kc is not None and (st.station, valid.month) in self.kc:
                    kc, season = self.kc[(st.station, valid.month)]
                    r.update(kc=round(kc, 4), rice_season=season, etc=round(kc * r["eto"], 3))
                rows.append(r)
        return pd.DataFrame(rows)


def sequences_from_archive(station: Station, messages: dict[date, Message], start: date | None = None, end: date | None = None):
    """All issue dates of an archive with a complete seven-day current message, as (Station, issue, (x, c))."""
    for issue in sorted(messages):
        if (start and issue < start) or (end and issue > end):
            continue
        seq = build_sequence(station, issue, messages)
        if seq is not None:
            yield station, issue, seq
