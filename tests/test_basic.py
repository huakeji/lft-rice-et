import json
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from lft_et import Forecaster, Message, Station, build_sequence, load_stations, rain_level

ROOT = Path(__file__).resolve().parents[1]


def example():
    obj = json.loads((ROOT / "examples/example_messages.json").read_text(encoding="utf-8"))[0]
    msgs = {date.fromisoformat(m["date"]): Message(m["weather"], m["high"], m["low"], m["wind"]) for m in obj["messages"]}
    return load_stations()[obj["station"]], date.fromisoformat(obj["issue_date"]), msgs


class Features(unittest.TestCase):
    def test_rain_level(self):
        self.assertEqual(rain_level("晴"), 0.0)
        self.assertEqual(rain_level("多云转小雨"), 2.0)
        self.assertEqual(rain_level("中到大雨"), 4.0)
        self.assertEqual(rain_level("暴雨转大暴雨"), 5.0)
        self.assertEqual(rain_level("小雪"), 1.5)

    def test_sequence_layout_and_missing_lags(self):
        st, issue, msgs = example()
        x, c = build_sequence(st, issue, msgs)
        self.assertEqual(x.shape, (7, 36))
        self.assertEqual(c.shape, (7, 15))
        self.assertTrue(np.all(x[:, 9] == np.arange(1, 8)))
        only_current = {issue: msgs[issue]}
        x2, c2 = build_sequence(st, issue, only_current)
        self.assertTrue(np.all(x2[:, 13:34:4] == 0))           # avail flags of all earlier issues
        self.assertTrue(np.allclose(x2[:, 10], x2[:, 0]))       # filled with the current issue

    def test_incomplete_current_issue(self):
        st, issue, msgs = example()
        bad = dict(msgs)
        m = msgs[issue]
        bad[issue] = Message(dict(m.weather, **{"3": ""}), m.high, m.low, m.wind)
        self.assertIsNone(build_sequence(st, issue, bad))


class Forecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.kc = Path(tempfile.mkdtemp()) / "kc.csv"
        cls.kc.write_text("station,month,Kc,rice_season_month\n57494,7,1.2,1\n")
        cls.fc = Forecaster(kc_path=cls.kc)

    def test_forecast_table(self):
        st, issue, msgs = example()
        out = self.fc.forecast(st, issue, msgs)
        self.assertEqual(len(out), 7)
        self.assertTrue((out.eto >= 0).all())
        q = out[[c for c in out.columns if c.startswith("eto_q")]].values
        self.assertTrue(np.all(np.diff(q, axis=1) >= 0))
        self.assertTrue(np.allclose(out.etc, out.kc * out.eto, atol=2e-3))
        self.assertTrue(np.allclose(out.eto, (out.eto_lft + out.eto_gru) / 2, atol=2e-3))

    def test_new_station(self):
        _, issue, msgs = example()
        st = Station("new", 30.6, 114.1, 25.0, "CC")
        out = self.fc.forecast(st, issue, msgs)
        self.assertTrue(out.kc.isna().all() if "kc" in out else True)


if __name__ == "__main__":
    unittest.main()
