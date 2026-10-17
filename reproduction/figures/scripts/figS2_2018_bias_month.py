"""Fig. S2 2018 short-history stress test: mean bias (forecast - reference ET0) by target month.

Ensemble rows in formal_summary/ens_lft_gru_y{Y}.npz carry no dates; they are row-aligned with the
per-seed outer predictions in formal/{lft,gru}_y{Y}_s*/outer_predictions.npz (verified below: identical
station order and ensemble == mean(seed-mean LFT, seed-mean GRU) to machine precision), whose 'origin'
gives the issue date. Target date = origin + lead - 1 (lead 1 = issue day). Aggregation: mean over all
station-days and leads falling in each target calendar month.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


def load(y):
    e = np.load(os.path.join(S.R, f"formal_summary/ens_lft_gru_y{y}.npz"))
    out, ref = {}, None
    for m in ("lft", "gru"):
        ps = [np.load(os.path.join(S.R, f"formal/{m}_y{y}_s{s}/outer_predictions.npz")) for s in range(3)]
        for p in ps[1:]:
            assert (p["origin"] == ps[0]["origin"]).all() and (p["station"] == ps[0]["station"]).all()
        out[m.upper()] = np.mean([p["prediction"] for p in ps], 0)
        ref = ps[0]
    assert (e["station"] == ref["station"]).all()
    assert np.abs((out["LFT"] + out["GRU"]) / 2 - e["prediction"]).max() < 1e-5
    out["LFT+GRU"] = e["prediction"]
    org = pd.to_datetime(ref["origin"]).values
    tgt = org[:, None] + (np.arange(7)[None, :] * np.timedelta64(1, "D"))
    month = pd.DatetimeIndex(tgt.ravel()).month.values
    res = {}
    for k, p in out.items():
        err = (p - e["target"]).ravel()
        res[k] = pd.Series(err).groupby(month).mean()
    return res


R = {y: load(y) for y in (2018, 2019, 2020)}
fig, ax = plt.subplots(figsize=(S.W15, 68 * S.MM))
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.15, top=0.88)
mo = np.arange(1, 13)
ax.axhline(0, color="#777777", lw=0.5)
for y, ls, c in ((2019, (0, (3, 1.5)), "#9E9E9E"), (2020, (0, (1, 1)), "#9E9E9E")):
    ax.plot(mo, R[y]["LFT+GRU"].reindex(mo), color=c, ls=ls, lw=1.0, label=f"LFT+GRU, {y}")
for m in ("LFT", "GRU", "LFT+GRU"):
    ax.plot(mo, R[2018][m].reindex(mo), color=S.MODEL_COL[m], marker=S.MODEL_MK[m], ms=3.2, mec="white", mew=0.3,
            lw=1.6 if m == "LFT+GRU" else 1.0, label=f"{m}, 2018", zorder=4 if m == "LFT+GRU" else 3)
ax.set_xticks(mo); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_xlim(0.6, 12.4)
ax.set_xlabel("Target month"); ax.set_ylabel(f"Mean bias of {S.ET0} ({S.UNIT})")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize=6.4, columnspacing=1.0, handlelength=2.0)
for y in R:
    print(y, {k: v.round(3).tolist() for k, v in R[y].items() if k == "LFT+GRU"})
print("2018 LFT", R[2018]["LFT"].round(3).tolist()); print("2018 GRU", R[2018]["GRU"].round(3).tolist())
S.save(fig, "FigS2_2018_bias_by_month")
