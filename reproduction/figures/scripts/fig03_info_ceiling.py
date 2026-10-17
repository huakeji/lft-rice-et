"""Fig. 3 Oracle-input diagnostic: RMSE when observed target-day variables are added to the public forecast.

Inputs (writing_materials/09_info_ceiling/, same HGB learner per lead for every level, 150 random stations, test 2020):
  national_ladder_150stations.csv        order 1: P -> +T -> +S -> +RH -> +wind   (national_info_ceiling.py)
  national_ladder_150stations_order2.csv order 2: P -> +S -> +RH -> +wind -> +T   (national_info_ceiling_order2.py)
Share of reducible MSE = drop of the cell-pooled MSE (mean of squared per-lead RMSE) caused by each variable,
divided by the drop from P to the all-observed level. The dashed line in (a, b) is the operational LFT+GRU
ensemble, 2020, on the same 150 stations.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

D1 = pd.read_csv(os.path.join(S.M, "09_info_ceiling/national_ladder_150stations.csv"))
D2 = pd.read_csv(os.path.join(S.M, "09_info_ceiling/national_ladder_150stations_order2.csv"))
A = Path(S.R) / "aligned_shards"
rng = np.random.default_rng(0); names = sorted(q.name for q in A.glob("station_*.npz"))
pick = [n[8:-4] for n in sorted(rng.choice(names, 150, replace=False))]
z = np.load(Path(S.R) / "formal_summary/ens_lft_gru_y2020.npz"); k = np.isin(z["station"].astype(str), pick)
ENS = np.sqrt(((z["prediction"][k] - z["target"][k]) ** 2).mean(0))   # per-lead RMSE, same stations

VAR_COL = {"P": "#3B3B3B", "T": "#4B2991", "S": "#C0369D", "H": "#FA7876", "W": "#EAB34A"}
VAR_LAB = {"P": "Public forecast (P)", "T": "+ obs. " + r"$T_\mathrm{max}$/$T_\mathrm{min}$", "S": "+ obs. sunshine",
           "H": "+ obs. RH", "W": "+ obs. wind"}
VAR_SHORT = {"T": "Temperature", "S": "Sunshine", "H": "RH", "W": "Wind"}
MK = {"P": "o", "T": "s", "S": "^", "H": "D", "W": "v"}
ORD = [(D1, ["P", "P+T", "P+TS", "P+TSH", "P+all"], ["P", "T", "S", "H", "W"]),
       (D2, ["P", "P+S", "P+SH", "P+SHW", "P+all"], ["P", "S", "H", "W", "T"])]

fig = plt.figure(figsize=(S.W2, 64 * S.MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=0.06, right=0.985, bottom=0.17, top=0.83, wspace=0.34)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
shares = []
for i, (d, levels, vars_) in enumerate(ORD):
    ax = axs[i]
    for lev, v in zip(levels, vars_):
        q = d[d.level == lev].sort_values("lead")
        ax.plot(q.lead, q.rmse, color=VAR_COL[v], marker=MK[v], ms=3.0, mec="white", mew=0.3, lw=1.0, label=VAR_LAB[v])
    ax.plot(range(1, 8), ENS, color="#D55E00", ls="--", lw=0.9, label="LFT+GRU (forecast only)")
    ax.set_xticks(range(1, 8)); ax.set_xlim(0.7, 7.3); ax.set_ylim(0, 0.98)
    ax.set_xlabel("Lead (d)"); ax.set_ylabel(f"RMSE of {S.ET0} ({S.UNIT})")
    ax.set_title(["Order 1: temperature first", "Order 2: sunshine first"][i], fontsize=7.5, pad=3)
    S.panel(ax, "ab"[i], x=-0.22, y=1.02, ha="left")
    mse = d.groupby("level").rmse.apply(lambda s: (s ** 2).mean()).reindex(levels).values
    tot = mse[0] - mse[-1]
    shares.append({v: (mse[j - 1] - mse[j]) / tot for j, v in enumerate(vars_) if j > 0})
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=6.4, columnspacing=1.0, handlelength=1.8)

cx = axs[2]
for row, sh in enumerate(shares):
    left = 0.0
    for v in (["T", "S", "H", "W"] if row == 0 else ["S", "H", "W", "T"]):
        w = 100 * sh[v]
        cx.barh(1 - row, w, left=left, height=0.55, color=VAR_COL[v], edgecolor="white", lw=0.5)
        if w >= 7:
            cx.text(left + w / 2, 1 - row, f"{w:.0f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if v in ("T", "S") else "black")
        else:
            cx.text(left + w / 2, 1 - row + 0.33, f"{w:.0f}%", ha="center", va="bottom", fontsize=6.3)
        left += w
cx.set_yticks([1, 0]); cx.set_yticklabels(["Order 1", "Order 2"])
cx.set_xlim(0, 100); cx.set_ylim(-0.6, 1.6)
cx.set_xlabel("Share of reducible MSE (%)")
for v in ["T", "S", "H", "W"]:
    cx.bar(0, 0, color=VAR_COL[v], label=VAR_SHORT[v])
cx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=4, fontsize=6.3, handlelength=1.0, columnspacing=0.8, frameon=False)
cx.tick_params(axis="y", length=0)
S.panel(cx, "c", x=-0.16, y=1.02, ha="left")
fig.subplots_adjust(bottom=0.24)
print("shares:", [{k: round(100 * v, 1) for k, v in s.items()} for s in shares]); print("ensemble per lead:", ENS.round(3), "mean", ENS.mean().round(3))
S.save(fig, "Fig03_info_ceiling")
