"""Fig. 7 Error attribution: season x forecast rain, lead x rain, rice zone."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

d = pd.read_csv(os.path.join(S.M, "05_attribution/error_by_group.csv"))
RC = {"no_rain": "#A8A8A8", "rain": "#56B4E9"}
RL = {"no_rain": "No rain forecast", "rain": "Rain forecast"}

fig = plt.figure(figsize=(S.W15, 100 * S.MM))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.72], width_ratios=[1.1, 1], hspace=0.42, wspace=0.42,
                      left=0.085, right=0.92, top=0.95, bottom=0.08)

# (a) season x rain, 2020
ax = fig.add_subplot(gs[0, 0])
sr = d[(d.group == "season+rain") & (d.year == 2020)]
seas = ["DJF", "MAM", "JJA", "SON"]
xx = np.arange(4); w = 0.36
ax2 = ax.twinx()
for i, r in enumerate(["no_rain", "rain"]):
    q = sr[sr.rain == r].set_index("season").reindex(seas)
    ax.bar(xx + (i - 0.5) * w, q.rmse, w * 0.92, color=RC[r], edgecolor="none", label=RL[r], zorder=2)
    ax2.plot(xx + (i - 0.5) * w, q.rel_rmse * 100, ls="", marker="o", ms=3.4, mfc="white", mec="k",
             mew=0.7, zorder=4)
ax.set_xticks(xx); ax.set_xticklabels(seas)
ax.set_ylabel(f"RMSE ({S.UNIT})"); ax.set_ylim(0, 1.5)
ax2.set_ylim(0, 45); ax2.set_ylabel("Relative RMSE (%)", labelpad=2)
ax2.spines["right"].set_visible(True); ax2.spines["right"].set_linewidth(0.5); ax2.spines["top"].set_visible(False)
h0, l0 = ax.get_legend_handles_labels()
h0 += [Line2D([], [], ls="", marker="o", ms=3.4, mfc="white", mec="k", mew=0.7)]; l0 += ["Relative RMSE (right axis)"]
ax.legend(h0, l0, loc="upper left", fontsize=6.2, handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
ax.text(0.98, 0.98, "2020", transform=ax.transAxes, ha="right", va="top", fontsize=7)
S.panel(ax, "a", x=-0.2, y=1.02, ha="left")

# (b) lead x rain, relative RMSE
bx = fig.add_subplot(gs[0, 1])
lr = d[d.group == "lead+rain"]
for r in ["no_rain", "rain"]:
    for y, ls, mk in ((2020, "-", "o"), (2019, "--", "s")):
        q = lr[(lr.rain == r) & (lr.year == y)].sort_values("lead")
        bx.plot(q.lead, q.rel_rmse * 100, ls=ls, marker=mk, color=RC[r] if r == "rain" else "#555555",
                ms=2.8, mec="white", mew=0.25, lw=1.0)
bx.set_xticks(range(1, 8)); bx.set_xlim(0.7, 7.3)
bx.set_xlabel("Lead (d)"); bx.set_ylabel("Relative RMSE (%)")
bx.set_ylim(18, 36); bx.yaxis.set_major_locator(plt.MultipleLocator(5))
hs = [Line2D([], [], color="#555555", lw=1.0, label="No rain forecast"),
      Line2D([], [], color=RC["rain"], lw=1.0, label="Rain forecast"),
      Line2D([], [], color="k", lw=0.8, ls="-", marker="o", ms=2.6, label="2020"),
      Line2D([], [], color="k", lw=0.8, ls="--", marker="s", ms=2.6, label="2019")]
bx.legend(handles=hs, loc="upper left", fontsize=6.2, ncol=2, columnspacing=0.8, handlelength=1.8, borderaxespad=0.2)
S.panel(bx, "b", x=-0.22, y=1.02, ha="left")

# (c) RMSE by rice zone, 2020 vs 2019
cx = fig.add_subplot(gs[1, :])
zq = d[d.group == "zone"]
xz = np.arange(6); w = 0.36
for i, y in enumerate([2019, 2020]):
    q = zq[zq.year == y].set_index("zone").reindex(S.ZONES)
    cx.bar(xz + (i - 0.5) * w, q.rmse, w * 0.92, color=[S.ZONE_COL[z] for z in S.ZONES],
           alpha=0.45 if y == 2019 else 1.0, edgecolor="none", zorder=2)
    for xi, v in zip(xz + (i - 0.5) * w, q.rmse):
        cx.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=5.8)
cx.set_xticks(xz); cx.set_xticklabels([f"{z}\n{S.ZONE_NAME[z]}" for z in S.ZONES], fontsize=6.6)
cx.set_ylabel(f"RMSE ({S.UNIT})"); cx.set_ylim(0, 0.95)
cx.legend(handles=[Patch(color="#666666", alpha=0.45, label="2019"), Patch(color="#666666", label="2020")],
          loc="upper right", ncol=2, fontsize=6.3, handlelength=1.2, borderaxespad=0.1)
S.panel(cx, "c", x=-0.085, y=1.02, ha="left")
print(sr[["season", "rain", "rmse", "rel_rmse"]].round(3).to_string())
print(zq[["year", "zone", "rmse", "rel_rmse"]].round(3).to_string())
S.save(fig, "Fig07_error_attribution")
