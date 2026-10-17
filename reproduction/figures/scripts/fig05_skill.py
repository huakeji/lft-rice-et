"""Fig. 5 Forecast skill vs climatology and persistence by lead (2020 solid, 2019 dashed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = pd.read_csv(os.path.join(S.M, "04_skill/skill_by_lead.csv"))
fig, axs = plt.subplots(1, 2, figsize=(S.W15, 66 * S.MM))
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.80, wspace=0.28)
for ax, col, lab, ttl in ((axs[0], "ss_clim", "a", "vs. station day-of-year climatology"),
                          (axs[1], "ss_persist", "b", "vs. persistence")):
    ax.axhline(0, color="#777777", lw=0.5, zorder=1)
    for m in S.MODEL_ORDER[::-1]:
        for y, ls in ((2020, "-"), (2019, "--")):
            r = d[(d.model == m) & (d.year == y)].sort_values("lead")
            ax.plot(r.lead, r[col], ls=ls, color=S.MODEL_COL[m], marker=S.MODEL_MK[m],
                    ms=3.2 if m == "LFT+GRU" else 2.6, mec="white", mew=0.25,
                    lw=1.5 if m == "LFT+GRU" else 0.8, zorder=5 if m == "LFT+GRU" else 3)
    ax.set_xticks(range(1, 8)); ax.set_xlim(0.7, 7.3)
    ax.set_xlabel("Lead (d)")
    ax.set_title(ttl, fontsize=7.5, pad=3)
    S.panel(ax, lab, x=-0.16, y=1.02, ha="left")
axs[0].set_ylabel("Skill score (1 − MSE/MSE$_\\mathrm{ref}$)")
axs[0].set_ylim(0, 0.7); axs[1].set_ylim(0, 0.7)
mh = [Line2D([], [], color=S.MODEL_COL[m], marker=S.MODEL_MK[m], mec="white", mew=0.25,
             lw=1.5 if m == "LFT+GRU" else 0.8, ms=3.0, label=m) for m in S.MODEL_ORDER]
yh = [Line2D([], [], color="k", lw=0.8, ls="-", label="2020"), Line2D([], [], color="k", lw=0.8, ls="--", label="2019")]
fig.legend(handles=mh + yh, loc="upper center", ncol=8, bbox_to_anchor=(0.53, 1.0), columnspacing=1.1,
           handlelength=2.0, fontsize=6.6)
# verify the skill definition from the file's own columns
r = d.iloc[0]
print("check ss_clim = 1-(rmse/rmse_clim)^2:", round(1 - (r.rmse / r.rmse_clim) ** 2, 4), round(r.ss_clim, 4))
print(d[d.model == "LFT+GRU"][["year", "lead", "ss_clim", "ss_persist"]].round(3).to_string())
S.save(fig, "Fig05_skill")
