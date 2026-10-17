"""Fig. S3 Monthly EC-derived crop coefficient (EC ET / observed ET0) vs product Kc at the three paddy sites."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = pd.read_csv(os.path.join(S.M, "14_ec_validation/monthly_kc_ec_vs_product.csv"))
SITES = ["Jurong", "Changling", "Panjin"]
YC = {2018: "#999933", 2019: "#88CCEE", 2020: "#882255"}
YM = {2018: "o", 2019: "s", 2020: "^"}
MN = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
fig, axs = plt.subplots(1, 3, figsize=(S.W2, 62 * S.MM), sharey=True)
fig.subplots_adjust(left=0.065, right=0.99, bottom=0.14, top=0.86, wspace=0.12)
for ax, site, lab in zip(axs, SITES, "abc"):
    q = d[d.site == site]
    kp = q.groupby("month").kc_product.first()
    ax.plot(kp.index, kp.values, color="k", lw=1.4, marker="D", ms=3.0, label="Product K$_\\mathrm{c}$", zorder=4)
    for y in sorted(q.year.unique()):
        months = np.arange(q.month.min(), q.month.max() + 1)
        r = q[q.year == y].set_index("month").reindex(months)   # gaps (QC-removed months) break the line
        ax.plot(months, r.kc_ec, color=YC[y], marker=YM[y], ms=3.2, lw=0.9, mec="white", mew=0.3, label=str(y))
    ax.set_xticks(sorted(q.month.unique())); ax.set_xticklabels([MN[m] for m in sorted(q.month.unique())])
    ax.set_xlim(min(q.month) - 0.4, max(q.month) + 0.4)
    ax.set_title(site, fontsize=7.5, pad=3)
    S.panel(ax, lab, x=-0.08 if lab != "a" else -0.16, y=1.02, ha="left")
    print(site, "kc_ec range", round(q.kc_ec.min(), 3), round(q.kc_ec.max(), 3), "product", round(kp.min(), 3), round(kp.max(), 3))
axs[0].set_ylabel(r"K$_\mathrm{c}$ or ratio EC ET / observed ET$_\mathrm{o}$")
axs[0].set_ylim(0.3, 1.45)
hs = [Line2D([], [], color="k", lw=1.4, marker="D", ms=3.0, label="Product K$_\\mathrm{c}$")] + \
     [Line2D([], [], color=YC[y], marker=YM[y], ms=3.2, lw=0.9, label=f"EC ratio, {y}") for y in (2018, 2019, 2020)]
fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0), fontsize=6.6, columnspacing=1.4)
S.save(fig, "FigS3_kc_monthly")
