"""Fig. 10 Physical limits of transfer: zero-shot degradation vs altitude / climate novelty; map."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import matplotlib as mpl

P = os.path.join(S.M, "08_transfer_physics")
d = pd.read_csv(os.path.join(P, "zero_shot_station_descriptors_3seed.csv"))
rho = json.load(open(os.path.join(P, "spearman_degradation_3seed.json")))

fig = plt.figure(figsize=(S.W15, 140 * S.MM))
ax = fig.add_axes([0.085, 0.62, 0.38, 0.34]); bx = fig.add_axes([0.585, 0.62, 0.38, 0.34], sharey=ax)
for a, col, xl, key, lab in ((ax, "alt", "Station altitude (m)", "alt", "a"),
                             (bx, "novelty", "Climatic distance to the other zones", "novelty", "b")):
    a.axhline(0, color="#777777", lw=0.5, ls="--", zorder=1)
    for z in S.ZONES:
        q = d[d.zone == z]
        a.scatter(q[col], q.degradation_pct, s=5, c=S.ZONE_COL[z], edgecolors="none", alpha=0.85, zorder=3, label=z)
    a.set_xlabel(xl)
    a.text(0.03, 0.97, r"Spearman $\rho$ = " + f"{rho[key]:.2f}", transform=a.transAxes, ha="left", va="top", fontsize=7)
    S.panel(a, lab, x=-0.2, y=1.02, ha="left")
ax.set_xscale("log"); ax.set_xlim(1.5, 6000)
ax.set_xticks([10, 100, 1000]); ax.set_xticklabels(["10", "100", "1000"])
ax.set_ylabel("Zero-shot RMSE increase (%)")
ax.set_ylim(-45, 225)
plt.setp(bx.get_yticklabels(), visible=False)
bx.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.2, mfc=S.ZONE_COL[z], mec="none", label=z) for z in S.ZONES],
          loc="upper right", ncol=2, fontsize=6.3, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.2)

# (c) map of degradation
bounds = np.array([-20, -10, -5, 0, 5, 10, 20, 40, 80])
neg = [plt.get_cmap("RdBu_r")(v) for v in np.linspace(0.10, 0.42, 4)]
pos = [plt.get_cmap("RdBu_r")(v) for v in np.linspace(0.56, 0.98, 6)]
cmap = ListedColormap(neg[1:] + pos[:-1]); cmap.set_under(neg[0]); cmap.set_over(pos[-1])
norm = BoundaryNorm(bounds, cmap.N)
mx = fig.add_axes([0.085, 0.07, 0.78, 0.46], projection=S.albers())
mx.set_anchor("W")
S.basemap(mx, fill="#FBFBFB"); S.gridlines(mx, size=6)
x, y = S.proj_xy(d.lon, d.lat)
o = np.argsort(np.abs(d.degradation_pct.values))
mx.scatter(x[o], y[o], c=d.degradation_pct.values[o], cmap=cmap, norm=norm, s=6, edgecolors="#333333",
           linewidths=0.15, zorder=3, transform=S.albers())
S.panel(mx, "c", x=0.012, y=0.985, ha="left", va="top")
fig.canvas.draw()
pos_ = mx.get_position()
S.scs_inset(fig, mx, w=0.15, h=0.27)
cax = fig.add_axes([pos_.x1 + 0.02, pos_.y0 + 0.05 * pos_.height, 0.018, 0.9 * pos_.height])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both", extendfrac=0.05,
                  ticks=bounds, spacing="uniform")
cb.set_ticklabels([f"{t:g}".replace("-", "−") for t in bounds])
cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=2, labelsize=6.3, direction="in")
cb.set_label("Zero-shot RMSE increase (%)", fontsize=7)
zx = fig.add_axes([cax.get_position().x1 + 0.13, pos_.y0 + 0.25 * pos_.height, 0.10, 0.5 * pos_.height])
med = d.groupby("zone").degradation_pct.median().reindex(S.ZONES)
yz = np.arange(6)[::-1]
zx.barh(yz, med.values, color=[S.ZONE_COL[z] for z in S.ZONES], height=0.65, edgecolor="none")
zx.axvline(0, color="k", lw=0.5)
for yi, v in zip(yz, med.values):
    zx.text(max(v, 0) + 1.5, yi, f"{v:.1f}".replace("-", "\u2212"), va="center", ha="left", fontsize=6)
zx.set_yticks(yz); zx.set_yticklabels(S.ZONES); zx.tick_params(axis="y", length=0)
zx.set_xlim(-12, 45); zx.set_xlabel("Median increase (%)", fontsize=6.5)
zx.tick_params(axis="x", labelsize=6)
zx.set_title("Zone median", fontsize=7, pad=3)
print("rho", rho)
print(d.groupby("zone").degradation_pct.median().round(1).to_dict())
S.save(fig, "Fig10_transfer_physics")
