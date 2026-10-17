"""Fig. 1 Study area: stations by rice zone + EC paddy sites; station counts per zone."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

st = pd.read_csv(os.path.join(S.M, "12_figures/station_metrics_map.csv"))
st = st.drop_duplicates("station")
# station 58326 (Chaohu, R2) has no test-year data and is absent from the metrics table; coordinates from its shard
st = pd.concat([st, pd.DataFrame([dict(station=58326, lat=31.5833, lon=117.8333, rice_zone="R2")])], ignore_index=True)
zones = pd.read_csv(os.path.join(S.R, "station_rice_zone.csv"))  # 583 stations, counts per zone
st = st.drop(columns=[c for c in ("rice",) if c in st.columns]).merge(zones[["station", "rice"]], on="station", how="left")
print("stations on map", len(st), "rice", int(st.rice.sum()))
counts = zones.rice_zone.value_counts().reindex(S.ZONES)

fig = plt.figure(figsize=(S.W2, 108 * S.MM))
ax = fig.add_axes([0.03, 0.06, 0.74, 0.90], projection=S.albers())
S.basemap(ax)
S.gridlines(ax)
x, y = S.proj_xy(st.lon, st.lat)
for z in S.ZONES:
    m = (st.rice_zone == z).values; r = (st.rice == 1).values
    ax.scatter(x[m & r], y[m & r], s=4.5, c=S.ZONE_COL[z], edgecolors="none", zorder=3, transform=S.albers())
    ax.scatter(x[m & ~r], y[m & ~r], s=4.5, facecolors="white", edgecolors=S.ZONE_COL[z], linewidths=0.5, zorder=3,
               transform=S.albers())
ex, ey = S.proj_xy([s[2] for s in S.EC_SITES], [s[1] for s in S.EC_SITES])
ax.scatter(ex, ey, marker="*", s=55, c="k", edgecolors="white", linewidths=0.4, zorder=5,
           transform=S.albers(), label="Paddy EC site")
offs = {"Jurong": (-34, -8), "Changling": (12, 6), "Panjin": (10, -9)}
for (n, la, lo), xx, yy in zip(S.EC_SITES, ex, ey):
    dx, dy = offs[n]
    ax.annotate(n, (xx, yy), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                ha="left", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.4, color="k", shrinkA=0, shrinkB=2),
                path_effects=S.halo())
from matplotlib.lines import Line2D
hs = [Line2D([], [], ls="", marker="o", ms=3.6, mfc=S.ZONE_COL[z], mec="none", label=f"{z} {S.ZONE_NAME[z]}")
      for z in S.ZONES] + [Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec="#555555", mew=0.6, label="Non-rice station"),
                           Line2D([], [], ls="", marker="*", ms=7, mfc="k", mec="white", mew=0.3, label="Paddy EC site")]
leg = ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.005, 0.005), fontsize=6.5,
                handletextpad=0.3, labelspacing=0.35, frameon=True, framealpha=0.9,
                edgecolor="none", title="Rice zone", title_fontsize=6.8)
leg._legend_box.align = "left"
S.panel(ax, "a", x=0.012, y=0.988, ha="left", va="top")
fig.canvas.draw()
S.scs_inset(fig, ax, w=0.15, h=0.27)

# (b) counts per zone
bx = fig.add_axes([0.835, 0.20, 0.155, 0.60])
yy = np.arange(len(S.ZONES))[::-1]
bx.barh(yy, counts.values, color=[S.ZONE_COL[z] for z in S.ZONES], height=0.65, edgecolor="none")
for yi, v in zip(yy, counts.values):
    bx.text(v + 3, yi, f"{v}", va="center", fontsize=6.8)
bx.set_yticks(yy)
bx.set_yticklabels(S.ZONES)
bx.set_xlim(0, 205)
bx.set_xlabel("Number of stations")
bx.tick_params(axis="y", length=0)
bx.set_title(f"Total {counts.sum()} stations", fontsize=7, pad=4)
S.panel(bx, "b", x=-0.30, y=1.08, ha="left", va="bottom")
S.save(fig, "Fig01_study_area")
print("stations mapped:", len(st), "counts:", counts.to_dict(), "total", counts.sum())
