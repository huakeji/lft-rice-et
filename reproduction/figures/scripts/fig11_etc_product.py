"""Fig. 11 Rice ETc product: example issue 15 July 2020 (day-1 and 7-day total) and 2020 verification."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import matplotlib as mpl

E = os.path.join(S.M, "10_etc_product")
p = pd.read_csv(os.path.join(E, "example_product_2020-07-15.csv"))
st = pd.read_csv(os.path.join(S.M, "12_figures/station_metrics_map.csv"))
st = st[st.year == 2020][["station", "lat", "lon"]]
p = p.merge(st, on="station", how="left")
assert p.lat.notna().all()
p["sum7"] = p[[f"etc_D{i}" for i in range(7)]].sum(axis=1)
bl = pd.read_csv(os.path.join(E, "etc_2020_by_lead.csv"))
bz = pd.read_csv(os.path.join(E, "etc_2020_by_zone.csv")).set_index("zone").reindex(S.ZONES)


def seq_cmap(name, n, lo=0.08, hi=0.95):
    c = [plt.get_cmap(name)(v) for v in np.linspace(lo, hi, n + 2)]
    cm = ListedColormap(c[1:-1]); cm.set_under(c[0]); cm.set_over(c[-1]); return cm


fig = plt.figure(figsize=(S.W2, 140 * S.MM))
x, y = S.proj_xy(p.lon, p.lat)
maps = [("a", p.etc_D0.values, np.arange(2, 10.01, 1.0), "YlGnBu",
         f"Day-1 {S.ETC} forecast ({S.UNIT})", "Issued 15 July 2020, lead 1"),
        ("b", p.sum7.values, np.arange(15, 60.01, 5.0), "YlOrRd",
         f"7-day cumulative {S.ETC} (mm)", "Issued 15 July 2020, leads 1–7")]
for k, (lab, v, bounds, cname, clabel, ttl) in enumerate(maps):
    cmap = seq_cmap(cname, len(bounds) - 1); norm = BoundaryNorm(bounds, cmap.N)
    ax = fig.add_axes([0.01 + 0.5 * k, 0.47, 0.47, 0.51], projection=S.albers())
    S.basemap(ax, fill="#FBFBFB"); S.gridlines(ax, size=6)
    o = np.argsort(v)
    ax.scatter(x[o], y[o], c=v[o], cmap=cmap, norm=norm, s=7, edgecolors="#333333", linewidths=0.15,
               zorder=3, transform=S.albers())
    S.panel(ax, lab, x=0.015, y=0.985, ha="left", va="top")
    ax.text(0.075, 0.983, ttl, transform=ax.transAxes, ha="left", va="top", fontsize=7)
    fig.canvas.draw(); pos_ = ax.get_position()
    S.scs_inset(fig, ax, w=0.15, h=0.27)
    cax = fig.add_axes([pos_.x0 + 0.1 * pos_.width, pos_.y0 - 0.048, 0.8 * pos_.width, 0.016])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend="both", extendfrac=0.05, ticks=bounds)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=2, labelsize=6.5, direction="in")
    cb.set_ticklabels([f"{t:g}" for t in bounds])
    cb.set_label(clabel, fontsize=7, labelpad=1.5)
    # inset histogram
    hx = fig.add_axes([pos_.x0 + 0.075 * pos_.width, pos_.y0 + 0.07 * pos_.height, 0.25 * pos_.width, 0.20 * pos_.height])
    cnt = np.histogram(v, bins=np.concatenate([[-np.inf], bounds, [np.inf]]))[0] / len(v) * 100
    cols = [cmap.get_under()] + [cmap(i) for i in range(cmap.N)] + [cmap.get_over()]
    hx.bar(np.arange(len(cnt)), cnt, width=0.85, color=cols, edgecolor="none")
    hx.set_xticks([]); hx.set_ylim(0, cnt.max() * 1.45); hx.patch.set_alpha(0)
    hx.set_ylabel("%", fontsize=6, labelpad=1); hx.tick_params(labelsize=5.5, length=1.5, pad=1)
    for s_ in ("left", "bottom"):
        hx.spines[s_].set_linewidth(0.4)
    hx.text(0.98, 0.98, f"median {np.median(v):.1f}", transform=hx.transAxes, ha="right", va="top", fontsize=5.8)
    print(lab, "n", len(v), "median", round(float(np.median(v)), 2), "mean", round(float(v.mean()), 2),
          "min", round(float(v.min()), 2), "max", round(float(v.max()), 2))

# (c) by lead
cx = fig.add_axes([0.075, 0.075, 0.38, 0.25]); cx2 = cx.twinx()
cx.bar(bl.lead, bl.rmse, 0.6, color="#8C8C8C", edgecolor="none", zorder=2, label="RMSE")
cx2.plot(bl.lead, bl.rel_rmse * 100, color="k", marker="o", ms=3.0, mfc="white", mew=0.8, lw=0.9, label="Relative RMSE")
for l_, v in zip(bl.lead, bl.rmse):
    cx.text(l_, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5.8)
cx.set_xticks(bl.lead); cx.set_xlabel("Lead (d)"); cx.set_ylabel(f"{S.ETC} RMSE ({S.UNIT})")
cx.set_ylim(0, 1.6); cx2.set_ylim(0, 40); cx2.set_ylabel("Relative RMSE (%)")
cx2.spines["right"].set_visible(True); cx2.spines["right"].set_linewidth(0.5); cx2.spines["top"].set_visible(False)
cx.legend(handles=[mpl.patches.Patch(color="#8C8C8C", label="RMSE"),
                   Line2D([], [], color="k", marker="o", ms=3.0, mfc="white", mew=0.8, lw=0.9, label="Relative RMSE")],
          loc="upper left", ncol=2, fontsize=6.3, borderaxespad=0.2, handlelength=1.4)
S.panel(cx, "c", x=-0.16, y=1.02, ha="left")

# (d) by rice zone
dx = fig.add_axes([0.585, 0.075, 0.36, 0.25]); dx2 = dx.twinx()
xz = np.arange(6)
dx.bar(xz, bz.rmse, 0.6, color=[S.ZONE_COL[z] for z in S.ZONES], edgecolor="none", zorder=2)
dx2.plot(xz, bz.rel_rmse * 100, ls="", marker="o", ms=3.2, mfc="white", mec="k", mew=0.8)
for xi, v, n in zip(xz, bz.rmse, bz.stations):
    dx.text(xi, 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=5.8, color="white")
dx.set_xticks(xz); dx.set_xticklabels([f"{z}\n({int(n)})" for z, n in zip(S.ZONES, bz.stations)], fontsize=6.5)
dx.tick_params(axis="x", length=0)
dx.set_ylim(0, 1.6); dx2.set_ylim(0, 40)
dx.set_ylabel(f"{S.ETC} RMSE ({S.UNIT})"); dx2.set_ylabel("Relative RMSE (%)")
dx2.spines["right"].set_visible(True); dx2.spines["right"].set_linewidth(0.5); dx2.spines["top"].set_visible(False)
dx.text(0.02, 0.97, "Rice zone (stations), all leads;  \u25CB relative RMSE (right axis)", transform=dx.transAxes, ha="left", va="top", fontsize=6.5)
S.panel(dx, "d", x=-0.16, y=1.02, ha="left")
print(bl.round(4).to_string()); print(bz[["stations", "rmse", "rel_rmse", "bias"]].round(4).to_string())
S.save(fig, "Fig11_etc_product")
