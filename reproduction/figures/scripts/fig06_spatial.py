"""Fig. 6 Spatial patterns (2020): station RMSE of LFT+GRU and relative RMSE reduction vs GRU."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib as mpl

st = pd.read_csv(os.path.join(S.M, "12_figures/station_metrics_map.csv"))
st = st[st.year == 2020].copy()
x, y = S.proj_xy(st.lon, st.lat)

# colour schemes
b_rmse = np.round(np.arange(0.45, 1.0501, 0.05), 2)
base = plt.get_cmap("RdYlBu_r")
c = [base(v) for v in np.linspace(0.0, 1.0, len(b_rmse) + 1)]
cm_rmse = ListedColormap(c[1:-1]); cm_rmse.set_under(c[0]); cm_rmse.set_over(c[-1])
b_gain = np.arange(-4, 8.01, 1.0)
neg = [plt.get_cmap("RdBu_r")(v) for v in np.linspace(0.12, 0.42, 5)]      # 4 bins + under
pos = [plt.get_cmap("RdBu_r")(v) for v in np.linspace(0.56, 0.97, 9)]      # 8 bins + over
cm_gain = ListedColormap(neg[1:] + pos[:-1]); cm_gain.set_under(neg[0]); cm_gain.set_over(pos[-1])

fig = plt.figure(figsize=(S.W2, 86 * S.MM))
specs = [("a", st.rmse_ens.values, b_rmse, cm_rmse, f"RMSE of {S.ET0} ({S.UNIT})", 0.005),
         ("b", st.gain_ens_vs_gru_pct.values, b_gain, cm_gain, "Relative RMSE reduction vs GRU (%)", 0.505)]
for lab, v, bounds, cmap, clabel, x0 in specs:
    norm = BoundaryNorm(bounds, cmap.N)
    ax = fig.add_axes([x0 + 0.012, 0.17, 0.375, 0.80], projection=S.albers())
    S.basemap(ax, fill="#FBFBFB")
    S.gridlines(ax, size=6, left=True, bottom=True)
    o = np.argsort(np.abs(v - np.median(v)))  # extremes drawn on top
    ax.scatter(x[o], y[o], c=v[o], cmap=cmap, norm=norm, s=6.5, edgecolors="#333333", linewidths=0.15,
               zorder=3, transform=S.albers())
    S.panel(ax, lab, x=0.015, y=0.985, ha="left", va="top")
    fig.canvas.draw()
    pos_ = ax.get_position()
    S.scs_inset(fig, ax, w=0.15, h=0.27)
    # colorbar
    cax = fig.add_axes([pos_.x0 + 0.08 * pos_.width, pos_.y0 - 0.115, 0.84 * pos_.width, 0.024])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      extend="both", extendfrac=0.06, ticks=bounds[::2] if lab == "a" else bounds[::2])
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(width=0.4, length=2, labelsize=6.5, direction="in")
    cb.set_label(clabel, fontsize=7, labelpad=1.5)
    if lab == "b":
        cb.set_ticks(bounds[::2]); cb.set_ticklabels([f"{t:g}".replace("-", "−") for t in bounds[::2]])
    # inset histogram (lower-left)
    hx = fig.add_axes([pos_.x0 + 0.075 * pos_.width, pos_.y0 + 0.07 * pos_.height, 0.25 * pos_.width, 0.20 * pos_.height])
    edges = np.concatenate([[-np.inf], bounds, [np.inf]])
    cnt = np.histogram(v, bins=edges)[0] / len(v) * 100
    cols = [cmap.get_under()] + [cmap(i) for i in range(cmap.N)] + [cmap.get_over()]
    hx.bar(np.arange(len(cnt)), cnt, width=0.85, color=cols, edgecolor="none")
    hx.set_xticks([]); hx.set_ylim(0, cnt.max() * 1.45)
    hx.set_ylabel("%", fontsize=6, labelpad=1); hx.tick_params(labelsize=5.5, length=1.5, pad=1)
    hx.patch.set_alpha(0)
    for s_ in ("left", "bottom"):
        hx.spines[s_].set_linewidth(0.4)
    if lab == "a":
        hx.text(0.98, 0.98, f"median {np.median(v):.3f}", transform=hx.transAxes, ha="right", va="top", fontsize=5.8)
        print("RMSE 2020 station median %.4f mean %.4f min %.4f max %.4f" % (np.median(v), v.mean(), v.min(), v.max()))
    else:
        share = (v > 0).mean() * 100
        k0 = 1 + np.searchsorted(bounds, 0)  # first positive bin index in cnt
        hx.axvline(k0 - 0.5, color="k", lw=0.5, ls="--")
        hx.text(0.98, 0.98, f"improved {share:.1f}%", transform=hx.transAxes, ha="right", va="top", fontsize=5.8)
        print("gain >0 share %.2f%%, median %.3f, mean %.3f" % (share, np.median(v), v.mean()))
    # marginal latitude profile (zonal mean in 2-degree bands), aligned on the projected y of lon=105E
    px = fig.add_axes([pos_.x1 + 0.006, pos_.y0, 0.055, pos_.height])
    bands = np.arange(18, 54, 2.0)
    mids, mu, sd = [], [], []
    for b0 in bands:
        m = (st.lat >= b0) & (st.lat < b0 + 2)
        if m.sum() >= 3:
            mids.append(b0 + 1); mu.append(v[m.values].mean()); sd.append(v[m.values].std())
    mids, mu, sd = map(np.array, (mids, mu, sd))
    yy = S.lat_to_y(mids)
    px.fill_betweenx(yy, mu - sd, mu + sd, color="#BDBDBD", lw=0, alpha=0.6)
    px.plot(mu, yy, color="k", lw=0.9)
    if lab == "b":
        px.axvline(0, color="#777777", lw=0.4, ls="--")
    px.set_ylim(S.MAIN_EXTENT[2], S.MAIN_EXTENT[3])
    px.set_yticks(S.lat_to_y([20, 30, 40, 50])); px.set_yticklabels(["20\u00b0N", "30\u00b0N", "40\u00b0N", "50\u00b0N"], fontsize=5.8)
    px.spines["left"].set_visible(False); px.spines["right"].set_visible(True); px.spines["right"].set_linewidth(0.5)
    px.yaxis.tick_right(); px.tick_params(axis="y", direction="in", length=2, pad=1.5)
    px.tick_params(axis="x", labelsize=5.8, length=1.5, pad=1)
    px.xaxis.set_major_locator(plt.MaxNLocator(2))
    px.set_xlabel("Zonal\nmean", fontsize=6, labelpad=1)
S.save(fig, "Fig06_spatial_patterns")
