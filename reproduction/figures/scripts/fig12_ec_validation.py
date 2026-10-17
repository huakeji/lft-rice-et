"""Fig. 12 Paddy eddy-covariance evaluation of forecast actual ET (Kc_EC x forecast ET0), forward-only calibration."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

V = os.path.join(S.M, "14_ec_validation")
# forward-only design: Kc_EC of each site-year calibrated on earlier years of the same site only
d = pd.read_csv(os.path.join(V, "actual_et_forward_daily.csv"), parse_dates=["date"]).rename(columns={"eta": "eta_fcst", "kc": "kc_ec"})
B = pd.read_csv(os.path.join(V, "actual_et_forward_by_lead.csv")).rename(columns={"rmse": "rmse_fcst", "ss_clim": "ss_vs_clim"})
bl = B[B.site != "All"]; pooled = B[B.site == "All"].rename(columns={"rmse_fcst": "rmse", "ss_vs_clim": "ss_clim"})
SITES = ["Jurong", "Changling", "Panjin"]
SC = {"Jurong": "#1B7837", "Changling": "#762A83", "Panjin": "#B35806"}
SM = {"Jurong": "o", "Changling": "s", "Panjin": "^"}
print("kc_ec unique per site-year:", d.groupby(["site", "year"]).kc_ec.nunique().to_dict())

fig = plt.figure(figsize=(S.W2, 165 * S.MM))
gs = fig.add_gridspec(4, 3, height_ratios=[0.62, 0.62, 0.62, 1.15], hspace=0.62, wspace=0.34,
                      left=0.065, right=0.99, top=0.965, bottom=0.06)
for i, site in enumerate(SITES):
    ax = fig.add_subplot(gs[i, :])
    yrs = d[(d.site == site) & (d.lead == 1)].groupby("year").size()
    yr = int(yrs.idxmax())
    q = d[(d.site == site) & (d.year == yr)]
    rng = pd.date_range(q.date.min(), q.date.max(), freq="D")
    l1 = q[q.lead == 1].set_index("date").reindex(rng)
    l7 = q[q.lead == 7].set_index("date").reindex(rng)
    ax.plot(rng, l1.et, ls="", marker="o", ms=2.0, color="k", label="EC ET (QC-passed)", zorder=4)
    ax.plot(rng, l1.et, color="k", lw=0.35, alpha=0.5, zorder=3)
    ax.plot(rng, l1.eta_fcst, color=SC[site], lw=1.1, marker="o", ms=1.3, label=r"Forecast ET$_\mathrm{a}$, lead 1", zorder=5)
    ax.plot(rng, l7.eta_fcst, color=SC[site], lw=0.6, ls=(0, (3, 1.5)), alpha=0.9,
            label=r"Forecast ET$_\mathrm{a}$, lead 7", zorder=5)
    ax.set_ylabel(f"ET ({S.UNIT})"); ax.set_ylim(0, 11); ax.yaxis.set_major_locator(plt.MultipleLocator(2.5))
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[10, 20]))
    ax.set_xlim(rng[0] - pd.Timedelta(days=1), rng[-1] + pd.Timedelta(days=1))
    r1 = q[q.lead == 1]
    rm = np.sqrt(np.mean((r1.eta_fcst - r1.et) ** 2)); bi = np.mean(r1.eta_fcst - r1.et)
    ax.text(0.0, 1.02, f"{site} {yr}  (n = {len(r1)} days; lead-1 RMSE {rm:.2f}, bias " + f"{bi:+.2f}".replace("-", "\u2212") + f" {S.UNIT})",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    S.panel(ax, "abc"[i], x=-0.055, y=1.02, ha="left", va="bottom")
    if i == 0:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=6.4, borderaxespad=0.1,
                  handlelength=2.0, columnspacing=1.2)
    print(site, yr, "days", len(r1), "rmse", round(rm, 3), "bias", round(bi, 3))

# (d) scatter EC vs lead-1 forecast, all sites/years
dx = fig.add_subplot(gs[3, 0])
q1 = d[d.lead == 1]
for site in SITES:
    s_ = q1[q1.site == site]
    dx.scatter(s_.eta_fcst, s_.et, s=4, marker=SM[site], c=SC[site], alpha=0.6, edgecolors="none", label=site)
lim = (0, 10.5)
dx.plot(lim, lim, color="k", lw=0.6, ls="--")
dx.set_xlim(lim); dx.set_ylim(lim); dx.set_aspect("equal"); dx.set_xticks(range(0, 11, 2)); dx.set_yticks(range(0, 11, 2))
dx.set_xlabel(f"Forecast ET$_\\mathrm{{a}}$, lead 1 ({S.UNIT})"); dx.set_ylabel(f"EC ET ({S.UNIT})")
e = q1.eta_fcst - q1.et
st_txt = (f"n = {len(q1)}\nRMSE = {np.sqrt((e ** 2).mean()):.2f}\nbias = {e.mean():+.2f}\n"
          f"r = {np.corrcoef(q1.et, q1.eta_fcst)[0, 1]:.2f}").replace("-", "−")
dx.text(0.03, 0.97, st_txt, transform=dx.transAxes, ha="left", va="top", fontsize=6.5, linespacing=1.15)
dx.legend(loc="lower right", fontsize=6.2, markerscale=1.8, handletextpad=0.2, borderaxespad=0.2)
S.panel(dx, "d", x=-0.25, y=1.02, ha="left")
print("pooled lead1:", st_txt.replace("\n", "; "))

# (e) RMSE by lead, forecast vs climatology
ex = fig.add_subplot(gs[3, 1])
for site in SITES:
    b = bl[bl.site == site].sort_values("lead")
    ex.plot(b.lead, b.rmse_fcst, color=SC[site], marker=SM[site], ms=3.0, mec="white", mew=0.25, lw=1.0)
    ex.plot(b.lead, b.rmse_clim, color=SC[site], lw=0.8, ls=(0, (3, 1.5)))
ex.set_xticks(range(1, 8)); ex.set_xlim(0.7, 7.3); ex.set_ylim(1.0, 1.9)
ex.set_xlabel("Lead (d)"); ex.set_ylabel(f"RMSE vs EC ET ({S.UNIT})")
ex.legend(handles=[Line2D([], [], color="k", lw=1.0, marker="o", ms=2.6, label="Forecast"),
                   Line2D([], [], color="k", lw=0.8, ls=(0, (3, 1.5)), label="Climatology")]
          + [Line2D([], [], color=SC[s], lw=1.0, marker=SM[s], ms=2.6, label=s) for s in SITES],
          loc="upper left", ncol=3, fontsize=6.0, handlelength=1.8, columnspacing=0.6, borderaxespad=0.1)
S.panel(ex, "e", x=-0.25, y=1.02, ha="left")

# (f) skill vs climatology by lead
fx = fig.add_subplot(gs[3, 2])
fx.axhline(0, color="#777777", lw=0.5)
for site in SITES:
    b = bl[bl.site == site].sort_values("lead")
    fx.plot(b.lead, b.ss_vs_clim, color=SC[site], marker=SM[site], ms=3.0, mec="white", mew=0.25, lw=1.0, label=site)
fx.plot(pooled.lead, pooled.ss_clim, color="k", lw=1.4, marker="o", ms=3.0, mec="white", mew=0.25, label="All sites")
fx.set_xticks(range(1, 8)); fx.set_xlim(0.7, 7.3); fx.set_ylim(-0.05, 0.6)
fx.set_xlabel("Lead (d)"); fx.set_ylabel("Skill score vs climatology")
fx.legend(loc="upper right", fontsize=6.2, ncol=2, handlelength=1.8, columnspacing=0.8, borderaxespad=0.1)
S.panel(fx, "f", x=-0.25, y=1.02, ha="left")
print(bl[["site", "lead", "rmse_fcst", "rmse_clim", "ss_vs_clim"]].round(3).to_string())
print(pooled[["lead", "rmse", "ss_clim"]].round(3).to_string())
S.save(fig, "Fig12_ec_validation")
