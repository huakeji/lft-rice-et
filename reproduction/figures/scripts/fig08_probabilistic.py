"""Fig. 8 Probabilistic forecasts: coverage, PIT (quantile-interval bins), width/QS, peak-loss variants."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

U = os.path.join(S.M, "06_uncertainty")
rel = pd.read_csv(os.path.join(U, "reliability_by_lead.csv"))
pit = pd.read_csv(os.path.join(U, "pit_histogram.csv"))
lv = pd.read_csv(os.path.join(U, "loss_variants.csv"))
IC = {90: "#332288", 80: "#44AA99", 50: "#999933"}

fig = plt.figure(figsize=(S.W2, 112 * S.MM))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.42, left=0.065, right=0.99, top=0.94, bottom=0.10)

# (a) coverage
ax = fig.add_subplot(gs[0, 0])
for lvl in (90, 80, 50):
    ax.axhline(lvl, color=IC[lvl], lw=0.6, ls=(0, (4, 2)), alpha=0.9, zorder=1)
    for y, ls, mk in ((2020, "-", "o"), (2019, "--", "s")):
        q = rel[rel.year == y].sort_values("lead")
        ax.plot(q.lead, q[f"cover{lvl}"] * 100, ls=ls, marker=mk, color=IC[lvl], ms=2.8, mec="white", mew=0.25, lw=1.0)
ax.set_xticks(range(1, 8)); ax.set_xlim(0.7, 7.3); ax.set_ylim(40, 100)
ax.set_xlabel("Lead (d)"); ax.set_ylabel("Empirical coverage (%)")
for lvl, yt in ((90, 92.2), (80, 82.2), (50, 54.5)):
    ax.text(7.25, yt, f"{lvl}% interval", ha="right", va="bottom", fontsize=6.3, color=IC[lvl])
ax.legend(handles=[Line2D([], [], color="k", lw=0.9, ls="-", marker="o", ms=2.6, label="2020"),
                   Line2D([], [], color="k", lw=0.9, ls="--", marker="s", ms=2.6, label="2019"),
                   Line2D([], [], color="k", lw=0.6, ls=(0, (4, 2)), label="Nominal")],
          loc="lower left", ncol=3, fontsize=6.3, columnspacing=1.0, handlelength=2.0, borderaxespad=0.2)
S.panel(ax, "a", x=-0.13, y=1.02, ha="left")

# (b) PIT over quantile intervals vs nominal probabilities
bx = fig.add_subplot(gs[0, 1])
nominal = np.array([0.05, 0.05, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])
labs = ["<P5", "P5–\nP10", "P10–\nP25", "P25–\nP50", "P50–\nP75", "P75–\nP90", "P90–\nP95", ">P95"]
xb = np.arange(8); w = 0.38
for i, (ld, col) in enumerate(((1, "#9ECAE1"), (7, "#3182BD"))):
    q = pit[(pit.year == 2020) & (pit.lead == ld)].sort_values("bin")
    bx.bar(xb + (i - 0.5) * w, q.freq * 100, w * 0.94, color=col, edgecolor="none", label=f"Lead {ld}", zorder=2)
for xi, nv in zip(xb, nominal):
    bx.plot([xi - 0.46, xi + 0.46], [nv * 100] * 2, color="k", lw=1.1, zorder=3)
bx.set_xticks(xb); bx.set_xticklabels(labs, fontsize=6.2)
bx.set_xlim(-0.6, 7.6); bx.set_ylim(0, 31)
bx.set_xlabel("Interval between predicted quantiles containing the observation", labelpad=2)
bx.set_ylabel("Frequency (%)")
h, l = bx.get_legend_handles_labels()
bx.legend(h + [Line2D([], [], color="k", lw=1.1)], l + ["Nominal probability"], loc="upper left",
          fontsize=6.3, handlelength=1.4, borderaxespad=0.2)
bx.text(0.98, 0.98, "2020", transform=bx.transAxes, ha="right", va="top", fontsize=7)
S.panel(bx, "b", x=-0.13, y=1.02, ha="left")

# (c) 90% interval width and mean quantile score QS (Eq. 5) = mean pinball loss = crps_approx / 2
cx = fig.add_subplot(gs[1, 0]); cx2 = cx.twinx()
for y, ls, mk in ((2020, "-", "o"), (2019, "--", "s")):
    q = rel[rel.year == y].sort_values("lead")
    cx.plot(q.lead, q.width90, ls=ls, marker=mk, color="#332288", ms=2.8, mec="white", mew=0.25, lw=1.0)
    cx2.plot(q.lead, q.crps_approx / 2, ls=ls, marker=mk, color="#CC6677", ms=2.8, mec="white", mew=0.25, lw=1.0)
cx.set_xticks(range(1, 8)); cx.set_xlim(0.7, 7.3)
cx.set_xlabel("Lead (d)")
cx.set_ylabel(f"90% interval width ({S.UNIT})", color="#332288")
cx2.set_ylabel(f"Quantile score, QS ({S.UNIT})", color="#CC6677", labelpad=2)
cx.set_ylim(1.6, 2.7); cx2.set_ylim(0.12, 0.185)
cx.tick_params(axis="y", colors="#332288"); cx2.tick_params(axis="y", colors="#CC6677")
cx2.spines["right"].set_visible(True); cx2.spines["right"].set_linewidth(0.5); cx2.spines["top"].set_visible(False)
cx.legend(handles=[Line2D([], [], color="#332288", lw=1.0, label="90% width"),
                   Line2D([], [], color="#CC6677", lw=1.0, label="QS"),
                   Line2D([], [], color="k", lw=0.9, ls="-", marker="o", ms=2.6, label="2020"),
                   Line2D([], [], color="k", lw=0.9, ls="--", marker="s", ms=2.6, label="2019")],
          loc="upper left", ncol=2, fontsize=6.3, columnspacing=1.0, handlelength=2.0, borderaxespad=0.2)
S.panel(cx, "c", x=-0.13, y=1.02, ha="left")

# (d) bias on top-10% ET0 days for loss variants
dx = fig.add_subplot(gs[1, 1])
var = [("mse", "MSE"), ("quant", "Quantile\n(pinball)"), ("peak1", "Peak-weighted\nweight ×2"),
       ("peak3", "Peak-weighted\nweight ×4")]
xv = np.arange(4); w = 0.36
YC = {2019: "#BDBDBD", 2020: "#525252"}
for i, y in enumerate((2019, 2020)):
    for j, (k, _) in enumerate(var):
        r = lv[lv.run == f"{k}_y{y}"].iloc[0]
        xi = xv[j] + (i - 0.5) * w
        dx.bar(xi, r.top10_bias, w * 0.92, color=YC[y], edgecolor="none", zorder=2)
        dx.text(xi, r.top10_bias - 0.03, f"{r.test_rmse:.3f}", ha="center", va="top", fontsize=5.8, rotation=90)
dx.axhline(0, color="k", lw=0.5)
dx.set_xticks(xv); dx.set_xticklabels([v for _, v in var], fontsize=6.4)
dx.set_ylim(-1.32, 0.05)
dx.set_ylabel(f"Bias, top-10% {S.ET0} days ({S.UNIT})")
dx.legend(handles=[Patch(color=YC[2019], label="2019"), Patch(color=YC[2020], label="2020")],
          loc="lower right", ncol=2, fontsize=6.3, handlelength=1.2, borderaxespad=0.2)
dx.text(0.02, 0.03, "Numbers: overall test RMSE (" + S.UNIT + ")", transform=dx.transAxes, fontsize=6.0,
        ha="left", va="bottom")
dx.tick_params(axis="x", length=0)
S.panel(dx, "d", x=-0.15, y=1.02, ha="left")
print(lv.round(4).to_string())
S.save(fig, "Fig08_probabilistic")
