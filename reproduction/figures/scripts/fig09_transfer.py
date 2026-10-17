"""Fig. 9 Pooled learning and transfer: scaling, zone settings, few-shot recovery."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

G = os.path.join(S.M, "07_group_transfer")
zs = pd.read_csv(os.path.join(G, "zone_settings.csv")).set_index("zone")

fig = plt.figure(figsize=(S.W2, 122 * S.MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.55, wspace=0.30,
                      left=0.065, right=0.99, top=0.93, bottom=0.085)

# (a) scaling: three random 99-station holdouts, increase relative to the national model on the same stations
ax = fig.add_subplot(gs[0, 0])
sd = pd.read_csv(os.path.join(G, "scaling_draws.csv")); sa = pd.read_csv(os.path.join(G, "scaling_draws_agg.csv"))
ax.fill_between(sa.train_stations, sa.min_rel, sa.max_rel, color="#BBBBBB", alpha=0.45, lw=0, zorder=1)
for dr, ls_ in zip(["s0", "d8", "d9"], ["-", "--", ":"]):
    q = sd[sd.draw == dr]
    ax.plot(q.train_stations, q.rel_pct, color="#777777", ls=ls_, lw=0.7, zorder=2)
ax.plot(sa.train_stations, sa.mean_rel, color="k", marker="o", ms=3.4, mfc="white", mec="k", mew=0.8, lw=1.1, zorder=3)
for n, v in zip(sa.train_stations, sa.mean_rel):
    ax.annotate(f"{v:.1f}", (n, v), xytext=(5, 0) if n == 484 else (0, 4), textcoords="offset points", ha="left" if n == 484 else "center",
                va="center" if n == 484 else "bottom", fontsize=5.9)
ax.axhline(0, color="#D55E00", lw=0.8, ls="--")
ax.text(23, 0.4, "National model, stations seen", color="#D55E00", fontsize=6.0, va="bottom", ha="left")
ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 484]); ax.set_xticklabels([25, 50, 100, 200, 484])
ax.minorticks_off(); ax.set_xticks([400], minor=True)
ax.set_xlim(20, 600); ax.set_ylim(-1.5, 17.5)
ax.set_xlabel("Number of training stations")
ax.set_ylabel("Zero-shot RMSE increase (%)")
ax.legend([Line2D([], [], color="k", marker="o", ms=3, mfc="white", lw=1.1), Patch(color="#BBBBBB", alpha=0.45),
           Line2D([], [], color="#777777", lw=0.7)], ["Mean of 3 holdouts", "Range", "Individual holdouts"],
          loc="upper right", fontsize=5.9, handlelength=1.6, borderaxespad=0.2)
S.panel(ax, "a", x=-0.24, y=1.02, ha="left")

# (b) zone settings (national and zero-shot: mean of three seeds, whiskers = range)
ms = pd.read_csv(os.path.join(G, "zone_multiseed.csv")).set_index("zone").reindex(S.ZONES)
pv = pd.read_csv(os.path.join(G, "pooled_vs_national_3seed.csv")).set_index("zone").reindex(S.ZONES)
bx = fig.add_subplot(gs[0, 1:])
sets = [("C", "National pooled", "#4D4D4D"), ("B", "Zone-only", "#9E9E9E"),
        ("A", "Single-station", "#DADADA"), ("F", "Leave-zone-out zero-shot", "#CC6677")]
xz = np.arange(6); w = 0.2
for i, (k, lab, col) in enumerate(sets):
    xi = xz + (i - 1.5) * w
    v = {"C": ms.C_mean, "B": pv.B_zone, "A": pv.A_single, "F": ms.F_mean}[k].values
    bx.bar(xi, v, w * 0.92, color=col, edgecolor="#4D4D4D" if k == "A" else "none", linewidth=0.4, label=lab, zorder=2)
    if k in ("C", "F"):
        lo, hi = (ms.C_min, ms.C_max) if k == "C" else (ms.F_min, ms.F_max)
        bx.errorbar(xi, v, yerr=[v - lo.values, hi.values - v], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=4)
    if k == "A":
        bx.plot(xi, pv.C_same_mean, ls="", marker="o", ms=3.4, mfc="white", mec="k", mew=0.7, zorder=4,
                label="National pooled, single-station subset")
    if k == "F":
        for xx_, vv, hh in zip(xi, v, ms.F_max.values):
            bx.text(xx_, hh + 0.03, f"{vv:.2f}", ha="center", va="bottom", fontsize=5.6, rotation=90)
bx.set_xticks(xz); bx.set_xticklabels([f"{z}\n{S.ZONE_NAME[z]}" for z in S.ZONES], fontsize=6.4)
bx.tick_params(axis="x", length=0)
bx.set_ylim(0, 1.42); bx.set_ylabel(f"RMSE ({S.UNIT})")
h, l = bx.get_legend_handles_labels()
order = [l.index(x) for x in ["National pooled", "Zone-only", "Single-station",
                              "National pooled, single-station subset", "Leave-zone-out zero-shot"]]
bx.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", ncol=3, fontsize=6.2, handlelength=1.2,
          columnspacing=1.0, borderaxespad=0.2)
S.panel(bx, "b", x=-0.08, y=1.02, ha="left")

# (c) few-shot recovery
M_ = [("full", "Full fine-tuning", "#332288"), ("head", "Head only", "#117733"),
      ("norm", "Norm/bias", "#CC6677"), ("affine", "Affine", "#999933")]
xd = np.arange(4); xl = ["0\n(zero-shot)", "30", "90", "365"]
def series(z, m):
    return [zs.loc[z, "F_zero_shot"]] + [zs.loc[z, f"{m}_{d}d"] for d in (30, 90, 365)]
cax = []
for j, z in enumerate(["R3", "R6"]):
    c = fig.add_subplot(gs[1, j]); cax.append(c)
    for m, lab, col in M_:
        c.plot(xd, series(z, m), color=col, marker="o", ms=2.8, mec="white", mew=0.25, lw=1.0, label=lab)
    c.axhspan(ms.loc[z, "C_min"], ms.loc[z, "C_max"], color="#BBBBBB", alpha=0.5, lw=0, zorder=0)
    c.axhline(ms.loc[z, "C_mean"], color="k", lw=0.7, ls="--")
    c.text(-0.2, ms.loc[z, "C_max"] + 0.006, "National pooled", ha="left", va="bottom", fontsize=6.0)
    c.set_title(f"{z} {S.ZONE_NAME[z]}", fontsize=7.5, pad=3)
    c.set_xticks(xd); c.set_xticklabels(xl); c.set_xlim(-0.25, 3.25)
    c.set_xlabel("Issue days of local data per station")
    c.set_ylim(0.66, 1.24)
c0 = cax[0]; c0.set_ylabel(f"RMSE ({S.UNIT})")
c0.legend(loc="upper right", fontsize=6.2, handlelength=1.6, borderaxespad=0.2)
S.panel(c0, "c", x=-0.2, y=1.02, ha="left")
c = fig.add_subplot(gs[1, 2])
for z in ["R1", "R2", "R4", "R5"]:
    c.plot(xd, series(z, "full"), color=S.ZONE_COL[z], marker="o", ms=2.8, mec="white", mew=0.25, lw=1.0, label=z)
    c.plot([-0.2, 3.2], [ms.loc[z, "C_mean"]] * 2, color=S.ZONE_COL[z], lw=0.6, ls="--")
c.set_title("Other zones, full fine-tuning", fontsize=7.5, pad=3)
c.set_xticks(xd); c.set_xticklabels(xl); c.set_xlim(-0.25, 3.25)
c.set_xlabel("Issue days of local data per station")
c.set_ylim(0.665, 0.85)
h, l = c.get_legend_handles_labels()
h += [Line2D([], [], color="#777777", lw=0.6, ls="--")]; l += ["National"]
c.legend(h, l, loc="lower left", ncol=5, fontsize=6.0, handlelength=1.4, columnspacing=0.6, handletextpad=0.3,
         borderaxespad=0.1)
print(ms[["C_mean", "F_mean", "F_min", "F_max"]].round(3).to_string()); print(pv.round(3).to_string()); print(sa.round(2).to_string())
S.save(fig, "Fig09_transfer")
