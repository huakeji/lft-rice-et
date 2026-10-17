"""Fig. 4 Model comparison: RMSE by lead (2020, 2019) and paired difference vs GRU."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

lead = pd.read_csv(os.path.join(S.M, "03_main_results/lead.csv"))
lead["model"] = lead.model.str.upper()
paired = pd.read_csv(os.path.join(S.M, "03_main_results/paired.csv"))
paired["model"] = paired.model.str.upper()
L = [f"L{i}" for i in range(1, 8)]
ens = {}
for y in (2019, 2020):
    a = np.load(os.path.join(S.R, f"formal_summary/ens_lft_gru_y{y}.npz"))
    ens[y] = np.sqrt(np.mean((a["prediction"] - a["target"]) ** 2, axis=0))
    print(y, "ensemble per-lead RMSE", np.round(ens[y], 4))
import pandas as _pd
_pe=_pd.read_csv(os.path.join(S.R, "writing_materials/03_main_results/paired_ensemble.csv"))
_pe=_pe[_pe.comp=="ENS-GRU"].set_index("year")
ENS_DIFF = {y: float(_pe.loc[y,"diff"]) for y in (2020,2019,2018)}   # station-block bootstrap, paired_ensemble.csv
ENS_CI = {y: (float(_pe.loc[y,"lo"]), float(_pe.loc[y,"hi"])) for y in (2020,2019,2018)}

fig = plt.figure(figsize=(S.W2, 118 * S.MM))
gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.55, wspace=0.9,
                      left=0.065, right=0.99, top=0.90, bottom=0.09)
axa = fig.add_subplot(gs[0, 0:3]); axb = fig.add_subplot(gs[0, 3:6], sharey=axa)
lx = np.arange(1, 8)
for ax, y, lab in ((axa, 2020, "a"), (axb, 2019, "b")):
    for m in S.MODEL_ORDER:
        if m == "LFT+GRU":
            v = ens[y]; kw = dict(lw=1.7, ms=3.8, zorder=5)
        else:
            v = lead[(lead.model == m) & (lead.year == y)][L].values[0]; kw = dict(lw=0.9, ms=3.0, zorder=3)
        ax.plot(lx, v, color=S.MODEL_COL[m], marker=S.MODEL_MK[m], mec="white", mew=0.3, label=m, **kw)
    ax.set_xticks(lx); ax.set_xlim(0.6, 7.4)
    ax.set_xlabel("Lead (d)")
    ax.text(0.03, 0.96, f"Test year {y}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    S.panel(ax, lab, x=-0.10, y=1.02, ha="left")
axa.set_ylabel(f"RMSE of {S.ET0} ({S.UNIT})")
axa.set_ylim(0.6, 0.9)
plt.setp(axb.get_yticklabels(), visible=False)
h, l = axa.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.53, 1.0), columnspacing=1.6, fontsize=7)

# (c) paired difference vs GRU
models = ["LFT", "PFT", "MLP", "CBM"]
years = [(2020, "Test year 2020"), (2019, "Test year 2019"), (2018, "2018 (short history)")]
gb = gs[1, :].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.62)
axs = [fig.add_subplot(gb[0, i]) for i in range(3)]
ypos = {"LFT+GRU": 4, "LFT": 3, "PFT": 2, "MLP": 1, "CBM": 0}
for ax, (y, title) in zip(axs, years):
    ax.axvline(0, color="#777777", lw=0.5, ls="--", zorder=1)
    for m in models:
        r = paired[(paired.model == m) & (paired.year == y)].iloc[0]
        ax.errorbar(r["diff"], ypos[m], xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt=S.MODEL_MK[m],
                    color=S.MODEL_COL[m], ms=3.6, mec="white", mew=0.3, elinewidth=0.9, capsize=1.6, capthick=0.6, zorder=3)
    ax.errorbar(ENS_DIFF[y], ypos["LFT+GRU"], xerr=[[ENS_DIFF[y]-ENS_CI[y][0]],[ENS_CI[y][1]-ENS_DIFF[y]]], fmt="o", ms=4.2, mfc="k", mec="k", ecolor="k", elinewidth=1.0, capsize=2, zorder=4)
    ax.set_yticks(list(ypos.values())); ax.set_ylim(-0.6, 4.6)
    ax.set_yticklabels(list(ypos.keys()) if ax is axs[0] else [])
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xlabel(r"$\Delta$RMSE vs GRU (" + S.UNIT + ")")
    ax.tick_params(axis="y", length=0)
    lo = min(ENS_CI[y][0], paired[paired.year == y].ci_lo.min()); hi = max(ENS_CI[y][1], paired[paired.year == y].ci_hi.max())
    pad = 0.12 * (hi - lo); ax.set_xlim(min(lo - pad, -pad), hi + pad)
    ax.annotate(f"{ENS_DIFF[y]:+.4f}".replace("-", "\u2212"), (ENS_DIFF[y], ypos["LFT+GRU"]),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5)
axs[0].xaxis.set_major_locator(plt.MaxNLocator(4))
axs[1].xaxis.set_major_locator(plt.MaxNLocator(4))
S.panel(axs[0], "c", x=-0.32, y=1.06, ha="left")

# (d) control experiments and practical baselines, 2020, paired 2-degree spatial-block bootstrap vs GRU
pb = pd.read_csv(os.path.join(S.M, "16_ablation_baselines/paired_spatial_block_2020.csv"))
pb = pb[(pb.reference == "GRU") & (pb.leads == "all")].set_index("model")
dx = fig.add_subplot(gb[0, 3])
rows = [("LFT+GRU", "LFT+GRU", "k", "o", "k"), ("LFT", "LFT", S.MODEL_COL["LFT"], "s", S.MODEL_COL["LFT"]),
        ("GRU+lag", "GRU-lag", S.MODEL_COL["GRU"], "^", "white"), ("LFT-current", "LFT-current", S.MODEL_COL["LFT"], "s", "white"),
        ("HGB+lag", "HGB-lag", "#555555", "D", "#555555"), ("HGB", "HGB", "#555555", "D", "white"), ("HS-cal", "HS-cal", "#999999", "v", "#999999")]
XMAX = 0.034
dx.axvline(0, color="#777777", lw=0.5, ls="--", zorder=1)
for i, (k, lab, col, mk, fc) in enumerate(rows):
    r = pb.loc[k]; yy = len(rows) - 1 - i
    if r["diff"] > XMAX:
        dx.annotate("", xy=(XMAX * 0.99, yy), xytext=(XMAX * 0.70, yy), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=6))
        dx.text(XMAX * 0.66, yy, f"+{r['diff']:.3f}", ha="right", va="center", fontsize=6.2, color="#444444")
        continue
    dx.errorbar(r["diff"], yy, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt=mk, color=col, mfc=fc, mec=col, ms=3.6, mew=0.8,
                elinewidth=0.9, capsize=1.6, capthick=0.6, zorder=3)
dx.set_yticks(range(len(rows))); dx.set_yticklabels([r[1] for r in rows][::-1], fontsize=6.6)
dx.set_ylim(-0.6, len(rows) - 0.4); dx.set_xlim(-0.024, XMAX)
dx.tick_params(axis="y", length=0); dx.xaxis.set_major_locator(plt.MaxNLocator(4))
dx.set_title("2020: controls and baselines", fontsize=7.5, pad=3)
dx.set_xlabel(r"$\Delta$RMSE vs GRU (" + S.UNIT + ")")
S.panel(dx, "d", x=-0.50, y=1.06, ha="left")
S.save(fig, "Fig04_model_comparison")
