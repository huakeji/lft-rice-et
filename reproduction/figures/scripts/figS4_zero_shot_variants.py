"""Fig. S4 Leave-zone-out zero-shot variants vs the national pooled model (outer 2020).
National, standard zero-shot and untuned zero-shot: mean over seeds (whiskers = range; untuned R1, R2, R4 one seed);
physics-prior zero-shot: seed 0."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

G = os.path.join(S.M, "07_group_transfer")
zs = pd.read_csv(os.path.join(G, "zone_settings.csv")).set_index("zone").reindex(S.ZONES)
ms = pd.read_csv(os.path.join(G, "zone_multiseed.csv")).set_index("zone").reindex(S.ZONES)
sets = [("C", "National pooled (zone seen)", "#4D4D4D"), ("F", "Zero-shot (standard)", "#CC6677"),
        ("P", "Zero-shot, physics prior", "#44AA99"), ("U", "Zero-shot, untuned", "#DDCC77")]
fig, ax = plt.subplots(figsize=(S.W15, 66 * S.MM))
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.86)
xz = np.arange(6); w = 0.2
for i, (k, lab, col) in enumerate(sets):
    xi = xz + (i - 1.5) * w
    v = zs["P_zero_shot_phys"].values if k == "P" else ms[f"{k}_mean"].values
    ax.bar(xi, v, w * 0.92, color=col, edgecolor="none", label=lab, zorder=2)
    top = v.copy()
    if k != "P":
        lo, hi = ms[f"{k}_min"].values, ms[f"{k}_max"].values
        n = ms[f"{k}_n"].values if k != "C" else np.full(6, 3)
        m = n > 1
        ax.errorbar(xi[m], v[m], yerr=[v[m] - lo[m], hi[m] - v[m]], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=4)
        top = np.where(m, hi, v)
    if k != "C":
        for xx, vv, tt in zip(xi, v, top):
            ax.text(xx, tt + 0.03, f"{vv:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
ax.set_xticks(xz); ax.set_xticklabels([f"{z}\n{S.ZONE_NAME[z]}" for z in S.ZONES], fontsize=6.5)
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1.45); ax.set_ylabel(f"RMSE ({S.UNIT})")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=6.4, columnspacing=1.0, handlelength=1.2)
print(ms[["C_mean", "F_mean", "U_n", "U_mean", "U_min", "U_max"]].round(3).to_string()); print(zs.P_zero_shot_phys.round(3).to_dict())
S.save(fig, "FigS4_zero_shot_variants")
