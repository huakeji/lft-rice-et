"""Fig. S5 Mean difference Kc x observed ET0 minus EC ET on matched days, EC as measured vs closure-corrected.
Source: 14_ec_validation/closure_matched.csv (Jurong, Changling: the same days for both flux versions) and
tableS2_advection_sensitivity.csv (Panjin: distributed fluxes, already energy-balance corrected by the provider)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

E = os.path.join(S.M, "14_ec_validation")
cm = pd.read_csv(os.path.join(E, "closure_matched.csv")).set_index(["site", "flux"])
t2 = pd.read_csv(os.path.join(E, "tableS2_advection_sensitivity.csv"))
pj = t2[(t2.advective_days == "retained") & (t2.site == "Panjin")].iloc[0]
fig, ax = plt.subplots(figsize=(S.W1, 64 * S.MM))
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.84)
ax.axhline(0, color="k", lw=0.5)
w = 0.36
bars = [(0 - w / 2, cm.loc[("Jurong", "as measured")], "#BDBDBD"), (0 + w / 2, cm.loc[("Jurong", "closure-corrected")], "#4D4D4D"),
        (1 - w / 2, cm.loc[("Changling", "as measured")], "#BDBDBD"), (1 + w / 2, cm.loc[("Changling", "closure-corrected")], "#4D4D4D")]
for xi, r, col in bars:
    ax.bar(xi, r.bias, w * 0.92, color=col, edgecolor="none", zorder=2)
    ax.text(xi, max(r.bias, 0) + 0.04, f"{r.bias:+.2f}".replace("-", "−"), ha="center", va="bottom", fontsize=6)
ax.bar(2, pj.obs_bias, w * 0.92, color="#8C6D9E", edgecolor="none", zorder=2)
ax.text(2, pj.obs_bias + 0.04, f"{pj.obs_bias:+.2f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([f"Jurong\n(n = {int(cm.loc[('Jurong', 'as measured')].n)})",
                                              f"Changling\n(n = {int(cm.loc[('Changling', 'as measured')].n)})",
                                              f"Panjin\n(n = {int(pj.n)})"], fontsize=6.6)
ax.set_xlim(-0.6, 2.6); ax.tick_params(axis="x", length=0); ax.set_ylim(-0.2, 2.0)
ax.set_ylabel(r"K$_\mathrm{c}\times$ET$_\mathrm{o}$(obs.) $-$ EC ET (" + S.UNIT + ")")
ax.legend(handles=[Patch(color="#BDBDBD", label="EC as measured"), Patch(color="#4D4D4D", label="Closure-corrected (same days)"),
                   Patch(color="#8C6D9E", label="Distributed, provider-corrected")],
          loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=2, fontsize=6.0, handlelength=1.2, columnspacing=0.8)
print(cm.round(3)); print("Panjin", pj.n, pj.obs_bias)
S.save(fig, "FigS5_ec_closure_bias")
