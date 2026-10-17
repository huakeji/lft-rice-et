"""Fig. S1 Hyper-parameter search: validation RMSE (2019, outer fold 2020, 195-station subset) per config."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rows = []
for f in sorted(glob.glob(os.path.join(S.R, "hpsearch/*/*/result.json"))):
    r = json.load(open(f)); cfg = os.path.basename(os.path.dirname(f))
    rows.append(dict(model=r["model"].upper(), cfg=cfg, val=r["val_rmse"]))
d = pd.DataFrame(rows)
fz = pd.read_csv(os.path.join(S.M, "02_models/frozen_configs.csv"))
frozen = dict(zip(fz.model.str.upper(), fz.frozen_from))
order = ["LFT", "GRU", "PFT", "MLP", "CBM"]

fig, ax = plt.subplots(figsize=(S.W15, 70 * S.MM))
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.90)
rng = np.random.default_rng(0)
for i, m in enumerate(order):
    q = d[d.model == m].sort_values("cfg")
    jit = (np.arange(len(q)) - (len(q) - 1) / 2) * 0.045
    ax.scatter(i + jit, q.val, s=14, c=S.MODEL_COL[m], alpha=0.75, edgecolors="none", zorder=3)
    sel = q[q.cfg == frozen[m]]
    ax.scatter(i + jit[q.cfg.values == frozen[m]], sel.val, s=48, facecolors="none", edgecolors="k", linewidths=0.9, zorder=4)
    best = q.val.min()
    ax.text(i + 0.32, best, f"best {best:.4f}\nfrozen {sel.val.iloc[0]:.4f}", fontsize=5.8, va="center", ha="left")
    print(m, "n configs", len(q), "best", q.loc[q.val.idxmin(), "cfg"], round(best, 4), "frozen", frozen[m], round(sel.val.iloc[0], 4))
ax.set_xticks(range(len(order))); ax.set_xticklabels(order); ax.set_xlim(-0.5, len(order) - 0.1)
ax.tick_params(axis="x", length=0)
ax.set_ylabel(f"Validation RMSE, 2019 ({S.UNIT})")
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, mfc="#777777", mec="none", label="Candidate configuration"),
                   Line2D([], [], ls="", marker="o", ms=6.5, mfc="none", mec="k", mew=0.9, label="Frozen configuration")],
          loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, fontsize=6.6)
S.save(fig, "FigS1_hpsearch")
