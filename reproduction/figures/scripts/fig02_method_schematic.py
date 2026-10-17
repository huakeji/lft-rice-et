"""Fig. 2 Method schematic: (a) time-lagged public forecasts; (b) LFT architecture. Pure matplotlib vector drawing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import afm_style as S
S.setup()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle

W, H = 190, 122
fig = plt.figure(figsize=(W * S.MM, H * S.MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
M = "−"
C_CUR, C_CUR_HI = "#FBD3BF", "#D55E00"      # current issue (row O)
C_OLD, C_OLD_HI = "#DCEAF5", "#0072B2"      # earlier issues
EDGE = "#4D4D4D"


def box(x, y, w, h, txt, fc="#FFFFFF", ec=EDGE, fs=6.6, lw=0.6, bold_title=None, ha="center", r=1.2, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    yy = y + h / 2
    if bold_title:
        ax.text(x + w / 2, y + h - 1.6, bold_title, ha="center", va="top", fontsize=fs + 0.3, fontweight="bold")
        yy = y + (h - 4.2) / 2
    if txt:
        ax.text(x + w / 2 if ha == "center" else x + 1.6, yy, txt, ha=ha, va="center", fontsize=fs,
                linespacing=1.25, **kw)


def arrow(x0, y0, x1, y1, lw=0.7, ms=6, color="k", **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, **kw))


# ---------------- (a) time-lagged forecast grid ----------------
ax.text(2, H - 2, "(a)", fontsize=8.5, fontweight="bold", ha="left", va="top")
ax.text(9, H - 2.3, "Time-lagged public forecasts available for the 7-day target window", fontsize=7.2,
        ha="left", va="top")
cw, ch = 9.6, 6.4
gx, gy = 25, 59           # lower-left of grid (row O at bottom)
T = 3                     # highlighted valid date T = O+3
for r in range(7):        # r = 0 -> issue O, r = 6 -> issue O-6
    for c in range(7):    # c = 0 -> valid O
        lead = c + r + 1
        x = gx + c * cw; y = gy + r * ch
        hi = (c == T)
        fc = (C_CUR_HI if hi else C_CUR) if r == 0 else (C_OLD_HI if hi else C_OLD)
        ax.add_patch(Rectangle((x, y), cw, ch, fc=fc, ec="white", lw=0.8))
        ax.text(x + cw / 2, y + ch / 2, f"{lead}", ha="center", va="center", fontsize=6.4,
                color="white" if hi else "#222222", fontweight="bold" if hi else "normal")
    lab = "O" if r == 0 else f"O{M}{r}"
    ax.text(gx - 1.5, gy + r * ch + ch / 2, lab, ha="right", va="center", fontsize=6.6)
for c in range(7):
    lab = "O" if c == 0 else f"O+{c}"
    ax.text(gx + c * cw + cw / 2, gy + 7 * ch + 1.2, lab, ha="center", va="bottom", fontsize=6.6)
ax.text(gx + 3.5 * cw, gy + 7 * ch + 5.6, "Valid (target) date", ha="center", va="bottom", fontsize=6.8)
ax.text(gx - 10.5, gy + 3.5 * ch, "Issue date", ha="center", va="center", rotation=90, fontsize=6.8)
# highlight column T
ax.add_patch(Rectangle((gx + T * cw, gy), cw, 7 * ch, fc="none", ec="k", lw=1.2, zorder=5))
ax.text(gx + T * cw + cw / 2, gy - 1.3, f"T = O+{T}", ha="center", va="top", fontsize=6.6, fontweight="bold")
# current-issue bracket
ax.add_patch(Rectangle((gx, gy), 7 * cw, ch, fc="none", ec=C_CUR_HI, lw=0.9, ls=(0, (3, 1.5)), zorder=4))
# explanatory callouts
kx = gx + 7 * cw + 6
ax.add_patch(Rectangle((kx, gy + 0.6 * ch), 4.2, 3.2, fc=C_CUR, ec="none"))
ax.text(kx + 5.6, gy + 0.6 * ch + 1.6, "Current issue O: leads 1–7 (lead 1 = issue day)", va="center", fontsize=6.4)
ax.add_patch(Rectangle((kx, gy + 2.1 * ch), 4.2, 3.2, fc=C_OLD, ec="none"))
ax.text(kx + 5.6, gy + 2.1 * ch + 1.6, f"Earlier issues O{M}1 … O{M}6 for the same days", va="center", fontsize=6.4)
ax.text(kx + 5.6, gy + 2.1 * ch - 1.6, "(raw messages carry 15 d, so leads up to 13 exist)", va="center",
        fontsize=6.0, color="#555555")
ax.add_patch(Rectangle((kx, gy + 3.9 * ch), 4.2, 3.2, fc="white", ec="k", lw=1.2))
ax.text(kx + 5.6, gy + 3.9 * ch + 1.6, f"Issue set for valid date T: lead k = {T + 1} from O,", va="center", fontsize=6.4)
ax.text(kx + 5.6, gy + 3.9 * ch - 1.6, f"leads k+1 … k+6 = {T + 2}…{T + 7} from O{M}1 … O{M}6",
        va="center", fontsize=6.4)
ax.text(kx, gy + 5.6 * ch, "Numbers in cells: lead (d) of that issue for that valid date", fontsize=6.2,
        va="center", color="#333333")

# baseline side box
bx0, by0, bw0, bh0 = kx, gy + 6.4 * ch, 76, 15.5
ax.add_patch(FancyBboxPatch((bx0, by0), bw0, bh0, boxstyle="round,pad=0,rounding_size=1.2", fc="#F5F5F5", ec="#9E9E9E", lw=0.6))
ax.text(bx0 + 1.8, by0 + bh0 - 1.5, "Baselines", fontsize=6.7, fontweight="bold", va="top", ha="left")
ax.text(bx0 + 1.8, by0 + bh0 - 5.2, "GRU, PFT (Transformer), MLP, CBM (CNN\u2013BiLSTM\u2013attention):\n"
        "current issue only (leads 1\u20137)\nOperational scheme: ensemble = mean(LFT, GRU)",
        fontsize=6.3, va="top", ha="left", linespacing=1.3)

# ---------------- (b) LFT architecture ----------------
ax.text(2, 50, "(b)", fontsize=8.5, fontweight="bold", ha="left", va="top")
ax.text(9, 49.7, "Lag-fusion Transformer (LFT)", fontsize=7.2, ha="left", va="top")
yb, hb = 20, 22
# 1 inputs
box(2, yb, 23, hb, f"Per valid day d\n(d = 1–7):\ncurrent issue O\n+ 6 earlier issues\nO{M}1 … O{M}6", bold_title=None, fs=6.3)
# 2 issue tokens
tx, tw = 29.5, 42.5
ax.add_patch(FancyBboxPatch((tx, yb), tw, hb, boxstyle="round,pad=0,rounding_size=1.2", fc="#FFF7F2", ec=C_CUR_HI, lw=0.7))
ax.text(tx + tw / 2, yb + hb - 1.4, "Issue tokens (7 per valid day)", ha="center", va="top", fontsize=6.6, fontweight="bold")
feat = [r"T$_\mathrm{max}$", r"T$_\mathrm{min}$", "rain\nlevel", r"$\Delta$T$_\mathrm{max}$", r"$\Delta$T$_\mathrm{min}$", "avail-\nability"]
fw = (tw - 4) / 6
for i, f in enumerate(feat):
    ax.add_patch(Rectangle((tx + 2 + i * fw + 0.3, yb + 10.2), fw - 0.6, 6.2, fc="white", ec=EDGE, lw=0.4))
    ax.text(tx + 2 + i * fw + fw / 2, yb + 13.3, f, ha="center", va="center", fontsize=5.6, linespacing=0.95)
ax.text(tx + tw / 2, yb + 6.6, "+ shared weather-text & wind-text embeddings", ha="center", va="center", fontsize=5.8)
ax.text(tx + tw / 2, yb + 3.0, "+ issue-age embedding", ha="center", va="center", fontsize=5.9)
# 3 pooling
px, pw = 76, 22
box(px, yb, pw, hb, "Masked\nattention pooling\nover issues\n(per valid day)", fc="#EEF5FB", ec=C_OLD_HI, fs=6.3)
# concat node
cx_, cy_ = 103, yb + hb / 2
ax.add_patch(Circle((cx_, cy_), 2.2, fc="white", ec="k", lw=0.7))
ax.plot([cx_ - 1.3, cx_ + 1.3], [cy_, cy_], color="k", lw=0.7); ax.plot([cx_, cx_], [cy_ - 1.3, cy_ + 1.3], color="k", lw=0.7)
# day context below
box(70, 2, 66, 11.5, "Day context: season, latitude, longitude, altitude, lead,\nDTR, inter-issue spread, climate zone",
    fc="#F5F5F5", ec="#9E9E9E", fs=6.2)
arrow(cx_, 13.5, cx_, cy_ - 2.2)
# 4 transformer
ex, ew = 108.5, 23
box(ex, yb, ew, hb, "Transformer\nencoder across\nthe 7 valid days", fc="#FFF7F2", ec=C_CUR_HI, fs=6.3)
# 5 head
hx, hw = 136.5, 29
box(hx, yb, hw, hb, f"point forecast or\nquantiles P5, P10, P25,\nP50, P75, P90, P95\nof {S.ET0}, leads 1\u20137",
    fc="#FFFFFF", ec=EDGE, fs=6.2, bold_title="Output head")
# 6 ETc
qx, qw = 170.5, 17.5
box(qx, yb + hb / 2 + 0.6, qw, hb / 2 - 0.6, f"{S.ETC} = K$_\\mathrm{{c}}$ × {S.ET0}\n(standard)", fc="#EAF4EA", ec="#1B7837", fs=6.0)
box(qx, yb - 9.5, qw, hb / 2 + 8.9, f"ET$_\\mathrm{{a}}$ = K$_\\mathrm{{c,EC}}$ × {S.ET0}\n(K$_\\mathrm{{c,EC}}$ from\nearlier EC years)\n\u2192 verified vs.\npaddy EC", fc="#F3EEF8", ec="#6A3D9A", fs=5.7)
for x0, x1 in ((25, tx), (tx + tw, px), (px + pw, cx_ - 2.2), (cx_ + 2.2, ex), (ex + ew, hx)):
    arrow(x0 + 0.2, cy_, x1 - 0.2, cy_)
arrow(hx + hw + 0.2, cy_ + 3, qx - 0.2, yb + hb * 0.78)
arrow(hx + hw + 0.2, cy_ - 3, qx - 0.2, yb + 2)
S.save(fig, "Fig02_method_schematic")
