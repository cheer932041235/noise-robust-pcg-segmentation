"""
Fig.1 — clean, publication-grade ARCHITECTURE schematic (no real-data panels).
The per-recording data comparisons live in dedicated figures (figW scatter, figZ gallery,
figA5 noise curve); Fig.1 is now a pure branch-merge flowchart of the decoding pipeline:

    Noisy PCG -> 1-D U-Net -> posteriors -> cycle-violation gate
        gate (rho <= tau, plausible) -> deep argmax        \
        gate (rho >  tau)            -> HSMM duration-Viterbi -> S1/sys/S2/dia
        (deep posteriors reused as HSMM emissions)

Pure vector (matplotlib), Times-compatible fonts, runs locally (no server data needed).
Writes fig1_overview.pdf/.png into ../figs.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")

# ---- palette (semantic, consistent with the rest of the paper) ----
C_INPUT = "#888888"; FC_INPUT = "#F2F2F2"
C_DEEP  = "#1F77B4"; FC_DEEP  = "#EAF2FB"
C_GATE  = "#2CA02C"; FC_GATE  = "#EAF7EA"
C_HSMM  = "#D9663F"; FC_HSMM  = "#FDEEE6"
C_OUT   = "#555555"; FC_OUT   = "#EEF3EE"
C_ARR   = "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

fig = plt.figure(figsize=(7.2, 2.45))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.0, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.018",
                       linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight=("bold" if bold else "normal"))


def diamond(cx, cy, w, h, text, fc, ec, fs=7.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.3, edgecolor=ec,
                         facecolor=fc, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(x0, y0, x1, y1, color=C_ARR, rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                 connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=12, lw=lw, color=color, zorder=1,
                 shrinkA=2, shrinkB=2))


# ---- nodes ----
box(0.015, 0.40, 0.135, 0.22, "Noisy PCG\n(in-band)", FC_INPUT, C_INPUT)
box(0.205, 0.40, 0.115, 0.22, "1-D U-Net", FC_DEEP, C_DEEP, bold=True)
diamond(0.455, 0.51, 0.155, 0.40, "cycle-violation\n" + r"$\rho > \tau$ ?", FC_GATE, C_GATE)
box(0.605, 0.685, 0.165, 0.20, "deep argmax", FC_DEEP, C_DEEP)
box(0.605, 0.135, 0.165, 0.20, "HSMM\nduration-Viterbi", FC_HSMM, C_HSMM)
box(0.825, 0.40, 0.16, 0.22, "S1 / systole /\nS2 / diastole", FC_OUT, C_OUT, bold=True)

# ---- main flow ----
arrow(0.150, 0.51, 0.205, 0.51)
arrow(0.320, 0.51, 0.378, 0.51)
ax.text(0.349, 0.565, r"posteriors $q_t(s)$", ha="center", va="bottom",
        fontsize=6.3, color="#444")

# gate -> deep argmax (plausible, up)
arrow(0.515, 0.595, 0.605, 0.74, color=C_GATE, rad=0.18)
ax.text(0.520, 0.715, r"$\rho \leq \tau$" + "\n(plausible)", ha="left", va="center",
        fontsize=6.3, color=C_GATE)

# gate -> HSMM (violation, down)
arrow(0.515, 0.425, 0.605, 0.255, color=C_HSMM, rad=-0.18)
ax.text(0.520, 0.305, r"$\rho > \tau$", ha="left", va="center",
        fontsize=6.3, color=C_HSMM)
ax.text(0.560, 0.045, "deep posteriors reused as HSMM emissions",
        ha="center", va="center", fontsize=6.3, color="#444", style="italic")

# merge -> output
arrow(0.770, 0.74, 0.86, 0.625, color=C_ARR, rad=-0.18)
arrow(0.770, 0.255, 0.86, 0.40, color=C_ARR, rad=0.18)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGS, f"fig1_overview.{ext}"),
                facecolor="white", bbox_inches="tight", dpi=300)
print("saved fig1_overview.pdf/.png")
