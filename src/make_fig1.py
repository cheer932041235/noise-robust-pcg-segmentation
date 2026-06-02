"""
Preliminary assembly of the Fig.1 overview/architecture figure: a left-to-right pipeline whose
stages are annotated with REAL data (waveform, posterior heatmap, deep-fragmented vs hybrid-rescued
segmentation strips) plus the real noise-robustness curve. Schematic boxes/arrows are plain vector
(to be polished later in Inkscape/TikZ; AI-gen only for decorative skins, never the data panels).
Uses qual_example.npz (real arrays from make_qualitative.py). IEEE palette, vector PDF.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps

FIGS = "/data2/sjx/projects/pcg_seg/figs"
NW = json.load(open("/data2/sjx/projects/pcg_seg/results.json"))["noise_inband_white"]
_tg = ["clean", "snr10", "snr5", "snr0"]
def _c(k):
    return [NW[k][t] for t in _tg]
FEAT_FS, DEEP_FS = 50, 1000
d = np.load(f"{FIGS}/qual_example.npz", allow_pickle=True)
gt, argd, hyb, post = d["gt"], d["argd"], d["hyb"], d["post"]
sd_noisy = d["sd_noisy"]
n = min(len(gt), len(argd), len(hyb))
gt, argd, hyb, post = gt[:n], argd[:n], hyb[:n], post[:n]
t1 = n / FEAT_FS
cmap = ListedColormap([ps.STATE[i] for i in (0, 1, 2, 3, 4)])

fig = plt.figure(figsize=(7.2, 4.6))
fig.patch.set_facecolor("white")

def box(x, y, w, h, text, fc="#EAF2FB", ec=ps.METHOD["deep"]):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                       linewidth=1.1, edgecolor=ec, facecolor=fc,
                       transform=fig.transFigure, zorder=2)
    fig.add_artist(p)
    fig.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, zorder=3)

def arrow(x0, y0, x1, y1, color="#555555"):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, zorder=1))

# ---- top row: schematic pipeline ----
box(0.04, 0.78, 0.13, 0.13, "Noisy PCG\n(in-band)", fc="#F2F2F2", ec="#888888")
box(0.25, 0.78, 0.13, 0.13, "1-D U-Net", fc="#EAF2FB", ec=ps.METHOD["deep"])
box(0.62, 0.78, 0.15, 0.13, "HSMM\nduration-Viterbi", fc="#FDEEE6", ec=ps.METHOD["hybrid"])
# gate diamond (as text box)
box(0.455, 0.785, 0.10, 0.115, "violation\n> τ ?", fc="#EAF7EA", ec=ps.METHOD["gated"])
arrow(0.17, 0.845, 0.25, 0.845)
arrow(0.38, 0.845, 0.455, 0.845)
arrow(0.555, 0.845, 0.62, 0.845)
arrow(0.77, 0.845, 0.85, 0.845)
fig.text(0.88, 0.845, "S1/sys/\nS2/dia", ha="center", va="center", fontsize=7)
fig.text(0.505, 0.76, "deep posteriors as emissions", ha="center", va="top", fontsize=6, color="#666")

# ---- middle: real data strips on one shared time axis ----
def strip(axpos, seq, label, viol=False):
    ax = fig.add_axes(axpos)
    ax.imshow(seq[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=4,
              extent=[0, t1, 0, 1], interpolation="nearest")
    ax.set_yticks([0.5]); ax.set_yticklabels([label], fontsize=7); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if viol:
        chg = np.where(np.diff(seq) != 0)[0]
        for i in chg:
            a, b = seq[i], seq[i + 1]
            if a in (1, 2, 3, 4) and b in (1, 2, 3, 4) and ({1: 2, 2: 3, 3: 4, 4: 1}[a] != b):
                ax.plot([t1 * i / n] * 2, [0, 1], color="#D62728", lw=0.8)
    return ax

axw = fig.add_axes([0.08, 0.62, 0.84, 0.08])
wav = sd_noisy[:int(n * DEEP_FS / FEAT_FS)]
axw.plot(np.arange(len(wav)) / DEEP_FS, wav, color="#333", lw=0.4)
axw.set_xlim(0, t1); axw.set_yticks([]); axw.set_xticks([]); axw.grid(False)
axw.set_ylabel("PCG", fontsize=7, rotation=0, ha="right", va="center")
for s in axw.spines.values():
    s.set_visible(False)
strip([0.08, 0.545, 0.84, 0.05], gt, "GT")
strip([0.08, 0.485, 0.84, 0.05], argd, "deep", viol=True)
strip([0.08, 0.425, 0.84, 0.05], hyb, "hybrid")
fig.text(0.92, 0.51, "fragmented\n(impossible)", fontsize=6, color="#D62728", va="center")
fig.text(0.92, 0.45, "rescued", fontsize=6, color=ps.METHOD["hybrid"], va="center")

# ---- bottom-left: posterior heatmap (real) ----
axp = fig.add_axes([0.08, 0.15, 0.42, 0.21])
im = axp.imshow(post.T, aspect="auto", origin="lower", cmap="magma",
                extent=[0, t1, 0.5, 4.5], vmin=0, vmax=1, interpolation="nearest")
axp.set_yticks([1, 2, 3, 4]); axp.set_yticklabels([ps.STATE_NAMES[i] for i in (1, 2, 3, 4)], fontsize=6)
axp.set_xlabel("time (s)", fontsize=7); axp.set_title("deep posterior under noise (confident-but-wrong)", fontsize=6.5)
axp.grid(False)

# ---- bottom-right: noise-robustness curve (real) ----
axc = fig.add_axes([0.62, 0.15, 0.32, 0.21])
x = [0, 1, 2, 3]; lab = ["clean", "10", "5", "0"]
axc.plot(x, _c("deep"), "o-", color=ps.METHOD["deep"], label="deep", ms=3)
axc.plot(x, _c("hybrid"), "s--", color=ps.METHOD["hybrid"], label="hybrid", ms=3)
axc.plot(x, _c("gated"), "^-", color=ps.METHOD["gated"], label="gated", ms=3)
axc.plot(x, _c("hsmm"), "d:", color=ps.METHOD["HSMM"], label="HSMM", ms=3)
axc.set_xticks(x); axc.set_xticklabels(lab, fontsize=6); axc.invert_xaxis()
axc.set_xlabel("SNR (dB)", fontsize=7); axc.set_ylabel("macro-F1", fontsize=7)
axc.set_title("in-band noise robustness", fontsize=6.5); axc.legend(fontsize=5.5, ncol=2)
axc.tick_params(labelsize=6)

leg = [Patch(facecolor=ps.STATE[i], label=ps.STATE_NAMES[i]) for i in (1, 2, 3, 4)]
leg.append(plt.Line2D([0], [0], color="#D62728", lw=1, label="cycle violation"))
fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.01))
# gate two-way branch: deep-argmax (down, kept) vs hybrid (right)
arrow(0.505, 0.785, 0.505, 0.705, color=ps.METHOD["gated"])
fig.text(0.515, 0.74, "deep argmax\nif plausible", fontsize=5.5, color=ps.METHOD["gated"], va="center")

for ext in ("pdf", "png"):
    fig.savefig(f"{FIGS}/fig1_overview.{ext}", facecolor="white")
print(f"saved {FIGS}/fig1_overview.pdf/.png")
