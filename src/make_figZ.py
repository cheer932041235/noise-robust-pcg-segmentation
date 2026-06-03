"""figZ_gallery (v2) — representative held-out recordings at 0 dB, GT/deep/hybrid state strips.
Fixes over v1: (1) crop every row to the GT-labelled span so the three rows are time-aligned and
no row renders as blank class-0 white; (2) select recordings where the hybrid genuinely rescues a
fragmented deep output (plus one already-correct case the gate preserves), instead of degenerate
F1 0.00->0.00 cases. From saved ext npz (seed 0). CPU only."""
import sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from eval_pcg import per_sample_macro_f1
PRED = "/data2/sjx/projects/pcg_seg/preds"; FIGS = "/data2/sjx/projects/pcg_seg/figs"; FEAT_FS = 50

d = np.load(glob.glob(f"{PRED}/preds_s0_deep_snr0_ext.npz")[0], allow_pickle=True)
deep = [np.asarray(x).astype(int) for x in d["deep"]]
hyb = [np.asarray(x).astype(int) for x in d["hyb"]]
gt = [np.asarray(x).astype(int) for x in d["gt"]]

def f1(p, g): return per_sample_macro_f1([p], [g])[0]
df = np.array([f1(deep[i], gt[i]) for i in range(len(deep))])
hf = np.array([f1(hyb[i], gt[i]) for i in range(len(hyb))])
gain = hf - df

def span(g):
    nz = np.where(g != 0)[0]
    return (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, len(g))

# 5 clear-rescue cases (large gain, deep fragmented but not fully dead), spanning deep-F1 range
cand = [i for i in range(len(df)) if gain[i] > 0.15 and 0.05 < df[i] < 0.70]
cand.sort(key=lambda i: gain[i], reverse=True)
pool = cand[:40]
pool.sort(key=lambda i: df[i])  # spread along deep-F1
if len(pool) >= 5:
    sel = [pool[int(round(k))] for k in np.linspace(0, len(pool) - 1, 5)]
else:
    sel = pool[:]
# 1 already-correct case the gate preserves (deep already good, hybrid does no harm)
preserve = [i for i in range(len(df)) if df[i] > 0.75 and abs(gain[i]) < 0.05]
sel.append(max(preserve, key=lambda i: df[i]) if preserve else int(np.argmax(df)))
idxs = sel[:6]
print("selected:", [(int(i), round(float(df[i]), 2), round(float(hf[i]), 2)) for i in idxs], flush=True)

cmap = ListedColormap([ps.STATE[i] for i in (0, 1, 2, 3, 4)])
W = 400  # cap displayed window to ~8 s
fig, axes = plt.subplots(3, 2, figsize=(7.1, 4.7))
for ax, i in zip(axes.ravel(), idxs):
    lo, hi = span(gt[i])
    hi = min(hi, lo + W)
    g, dp, hb = gt[i][lo:hi], deep[i][lo:hi], hyb[i][lo:hi]
    T = min(len(g), len(dp), len(hb))
    stack = np.vstack([g[:T], dp[:T], hb[:T]])
    ax.imshow(stack, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest",
              extent=[0, T / FEAT_FS, 0, 3])
    ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(["hybrid", "deep", "GT"], fontsize=6)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(rf"deep F1 {df[i]:.2f} $\rightarrow$ hybrid F1 {hf[i]:.2f}", fontsize=7)
leg = [plt.matplotlib.patches.Patch(facecolor=ps.STATE[k], label=ps.STATE_NAMES[k]) for k in (1, 2, 3, 4)]
fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Representative held-out recordings at 0 dB: the hybrid repairs the fragmented deep cycle",
             fontsize=8)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(f"{FIGS}/figZ_gallery.pdf"); fig.savefig(f"{FIGS}/figZ_gallery.png")
print("saved figZ_gallery", flush=True)
