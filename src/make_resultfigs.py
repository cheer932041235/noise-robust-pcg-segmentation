"""Two real-data result-difference figures (reviewer Q2):
  figW_scatter   : per-recording deep vs hybrid macro-F1 at 0 dB (paired, n~638), coloured by gain.
  figZ_gallery   : a gallery of representative held-out recordings at 0 dB — GT / deep / hybrid
                   state strips with per-recording macro-F1, mirroring a qualitative comparison grid.
From saved ext npz (seed 0). CPU only."""
import sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from eval_pcg import per_sample_macro_f1
from hybrid_decode import cycle_violation_rate
PRED = "/data2/sjx/projects/pcg_seg/preds"; FIGS = "/data2/sjx/projects/pcg_seg/figs"; FEAT_FS = 50

d = np.load(glob.glob(f"{PRED}/preds_s0_deep_snr0_ext.npz")[0], allow_pickle=True)
deep = [np.asarray(x).astype(int) for x in d["deep"]]
hyb = [np.asarray(x).astype(int) for x in d["hyb"]]
gt = [np.asarray(x).astype(int) for x in d["gt"]]
dc = np.load(glob.glob(f"{PRED}/preds_s0_deep_clean_ext.npz")[0], allow_pickle=True)
tau = float(np.percentile([cycle_violation_rate(np.asarray(x).astype(int)) for x in dc["deep"]], 95))

def f1(p, g): return per_sample_macro_f1([p], [g])[0]
df = np.array([f1(deep[i], gt[i]) for i in range(len(deep))])
hf = np.array([f1(hyb[i], gt[i]) for i in range(len(hyb))])
print(f"n={len(df)} deep_mean={df.mean():.3f} hybrid_mean={hf.mean():.3f} above_diag={ (hf>df).mean()*100:.0f}%", flush=True)

# ---- figW: paired scatter ----
fig, ax = plt.subplots(figsize=(3.6, 3.3))
sc = ax.scatter(df, hf, c=(hf - df), cmap="coolwarm_r", vmin=-0.4, vmax=0.4, s=9, edgecolor="none", alpha=0.8)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("deep macro-F1"); ax.set_ylabel("hybrid macro-F1")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("hybrid $-$ deep", fontsize=7); cb.ax.tick_params(labelsize=6)
ax.text(0.04, 0.96, f"{(hf>df).mean()*100:.0f}% above diagonal\n(hybrid better)", transform=ax.transAxes, fontsize=7, va="top")
fig.tight_layout(); fig.savefig(f"{FIGS}/figW_scatter.pdf"); fig.savefig(f"{FIGS}/figW_scatter.png")
print("saved figW_scatter", flush=True)

# ---- figZ: gallery of representative recordings ----
order = np.argsort(df)
idxs = [order[0], order[2], order[len(order)//4], order[len(order)//2], order[-25], order[-4]]
cmap = ListedColormap([ps.STATE[i] for i in (0, 1, 2, 3, 4)])
W = 400  # show first 8 s
fig, axes = plt.subplots(3, 2, figsize=(7.1, 4.6))
for ax, i in zip(axes.ravel(), idxs):
    T = min(len(gt[i]), len(deep[i]), len(hyb[i]), W)
    stack = np.vstack([gt[i][:T], deep[i][:T], hyb[i][:T]])
    ax.imshow(stack, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest",
              extent=[0, T/FEAT_FS, 0, 3])
    ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels(["hybrid", "deep", "GT"], fontsize=6)
    ax.set_xticks([]); 
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(rf"deep F1 {f1(deep[i],gt[i]):.2f} $\rightarrow$ hybrid F1 {f1(hyb[i],gt[i]):.2f}", fontsize=7)
leg = [plt.matplotlib.patches.Patch(facecolor=ps.STATE[k], label=ps.STATE_NAMES[k]) for k in (1, 2, 3, 4)]
fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.04, 1, 0.97]); fig.savefig(f"{FIGS}/figZ_gallery.pdf"); fig.savefig(f"{FIGS}/figZ_gallery.png")
print("saved figZ_gallery", flush=True)
