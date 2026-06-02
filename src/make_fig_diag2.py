"""figFG: combine per-state F1 + boundary-onset error into ONE 2-panel float (reduces float
count / white space). Reads white in-band clean npz. IEEE palette, vector PDF."""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from eval_pcg import _onsets

PRED = "/data2/sjx/projects/pcg_seg/preds"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
FEAT_FS = 50
STATES = [1, 2, 3, 4]
NAMES = [ps.STATE_NAMES[c] for c in STATES]

deep, gt = [], []
for f in sorted(glob.glob(f"{PRED}/preds_s*_deep_clean.npz")):
    d = np.load(f, allow_pickle=True); deep += list(d["deep"]); gt += list(d["gt"])

# per-state F1
f1 = []
for c in STATES:
    tp = fp = fn = 0
    for p, g in zip(deep, gt):
        n = min(len(p), len(g)); p, g = p[:n], g[:n]; ann = g > 0
        tp += int(((p == c) & (g == c) & ann).sum()); fp += int(((p == c) & (g != c) & ann).sum())
        fn += int(((p != c) & (g == c) & ann).sum())
    se = tp / max(tp + fn, 1); pp = tp / max(tp + fp, 1); f1.append(2 * se * pp / max(se + pp, 1e-9))

# boundary onset error (ms) for S1, S2
def berr(state):
    e = []
    for p, g in zip(deep, gt):
        n = min(len(p), len(g)); pj, gj = _onsets(p[:n], state), _onsets(g[:n], state)
        for go in gj:
            if len(pj):
                k = int(np.argmin(np.abs(pj - go)))
                if abs(pj[k] - go) <= int(0.15 * FEAT_FS):
                    e.append((pj[k] - go) / FEAT_FS * 1000)
    return np.array(e)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
ax1.bar(range(4), f1, color=[ps.STATE[c] for c in STATES], edgecolor="#333", linewidth=0.6)
ax1.set_xticks(range(4)); ax1.set_xticklabels(NAMES, fontsize=7); ax1.set_ylim(0.7, 0.95)
ax1.set_ylabel("per-state F1 (clean)", fontsize=8); ax1.set_title("(a) S2 is the hardest state", fontsize=8)
for i, v in enumerate(f1):
    ax1.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=6)
for c, col in [(1, ps.STATE[1]), (3, ps.STATE[3])]:
    e = berr(c)
    ax2.hist(e, bins=np.arange(-150, 151, 15), alpha=0.6, color=col,
             label=f"{ps.STATE_NAMES[c]} ($\\mu$={e.mean():.0f}$\\pm${e.std():.0f} ms)")
ax2.axvline(0, color="#333", lw=0.8, ls="--")
ax2.set_xlabel("onset error: pred $-$ GT (ms)", fontsize=8); ax2.set_ylabel("count", fontsize=8)
ax2.set_title("(b) boundary timing (clean)", fontsize=8); ax2.legend(fontsize=6)
fig.tight_layout()
fig.savefig(f"{FIGS}/figFG_diag.pdf"); fig.savefig(f"{FIGS}/figFG_diag.png")
print("saved figFG_diag; per-state F1:", [round(x, 3) for x in f1])
