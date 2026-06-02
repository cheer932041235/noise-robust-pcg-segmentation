"""Diagnostic figures from existing predictions (white in-band npz, domain=deep):
  figE: state confusion matrices, deep clean vs deep 0 dB (failure structure under noise)
  figF: per-state F1 (deep, clean) — S2 is the hardest state
  figG: S1/S2 boundary-onset error histogram (ms), deep clean (Springer-style timing accuracy)
IEEE palette, vector PDF. No GPU / no re-run."""
import sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from eval_pcg import _onsets, _match

PRED = "/data2/sjx/projects/pcg_seg/preds"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
FEAT_FS = 50
STATES = [1, 2, 3, 4]
NAMES = [ps.STATE_NAMES[c] for c in STATES]


def load(tag):
    deep, gt = [], []
    for f in sorted(glob.glob(f"{PRED}/preds_s*_deep_{tag}.npz")):
        d = np.load(f, allow_pickle=True)
        deep += list(d["deep"]); gt += list(d["gt"])
    return deep, gt


def confusion(deep, gt):
    M = np.zeros((4, 4))
    for p, g in zip(deep, gt):
        n = min(len(p), len(g)); p, g = p[:n], g[:n]
        for i, gc in enumerate(STATES):
            for j, pc in enumerate(STATES):
                M[i, j] += np.sum((g == gc) & (p == pc))
    return M / (M.sum(1, keepdims=True) + 1e-9)


def per_state_f1(deep, gt):
    f1 = []
    for c in STATES:
        tp = fp = fn = 0
        for p, g in zip(deep, gt):
            n = min(len(p), len(g)); p, g = p[:n], g[:n]; ann = g > 0
            tp += int(((p == c) & (g == c) & ann).sum())
            fp += int(((p == c) & (g != c) & ann).sum())
            fn += int(((p != c) & (g == c) & ann).sum())
        se = tp / max(tp + fn, 1); pp = tp / max(tp + fp, 1)
        f1.append(2 * se * pp / max(se + pp, 1e-9))
    return f1


def boundary_errors_ms(deep, gt, state):
    """signed onset error (pred-gt) in ms for matched boundaries of `state`."""
    errs = []
    for p, g in zip(deep, gt):
        n = min(len(p), len(g))
        pj, gj = _onsets(p[:n], state), _onsets(g[:n], state)
        for go in gj:
            if len(pj) == 0:
                continue
            k = int(np.argmin(np.abs(pj - go)))
            if abs(pj[k] - go) <= int(0.15 * FEAT_FS):   # within 150 ms = matched
                errs.append((pj[k] - go) / FEAT_FS * 1000)
    return np.array(errs)


# ---------- figE: confusion clean vs 0dB ----------
dc, gc = load("clean"); d0, g0 = load("snr0")
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
for ax, (M, ttl) in zip(axes, [(confusion(dc, gc), "clean"), (confusion(d0, g0), "0 dB (in-band)")]):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(NAMES, rotation=30, fontsize=6)
    ax.set_yticks(range(4)); ax.set_yticklabels(NAMES, fontsize=6)
    ax.set_xlabel("predicted", fontsize=7); ax.set_title(f"deep, {ttl}", fontsize=8)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.5 else "black")
axes[0].set_ylabel("ground truth", fontsize=7)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
fig.savefig(f"{FIGS}/figE_confusion.pdf"); fig.savefig(f"{FIGS}/figE_confusion.png"); plt.close(fig)

# ---------- figF: per-state F1 (clean) ----------
f1c = per_state_f1(dc, gc)
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.bar(range(4), f1c, color=[ps.STATE[c] for c in STATES], edgecolor="#333", linewidth=0.6)
ax.set_xticks(range(4)); ax.set_xticklabels(NAMES, fontsize=7); ax.set_ylim(0.7, 0.95)
ax.set_ylabel("per-state F1 (clean)", fontsize=8); ax.set_title("S2 is the hardest state", fontsize=8)
for i, v in enumerate(f1c):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=6)
fig.savefig(f"{FIGS}/figF_perstate.pdf"); fig.savefig(f"{FIGS}/figF_perstate.png"); plt.close(fig)

# ---------- figG: boundary onset error (ms), clean ----------
fig, ax = plt.subplots(figsize=(3.8, 2.8))
for c, col in [(1, ps.STATE[1]), (3, ps.STATE[3])]:
    e = boundary_errors_ms(dc, gc, c)
    ax.hist(e, bins=np.arange(-150, 151, 15), alpha=0.6, color=col,
            label=f"{ps.STATE_NAMES[c]} (μ={e.mean():.0f}±{e.std():.0f} ms)")
ax.axvline(0, color="#333", lw=0.8, ls="--")
ax.set_xlabel("onset error: pred − GT (ms)", fontsize=8); ax.set_ylabel("count", fontsize=8)
ax.set_title("boundary timing (deep, clean)", fontsize=8); ax.legend(fontsize=6)
fig.savefig(f"{FIGS}/figG_boundary.pdf"); fig.savefig(f"{FIGS}/figG_boundary.png"); plt.close(fig)

print("saved figE/figF/figG; per-state F1 clean:", [round(x, 3) for x in f1c])
