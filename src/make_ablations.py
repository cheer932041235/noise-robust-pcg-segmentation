"""Two CPU-only ablations from the saved ext npz (no retrain):
(2) Naive-smoothing baseline: best median-filtered deep argmax vs hybrid -> shows the HSMM
    duration/order structure contributes beyond mere temporal smoothing.
Averaged over 5 seeds. Reuses saved deep argmax / hybrid decode / gt (posteriors not needed)."""
import sys, glob, json
import numpy as np
from scipy.signal import medfilt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from eval_pcg import per_sample_macro_f1
from hybrid_decode import cycle_violation_rate

PRED = "/data2/sjx/projects/pcg_seg/preds"; FIGS = "/data2/sjx/projects/pcg_seg/figs"
SEEDS = [0, 1, 2, 3, 4]
TAGS_NEED = ["clean", "snr0", "snr-5"]

def f1(p, g): return per_sample_macro_f1(list(p), list(g))[0]

# ---- preload + cache (deep, hyb, gt, deep-violations) once per (seed,tag) ----
CACHE = {}
for s in SEEDS:
    for tag in TAGS_NEED:
        fs = glob.glob(f"{PRED}/preds_s{s}_deep_{tag}_ext.npz")
        if not fs:
            continue
        d = np.load(fs[0], allow_pickle=True)
        deep = [np.asarray(x).astype(int) for x in d["deep"]]
        hyb = [np.asarray(x).astype(int) for x in d["hyb"]]
        gt = [np.asarray(x).astype(int) for x in d["gt"]]
        viol = np.array([cycle_violation_rate(x) for x in deep])
        CACHE[(s, tag)] = (deep, hyb, gt, viol)
print("cached", len(CACHE), "seed-tag pairs", flush=True)

# clean-calibrated tau* per seed (95th pctile of clean violation)
tau_star = {s: float(np.percentile(CACHE[(s, "clean")][3], 95)) for s in SEEDS if (s, "clean") in CACHE}
mean_tau = float(np.mean(list(tau_star.values())))
print("tau* per seed:", {k: round(v, 3) for k, v in tau_star.items()}, "mean", round(mean_tau, 3), flush=True)

# ---- (1) gate-threshold sweep at 0 and -5 dB ----
taus = np.linspace(0.0, 0.6, 25)
fig, ax = plt.subplots(figsize=(4.2, 3.2))
for tag, lab, col in [("snr0", "0 dB", ps.METHOD["hybrid"]), ("snr-5", "$-5$ dB", ps.METHOD["gated"])]:
    curve = []
    for t in taus:
        per = []
        for s in SEEDS:
            if (s, tag) not in CACHE: continue
            deep, hyb, gt, v = CACHE[(s, tag)]
            gated = [hyb[i] if v[i] > t else deep[i] for i in range(len(deep))]
            per.append(f1(gated, gt))
        curve.append(np.mean(per))
    ax.plot(taus, curve, "o-", ms=3, color=col, label=lab)
ax.axvline(mean_tau, ls="--", color="#666", lw=1)
ylo = ax.get_ylim()[0]
ax.text(mean_tau + 0.01, ylo + 0.01, f"$\tau^*={mean_tau:.2f}$", fontsize=7, color="#444")
ax.set_xlabel("gate threshold $\tau$"); ax.set_ylabel("gated macro-F1")
ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIGS}/figY_gatesweep.pdf"); fig.savefig(f"{FIGS}/figY_gatesweep.png")
print("saved figY_gatesweep", flush=True)

# ---- (2) naive smoothing baseline vs hybrid ----
wins = [3, 5, 7, 9, 11]
block = {}
for tag, lab in [("clean", "clean"), ("snr0", "0"), ("snr-5", "-5")]:
    deep_f1, med_best, hyb_f1 = [], [], []
    for s in SEEDS:
        if (s, tag) not in CACHE: continue
        deep, hyb, gt, _ = CACHE[(s, tag)]
        deep_f1.append(f1(deep, gt)); hyb_f1.append(f1(hyb, gt))
        med_best.append(max(f1([medfilt(x, w).astype(int) for x in deep], gt) for w in wins))
    block[tag] = {"deep": [round(float(np.mean(deep_f1)), 4), round(float(np.std(deep_f1)), 4)],
                  "median_best": [round(float(np.mean(med_best)), 4), round(float(np.std(med_best)), 4)],
                  "hybrid": [round(float(np.mean(hyb_f1)), 4), round(float(np.std(hyb_f1)), 4)]}
    print(tag, "deep", round(np.mean(deep_f1), 3), "median", round(np.mean(med_best), 3),
          "hybrid", round(np.mean(hyb_f1), 3), flush=True)

rj = "/data2/sjx/projects/pcg_seg/results.json"; R = json.load(open(rj))
R["ablation_gate_tau_star"] = round(mean_tau, 4)
R["ablation_smoothing_vs_hybrid"] = block
json.dump(R, open(rj, "w"), indent=2)
print("DONE: saved figY_gatesweep + ablation blocks", flush=True)
