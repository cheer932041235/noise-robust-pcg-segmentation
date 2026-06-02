"""5-seed extended-SNR in-band noise curve (deep/hybrid/gated) with CI error bars, from the
'ext' npz. Per-seed pooled macro-F1 -> mean+-std over 5 seeds. Also dumps a results.json block.
Gated uses per-seed clean-calibrated plausibility threshold (95th pctile of clean violation)."""
import sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from eval_pcg import per_sample_macro_f1
from hybrid_decode import cycle_violation_rate

PRED = "/data2/sjx/projects/pcg_seg/preds"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
TAGS = [("clean", "clean"), ("snr15", "15"), ("snr10", "10"), ("snr5", "5"), ("snr0", "0"), ("snr-5", "-5")]
SEEDS = [0, 1, 2, 3, 4]


def f1(p, g):
    return per_sample_macro_f1(list(p), list(g))[0]


def load_seed(seed, tag):
    fs = glob.glob(f"{PRED}/preds_s{seed}_deep_{tag}_ext.npz")
    if not fs:
        return None
    d = np.load(fs[0], allow_pickle=True)
    return list(d["deep"]), list(d["hyb"]), list(d["gt"])


# per-seed clean-calibrated tau
tau = {}
for s in SEEDS:
    r = load_seed(s, "clean")
    if r:
        tau[s] = float(np.percentile([cycle_violation_rate(x) for x in r[0]], 95))

curves = {"deep": [], "hybrid": [], "gated": []}
errs = {"deep": [], "hybrid": [], "gated": []}
block = {"deep": {}, "hybrid": {}, "gated": {}}
for tag, lab in TAGS:
    per = {"deep": [], "hybrid": [], "gated": []}
    for s in SEEDS:
        r = load_seed(s, tag)
        if not r:
            continue
        deep, hyb, gt = r
        viol = np.array([cycle_violation_rate(x) for x in deep])
        gated = [hyb[i] if viol[i] > tau.get(s, 0.2) else deep[i] for i in range(len(deep))]
        per["deep"].append(f1(deep, gt)); per["hybrid"].append(f1(hyb, gt)); per["gated"].append(f1(gated, gt))
    for k in curves:
        curves[k].append(np.mean(per[k])); errs[k].append(np.std(per[k]))
        block[k][tag] = [round(float(np.mean(per[k])), 4), round(float(np.std(per[k])), 4)]

x = list(range(len(TAGS)))
plt.figure(figsize=(6, 4))
for k, mk in [("deep", "o-"), ("hybrid", "s--"), ("gated", "^-")]:
    plt.errorbar(x, curves[k], yerr=errs[k], fmt=mk, capsize=3, color=ps.METHOD[k], label=k)
plt.xticks(x, [t[1] for t in TAGS]); plt.gca().invert_xaxis()
plt.xlabel("test SNR (in-band noise, dB)"); plt.ylabel("per-sample macro-F1")
plt.title("In-band noise robustness (5 seeds, mean$\\pm$std)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIGS}/figA5_noise5seed.pdf"); plt.savefig(f"{FIGS}/figA5_noise5seed.png")

# merge into results.json
rj = "/data2/sjx/projects/pcg_seg/results.json"
R = json.load(open(rj)); R["noise_inband_white_5seed_extended"] = block
json.dump(R, open(rj, "w"), indent=2)
print("deep:", [round(v, 3) for v in curves["deep"]])
print("hybrid:", [round(v, 3) for v in curves["hybrid"]])
print("gated:", [round(v, 3) for v in curves["gated"]])
print("saved figA5_noise5seed + results.json block")
