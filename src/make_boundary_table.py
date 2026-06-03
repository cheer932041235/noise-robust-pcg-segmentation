"""+-60ms boundary-onset F1 (Springer protocol) vs SNR, 5-seed, for deep/hybrid/gated,
computed offline from the saved ext npz (no GPU). Answers reviewer P1 (quantify the
boundary-precision trade-off across noise levels)."""
import sys, glob, json
import numpy as np
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
from eval_pcg import boundary_f1
from hybrid_decode import cycle_violation_rate
PRED = "/data2/sjx/projects/pcg_seg/preds"; FEAT_FS = 50
SEEDS = [0, 1, 2, 3, 4]
TAGS = [("clean", "clean"), ("snr15", "15"), ("snr10", "10"), ("snr5", "5"), ("snr0", "0"), ("snr-5", "-5")]

def load(s, tag):
    fs = glob.glob(f"{PRED}/preds_s{s}_deep_{tag}_ext.npz")
    if not fs: return None
    d = np.load(fs[0], allow_pickle=True)
    return ([np.asarray(x).astype(int) for x in d["deep"]],
            [np.asarray(x).astype(int) for x in d["hyb"]],
            [np.asarray(x).astype(int) for x in d["gt"]])

tau = {}
for s in SEEDS:
    r = load(s, "clean")
    if r: tau[s] = float(np.percentile([cycle_violation_rate(x) for x in r[0]], 95))

def bf1(p, g): return boundary_f1(p, g, FEAT_FS)[0]

block = {"deep": {}, "hybrid": {}, "gated": {}}
for tag, lab in TAGS:
    per = {"deep": [], "hybrid": [], "gated": []}
    for s in SEEDS:
        r = load(s, tag)
        if not r: continue
        deep, hyb, gt = r
        v = np.array([cycle_violation_rate(x) for x in deep])
        gated = [hyb[i] if v[i] > tau.get(s, 0.2) else deep[i] for i in range(len(deep))]
        per["deep"].append(bf1(deep, gt)); per["hybrid"].append(bf1(hyb, gt)); per["gated"].append(bf1(gated, gt))
    for k in block:
        block[k][lab] = [round(float(np.mean(per[k])), 4), round(float(np.std(per[k])), 4)]
    print(lab, {k: round(float(np.mean(per[k])), 3) for k in per}, flush=True)

rj = "/data2/sjx/projects/pcg_seg/results.json"; R = json.load(open(rj))
R["boundary_f1_vs_snr_5seed"] = block; json.dump(R, open(rj, "w"), indent=2)
print("saved boundary_f1_vs_snr_5seed", flush=True)
