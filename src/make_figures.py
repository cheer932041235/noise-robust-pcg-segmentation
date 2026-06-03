"""
Publication figures from saved predictions (preds/*.npz) + HSMM logs. No GPU.
  Fig A: noise-robustness curves (in-band) — deep / always-hybrid / gated / HSMM vs SNR.
  Fig B: cycle-violation rate vs SNR (deep's label-free degradation signal), mean±std over seeds.
Saves PNGs to figs/. Run after all in-band hybrid + HSMM-in-band runs finish.
Usage: python make_figures.py [tag]   (tag suffix on npz, e.g. b32d5 for strong baseline; default base)
"""
import sys, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from eval_pcg import per_sample_macro_f1
from hybrid_decode import cycle_violation_rate

PRED = "/data2/sjx/projects/pcg_seg/preds"
LOGS = "/data2/sjx/projects/pcg_seg/logs"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
os.makedirs(FIGS, exist_ok=True)
TAGS = [("clean", "clean"), ("snr10", "10 dB"), ("snr5", "5 dB"), ("snr0", "0 dB")]
XV = [99, 10, 5, 0]   # clean plotted at 99 (relabeled)


def f1(p, g):
    return per_sample_macro_f1(list(p), list(g))[0]


def per_seed(domain, tag, sfx=""):
    """returns list over seeds of (deep_f1, hyb_f1, gated_f1, viol_mean) using clean-calibrated tau."""
    out = []
    for f in sorted(glob.glob(f"{PRED}/preds_s*_{domain}_{tag}{sfx}.npz")):
        d = np.load(f, allow_pickle=True)
        deep, hyb, gt = list(d["deep"]), list(d["hyb"]), list(d["gt"])
        viol = np.array([cycle_violation_rate(x) for x in d["deep"]])
        out.append((f, deep, hyb, gt, viol))
    return out


def hsmm_inband():
    """mean±std overall macroF1 for HSMM in-band at snr5, snr0 (clean from main hsmm logs)."""
    res = {}
    for snr in [10, 5, 0]:
        vals = []
        for f in glob.glob(f"{LOGS}/hsmmIB_s*_snr{snr}.log"):
            t = open(f).read(); m = re.search(r"overall_macroF1=([0-9.]+)", t)
            if m:
                vals.append(float(m.group(1)))
        if vals:
            res[snr] = (np.mean(vals), np.std(vals))
    # clean HSMM from the 5-seed clean logs
    vals = []
    for f in glob.glob(f"{LOGS}/hsmm_seed*.log"):
        t = open(f).read(); m = re.search(r"overall_macroF1=([0-9.]+)", t)
        if m:
            vals.append(float(m.group(1)))
    if vals:
        res["clean"] = (np.mean(vals), np.std(vals))
    return res


def main():
    sfx = ("_" + sys.argv[1]) if len(sys.argv) > 1 else ""
    # calibrate tau on clean violation (95th pctile across seeds)
    clean = per_seed("deep", "clean", sfx)
    if not clean:
        print("no clean npz; abort"); return
    tau = float(np.percentile(np.concatenate([c[4] for c in clean]), 95))
    print(f"clean-calibrated tau={tau:.3f}")

    curves = {"deep": [], "hybrid": [], "gated": []}
    errs = {"deep": [], "hybrid": [], "gated": []}
    viol_m, viol_s = [], []
    for tag, _ in TAGS:
        seeds = per_seed("deep", tag, sfx)
        df, hf, gf, vv = [], [], [], []
        for _, deep, hyb, gt, viol in seeds:
            use_h = viol > tau
            gated = [hyb[i] if use_h[i] else deep[i] for i in range(len(deep))]
            df.append(f1(deep, gt)); hf.append(f1(hyb, gt)); gf.append(f1(gated, gt))
            vv.append(viol.mean())
        for k, arr in [("deep", df), ("hybrid", hf), ("gated", gf)]:
            curves[k].append(np.mean(arr)); errs[k].append(np.std(arr))
        viol_m.append(np.mean(vv)); viol_s.append(np.std(vv))

    hs = hsmm_inband()

    # ---- Fig A: noise curves ----
    x = list(range(len(TAGS)))
    plt.figure(figsize=(6, 4))
    for k, mk in [("deep", "o-"), ("hybrid", "s--"), ("gated", "^-")]:
        plt.errorbar(x, curves[k], yerr=errs[k], fmt=mk, capsize=3, label=k)
    hk = [("clean", 0), (10, 1), (5, 2), (0, 3)]
    hx = [i for _, i in hk]; hy = [hs.get(k, (np.nan,))[0] for k, _ in hk]
    if not all(np.isnan(hy)):
        plt.plot(hx, hy, "d:", color="gray", label="HSMM")
    plt.xticks(x, [t[1] for t in TAGS]); plt.gca().invert_xaxis()
    plt.xlabel("test SNR (in-band noise)"); plt.ylabel("per-sample macro-F1")
    plt.title("Noise robustness: deep vs hybrid vs gated"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIGS}/figA_noise_curves{sfx}.pdf"); plt.savefig(f"{FIGS}/figA_noise_curves{sfx}.png", dpi=150); plt.close()

    # ---- Fig B: violation signal ----
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, viol_m, yerr=viol_s, fmt="o-", color="crimson", capsize=3)
    plt.axhline(tau, ls="--", color="k", alpha=0.6, label=f"gate τ={tau:.2f}")
    plt.xticks(x, [t[1] for t in TAGS]); plt.gca().invert_xaxis()
    plt.xlabel("test SNR (in-band noise)"); plt.ylabel("cycle-violation rate of deep output")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIGS}/figB_violation{sfx}.pdf"); plt.savefig(f"{FIGS}/figB_violation{sfx}.png", dpi=150); plt.close()

    print("curves:", {k: [round(v, 4) for v in curves[k]] for k in curves})
    print("viol:", [round(v, 3) for v in viol_m])
    print(f"saved figA/figB to {FIGS}")


if __name__ == "__main__":
    main()
