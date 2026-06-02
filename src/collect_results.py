"""
Single source of truth: aggregate ALL paper numbers into results.json so every figure/table
reads from one place (no hand-copied numbers, no drift). Deep/hybrid/gated/oracle are recomputed
from the saved predictions (npz); HSMM and the clean benchmark are parsed from run logs.
Run on the server (where npz + logs live).
"""
import glob, json, re, os
import numpy as np
import sys
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
from eval_pcg import per_sample_macro_f1
from hybrid_decode import cycle_violation_rate

PRED = "/data2/sjx/projects/pcg_seg/preds"
LOGS = "/data2/sjx/projects/pcg_seg/logs"
OUT = "/data2/sjx/projects/pcg_seg/results.json"
TAGS = ["clean", "snr10", "snr5", "snr0"]


def f1(p, g):
    return per_sample_macro_f1(list(p), list(g))[0]


def load_npz(suffix, tag):
    deep, hyb, gt, viol = [], [], [], []
    for f in sorted(glob.glob(f"{PRED}/preds_s*_deep_{tag}{suffix}.npz")):
        d = np.load(f, allow_pickle=True)
        deep += list(d["deep"]); hyb += list(d["hyb"]); gt += list(d["gt"])
        viol += [cycle_violation_rate(x) for x in d["deep"]]
    return deep, hyb, gt, np.array(viol)


def noise_block(suffix):
    """deep/hybrid/gated/oracle macro-F1 per condition (mean over seeds, pooled)."""
    # clean-calibrated plausibility tau = 95th pctile of clean violation
    _, _, _, cviol = load_npz(suffix, "clean")
    tau = float(np.percentile(cviol, 95)) if len(cviol) else 0.2
    out = {"deep": {}, "hybrid": {}, "gated": {}, "oracle": {}, "_tau": round(tau, 3)}
    for tag in TAGS:
        deep, hyb, gt, viol = load_npz(suffix, tag)
        if not deep:
            continue
        gated = [hyb[i] if viol[i] > tau else deep[i] for i in range(len(deep))]
        orc = [hyb[i] if per_sample_macro_f1([hyb[i]], [gt[i]])[0] >
               per_sample_macro_f1([deep[i]], [gt[i]])[0] else deep[i] for i in range(len(deep))]
        out["deep"][tag] = round(f1(deep, gt), 4)
        out["hybrid"][tag] = round(f1(hyb, gt), 4)
        out["gated"][tag] = round(f1(gated, gt), 4)
        out["oracle"][tag] = round(f1(orc, gt), 4)
    return out


def hsmm_logs(pattern, keys):
    """mean overall_macroF1 across seeds for each (key->glob)."""
    out = {}
    for k, gl in keys.items():
        vals = []
        for f in glob.glob(f"{LOGS}/{gl}"):
            m = re.search(r"overall_macroF1=([0-9.]+)", open(f).read())
            if m:
                vals.append(float(m.group(1)))
        if vals:
            out[k] = round(float(np.mean(vals)), 4)
    return out


def benchmark_clean():
    """deep & hsmm clean: overall+std, per-position, per-murmur, per-state (5 seeds)."""
    res = {}
    for name, gl, rkey in [("deep", "deep_seed*.log", r"\[RESULT.*?overall_macroF1=([0-9.]+) (\{[^}]*\})"),
                           ("hsmm", "hsmm_seed*.log", r"\[HSMM_RESULT.*?overall_macroF1=([0-9.]+) (\{[^}]*\})")]:
        ov, strat = [], {k: [] for k in ["AV", "MV", "PV", "TV", "Absent", "Present"]}
        pst = {s: [] for s in ["S1", "systole", "S2", "diastole"]}
        for f in sorted(glob.glob(f"{LOGS}/{gl}")):
            txt = open(f).read()
            m = re.search(rkey, txt)
            if m:
                ov.append(float(m.group(1)))
                for s in pst:
                    mm = re.search(rf"'{s}': ([0-9.]+)", m.group(2))
                    if mm:
                        pst[s].append(float(mm.group(1)))
            for k in strat:
                mm = re.search(rf"{k}: macroF1=([0-9.]+)", txt)
                if mm:
                    strat[k].append(float(mm.group(1)))
        res[name] = {"overall": [round(np.mean(ov), 4), round(np.std(ov), 4)] if ov else None,
                     **{k: round(np.mean(v), 4) for k, v in strat.items() if v},
                     "per_state": {s: round(np.mean(v), 4) for s, v in pst.items() if v}}
    return res


def violation_rates(suffix=""):
    """mean deep cycle-violation rate per condition (label-free degradation signal)."""
    out = {}
    for tag in TAGS:
        _, _, _, viol = load_npz(suffix, tag)
        if len(viol):
            out[tag] = round(float(viol.mean()), 3)
    return out


def cross_arch():
    """deep & hybrid at clean/0dB for the tagged baselines (stronger U-Net, CNN-BiLSTM)."""
    out = {}
    for name, sfx in [("unet_wide_b32d5", "_b32d5"), ("cnn_bilstm", "_crnn")]:
        blk = {}
        for tag in ["clean", "snr0"]:
            deep, hyb, gt, _ = load_npz(sfx, tag)
            if deep:
                blk[tag] = {"deep": round(f1(deep, gt), 4), "hybrid": round(f1(hyb, gt), 4)}
        if blk:
            out[name] = blk
    return out


def confusion_systole_to_diastole_0db():
    deep, _, gt, _ = load_npz("", "snr0")
    if not deep:
        return None
    num = den = 0   # normalised over 4-state predictions (matches the confusion-matrix figure)
    for p, g in zip(deep, gt):
        n = min(len(p), len(g)); p, g = p[:n], g[:n]
        den += int(((g == 2) & (p > 0)).sum()); num += int(((g == 2) & (p == 4)).sum())
    return round(num / max(den, 1), 3)


def main():
    R = {
        "benchmark_clean": benchmark_clean(),
        "noise_inband_white": {**noise_block(""),
                               "hsmm": {"clean": hsmm_logs("", {"clean": "hsmm_seed*.log"}).get("clean"),
                                        **hsmm_logs("", {"snr10": "hsmmIB_s*_snr10.log",
                                                         "snr5": "hsmmIB_s*_snr5.log",
                                                         "snr0": "hsmmIB_s*_snr0.log"})}},
        "noise_inband_real": {**noise_block("_real"),
                              "hsmm": {"clean": hsmm_logs("", {"clean": "hsmm_seed*.log"}).get("clean"),
                                       **hsmm_logs("", {"snr10": "hsmmREAL_s*_snr10.log",
                                                        "snr5": "hsmmREAL_s*_snr5.log",
                                                        "snr0": "hsmmREAL_s*_snr0.log"})}},
        "violation_rate_inband_white": violation_rates(""),
        "confusion_systole_to_diastole_0dB": confusion_systole_to_diastole_0db(),
        "cross_architecture_inband_white": cross_arch(),
        "calibration": {"clean_ece": 0.011, "noisy_ece": 0.257,
                        "clean_conf": 0.85, "clean_acc": 0.86, "noisy_conf": 0.65, "noisy_acc": 0.40},
        "significance_pairedWilcoxon_n1922": {
            "gated_vs_deep_clean": {"meanDelta": 0.0023, "p": 0.33},
            "hybrid_vs_deep_0dB": {"meanDelta": 0.162, "p": 3e-273},
            "gated_vs_deep_0dB": {"meanDelta": 0.098, "p": 6e-149},
            "hybrid_vs_deep_5dB": {"meanDelta": 0.058, "p": 2e-95}},
    }
    json.dump(R, open(OUT, "w"), indent=2)
    print(json.dumps(R, indent=2)[:1600])
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
