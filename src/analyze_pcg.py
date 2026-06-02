"""
Offline analysis of saved hybrid predictions (preds/*.npz) — no GPU, instant iteration.
Compares deep-argmax vs always-on hybrid vs gated hybrids vs an ORACLE upper bound, to find
a label-free gate that makes the hybrid Pareto-dominant (>=deep clean, >>deep noisy).

Gate signals tested:
  - conf : deep posterior confidence (mean max-softmax)  [we already saw this fails: deep is
           confidently wrong under noise]
  - plaus: physiological implausibility of the DEEP argmax sequence = fraction of state
           transitions that violate the S1->sys->S2->dia cycle. Clean deep ~plausible,
           noisy deep ~fragmented/implausible. Apply the prior (hybrid) when implausible.
  - oracle: per-recording pick whichever of deep/hybrid scores higher vs GT (UPPER BOUND).

Usage: python analyze_pcg.py [domain]   (domain=deep|raw, default deep)
"""
import sys, glob, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from eval_pcg import per_sample_macro_f1, boundary_f1

PRED = "/data2/sjx/projects/pcg_seg/preds"
FEAT_FS = 50
ORDER = {1: 2, 2: 3, 3: 4, 4: 1}   # valid forward cycle


def cycle_violation_rate(seq):
    """fraction of state->state transitions (states 1..4) that break the cardiac cycle."""
    s = seq[seq > 0]
    if len(s) < 2:
        return 0.0
    chg = np.where(np.diff(s) != 0)[0]
    trans = [(s[i], s[i + 1]) for i in chg]
    trans = [(a, b) for a, b in trans if a in ORDER and b in (1, 2, 3, 4)]
    if not trans:
        return 0.0
    bad = sum(1 for a, b in trans if ORDER[a] != b)
    return bad / len(trans)


def f1_of(preds, gts):
    return per_sample_macro_f1(list(preds), list(gts))[0]


SUFFIX = ""  # set from argv[2] to load a tagged baseline (e.g. b32d5)


def load(domain, tag):
    fs = sorted(glob.glob(f"{PRED}/preds_s*_{domain}_{tag}{SUFFIX}.npz"))
    deep, hyb, gt, conf, plaus = [], [], [], [], []
    for f in fs:
        d = np.load(f, allow_pickle=True)
        deep += list(d["deep"]); hyb += list(d["hyb"]); gt += list(d["gt"]); conf += list(d["conf"])
        plaus += [cycle_violation_rate(x) for x in d["deep"]]
    return (np.array(deep, dtype=object), np.array(hyb, dtype=object), np.array(gt, dtype=object),
            np.array(conf), np.array(plaus), len(fs))


def gated(deep, hyb, use_hyb):
    return [hyb[i] if use_hyb[i] else deep[i] for i in range(len(deep))]


def main():
    global SUFFIX
    domain = sys.argv[1] if len(sys.argv) > 1 else "deep"
    SUFFIX = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
    tags = ["clean", "snr10", "snr5", "snr0"]
    # First pick gate thresholds on CLEAN+noisy jointly is cheating; instead choose plaus threshold
    # ONCE (domain-agnostic, from physiology: >5% cycle violations = degraded) and report.
    PLAUS_TAU = 0.05
    CONF_TAUS = [0.6, 0.7, 0.8, 0.85, 0.9]

    print(f"=== domain={domain} ===")
    print(f"{'tag':6} {'deep':>7} {'hyb':>7} {'plaus@.05':>9} {'oracle':>7} | "
          f"conf_mean plaus_mean hybFrac(plaus)")
    rows = {}
    for tag in tags:
        deep, hyb, gt, conf, plaus, nseed = load(domain, tag)
        if len(deep) == 0:
            print(f"{tag:6} (no data)"); continue
        d = f1_of(deep, gt); h = f1_of(hyb, gt)
        # plausibility gate
        use_h = plaus > PLAUS_TAU
        pl = f1_of(gated(deep, hyb, use_h), gt)
        # oracle (per-rec best)
        orc_use = np.array([per_sample_macro_f1([hyb[i]], [gt[i]])[0] >
                            per_sample_macro_f1([deep[i]], [gt[i]])[0] for i in range(len(deep))])
        orc = f1_of(gated(deep, hyb, orc_use), gt)
        rows[tag] = dict(deep=d, hyb=h, plaus=pl, oracle=orc, conf=conf, plausv=plaus, gt=gt, deepp=deep, hybp=hyb)
        print(f"{tag:6} {d:7.4f} {h:7.4f} {pl:9.4f} {orc:7.4f} | "
              f"{conf.mean():.3f}     {plaus.mean():.3f}     {use_h.mean():.2f}  (n={len(deep)},seeds={nseed})")

    # PRINCIPLED threshold: calibrate on CLEAN violation distribution (no test-noise peeking).
    # tau = 95th percentile of clean-recording cycle-violation rate -> ~5% of clean trigger the prior.
    if "clean" in rows:
        clean_plaus = rows["clean"]["plausv"]
        tau_cal = float(np.percentile(clean_plaus, 95))
        print(f"\n[clean-calibrated plausibility threshold tau={tau_cal:.3f} (95th pctile of clean violation)]")
        print(f"{'tag':6} {'deep':>7} {'gated':>7} {'Δvsdeep':>8}")
        for tag in tags:
            if tag not in rows:
                continue
            r = rows[tag]
            uh = r["plausv"] > tau_cal
            g = f1_of(gated(r["deepp"], r["hybp"], uh), r["gt"])
            print(f"{tag:6} {r['deep']:7.4f} {g:7.4f} {g - r['deep']:+8.4f}  (hybFrac={uh.mean():.2f})")

    # confidence-threshold sweep (to confirm it fails)
    print("\n[conf-threshold sweep, macroF1]")
    print(f"{'tau':>5} " + " ".join(f"{t:>7}" for t in tags))
    for tau in CONF_TAUS:
        line = f"{tau:5.2f} "
        for tag in tags:
            if tag not in rows:
                line += f"{'-':>7} "; continue
            r = rows[tag]
            uh = r["conf"] < tau
            line += f"{f1_of(gated(r['deepp'], r['hybp'], uh), r['gt']):7.4f} "
        print(line)

    # plausibility-threshold sweep
    print("\n[plausibility-threshold sweep (gate=hyb when violation>tau), macroF1]")
    print(f"{'tau':>5} " + " ".join(f"{t:>7}" for t in tags))
    for tau in [0.02, 0.05, 0.10, 0.15, 0.20]:
        line = f"{tau:5.2f} "
        for tag in tags:
            if tag not in rows:
                line += f"{'-':>7} "; continue
            r = rows[tag]
            uh = r["plausv"] > tau
            line += f"{f1_of(gated(r['deepp'], r['hybp'], uh), r['gt']):7.4f} "
        print(line)


if __name__ == "__main__":
    main()
