"""
Paired significance tests for the headline claims, per-recording (Wilcoxon signed-rank),
pooled across seeds, from saved preds. Reports median per-recording macro-F1 + p-value.
Claims tested (in-band domain):
  - gated vs deep @ clean   (want: NOT worse -> p high or gated>=deep)
  - hybrid vs deep @ 0dB     (want: hybrid >> deep, p<<.05)
  - hybrid vs HSMM @ 0dB     (deep-emission hybrid beats classical; uses HSMM logs not per-rec -> skip pairing, report means)
  - gated vs deep @ 0dB      (Pareto: gated > deep)
"""
import sys, glob
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
from hybrid_decode import cycle_violation_rate

PRED = "/data2/sjx/projects/pcg_seg/preds"
STATES = [1, 2, 3, 4]


def rec_macro_f1(pred, gt):
    """per-recording macro-F1 over states present in GT."""
    n = min(len(pred), len(gt)); pred = pred[:n]; gt = gt[:n]; ann = gt > 0
    f1s = []
    for c in STATES:
        if (gt == c).sum() == 0:
            continue
        tp = int(((pred == c) & (gt == c) & ann).sum())
        fp = int(((pred == c) & (gt != c) & ann).sum())
        fn = int(((pred != c) & (gt == c) & ann).sum())
        se = tp / max(tp + fn, 1); pp = tp / max(tp + fp, 1)
        f1s.append(2 * se * pp / max(se + pp, 1e-9))
    return np.mean(f1s) if f1s else np.nan


def load(domain, tag):
    deep, hyb, gt, viol = [], [], [], []
    for f in sorted(glob.glob(f"{PRED}/preds_s*_{domain}_{tag}.npz")):
        d = np.load(f, allow_pickle=True)
        deep += list(d["deep"]); hyb += list(d["hyb"]); gt += list(d["gt"])
        viol += [cycle_violation_rate(x) for x in d["deep"]]
    return deep, hyb, gt, np.array(viol)


def gated_pred(deep, hyb, viol, tau):
    return [hyb[i] if viol[i] > tau else deep[i] for i in range(len(deep))]


def paired(a_list, b_list, gt):
    A = np.array([rec_macro_f1(a_list[i], gt[i]) for i in range(len(gt))])
    B = np.array([rec_macro_f1(b_list[i], gt[i]) for i in range(len(gt))])
    ok = ~(np.isnan(A) | np.isnan(B)); A, B = A[ok], B[ok]
    try:
        p = wilcoxon(A, B, zero_method="zsplit").pvalue
    except Exception:
        p = np.nan
    return np.median(A), np.median(B), float(np.mean(A - B)), p, len(A)


def main():
    dom = "deep"
    # clean-calibrated tau
    _, _, _, cviol = load(dom, "clean")
    tau = float(np.percentile(cviol, 95))
    print(f"[in-band] clean-calibrated tau={tau:.3f}\n")
    print(f"{'comparison':28} {'medA':>7} {'medB':>7} {'meanΔ':>8} {'p(Wilcoxon)':>13} n")

    # clean: gated vs deep
    deep, hyb, gt, viol = load(dom, "clean")
    g = gated_pred(deep, hyb, viol, tau)
    mA, mB, dl, p, n = paired(g, deep, gt)
    print(f"{'gated vs deep @clean':28} {mA:7.4f} {mB:7.4f} {dl:+8.4f} {p:13.2e} {n}")

    # 0dB: hybrid vs deep, gated vs deep
    deep, hyb, gt, viol = load(dom, "snr0")
    mA, mB, dl, p, n = paired(hyb, deep, gt)
    print(f"{'hybrid vs deep @0dB':28} {mA:7.4f} {mB:7.4f} {dl:+8.4f} {p:13.2e} {n}")
    g = gated_pred(deep, hyb, viol, tau)
    mA, mB, dl, p, n = paired(g, deep, gt)
    print(f"{'gated vs deep @0dB':28} {mA:7.4f} {mB:7.4f} {dl:+8.4f} {p:13.2e} {n}")

    # 5dB hybrid vs deep
    deep, hyb, gt, viol = load(dom, "snr5")
    mA, mB, dl, p, n = paired(hyb, deep, gt)
    print(f"{'hybrid vs deep @5dB':28} {mA:7.4f} {mB:7.4f} {dl:+8.4f} {p:13.2e} {n}")


if __name__ == "__main__":
    main()
