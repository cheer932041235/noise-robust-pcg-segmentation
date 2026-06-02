"""
Shared PCG evaluation utilities, used by both the deep U-Net and the HSMM baseline so
the comparison is apples-to-apples.

Two metrics:
  1) per-sample macro-F1 over the 4 states (strict; penalizes boundary jitter heavily).
  2) boundary detection F1 with a tolerance window (the Springer/CirCor literature protocol):
     detect S1 and S2 onsets, match pred<->GT within +/- tol_ms. This is the fairer,
     literature-comparable number and the one reviewers expect.

Plus a noise-injection helper for the noise-robustness axis of the study.
"""
import numpy as np

STATES = [1, 2, 3, 4]
STATE_NAMES = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def per_sample_macro_f1(preds, gts):
    """preds/gts: lists of per-sample int label arrays at the SAME rate. Scores on
    annotated (gt>0) samples only. Returns (macroF1, {state: f1})."""
    tp = {c: 0 for c in STATES}; fp = {c: 0 for c in STATES}; fn = {c: 0 for c in STATES}
    for p, g in zip(preds, gts):
        n = min(len(p), len(g)); p = p[:n]; g = g[:n]; ann = g > 0
        for c in STATES:
            tp[c] += int(((p == c) & (g == c) & ann).sum())
            fp[c] += int(((p == c) & (g != c) & ann).sum())
            fn[c] += int(((p != c) & (g == c) & ann).sum())
    f1s = []
    for c in STATES:
        se = tp[c] / max(tp[c] + fn[c], 1); pp = tp[c] / max(tp[c] + fp[c], 1)
        f1s.append(2 * se * pp / max(se + pp, 1e-9))
    return float(np.mean(f1s)), {STATE_NAMES[c]: round(f1s[i], 4) for i, c in enumerate(STATES)}


def _onsets(seg, state):
    """sample indices where `seg` transitions into `state`."""
    m = (seg == state).astype(int)
    return np.where(np.diff(np.concatenate([[0], m])) == 1)[0]


def _match(pred_idx, gt_idx, tol):
    """greedy nearest matching within tol; returns (tp, fp, fn)."""
    used = np.zeros(len(gt_idx), dtype=bool); tp = 0
    for p in pred_idx:
        if len(gt_idx) == 0:
            break
        d = np.abs(gt_idx - p); j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True; tp += 1
    fp = len(pred_idx) - tp; fn = len(gt_idx) - used.sum()
    return tp, fp, int(fn)


def boundary_f1(preds, gts, fs, tol_ms=60):
    """S1/S2 onset detection F1 within +/- tol_ms (Springer protocol). preds/gts at rate fs.
    Returns (mean_f1_over_S1_S2, {S1:f1, S2:f1})."""
    tol = int(round(tol_ms * fs / 1000.0))
    out = {}
    for c in (1, 3):  # S1, S2
        TP = FP = FN = 0
        for p, g in zip(preds, gts):
            n = min(len(p), len(g))
            tp, fp, fn = _match(_onsets(p[:n], c), _onsets(g[:n], c), tol)
            TP += tp; FP += fp; FN += fn
        se = TP / max(TP + FN, 1); pp = TP / max(TP + FP, 1)
        out[STATE_NAMES[c]] = round(2 * se * pp / max(se + pp, 1e-9), 4)
    return float(np.mean(list(out.values()))), out


def add_noise(sig, snr_db, seed=0):
    """Add white Gaussian noise at a target SNR (dB) relative to signal power."""
    if snr_db is None:
        return sig
    rng = np.random.RandomState(seed)
    p_sig = np.mean(sig.astype(np.float64) ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    noise = rng.normal(0, np.sqrt(p_noise), size=sig.shape).astype(sig.dtype)
    return sig + noise
