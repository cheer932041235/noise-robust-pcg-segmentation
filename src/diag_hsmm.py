"""Diagnose HSMM macro-F1≈0.24: is it a state-label convention offset, a time
misalignment, or broken features? Tries all cyclic label rotations + dumps a snippet."""
import sys, numpy as np
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/hsmm")
from pcg_loader import load_circor
from segmentation_model import SegmentationModel
from duration_distributions import DataDistribution

DATA = "/data2/sjx/projects/pcg_seg/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
STATES = [1, 2, 3, 4]


def resample_seg(seg, n):
    idx = (np.arange(n) * (len(seg) / n)).astype(int).clip(0, len(seg) - 1)
    return seg[idx]


def mf1(preds, gts):
    tp = {c: 0 for c in STATES}; fp = {c: 0 for c in STATES}; fn = {c: 0 for c in STATES}
    for p, g in zip(preds, gts):
        n = min(len(p), len(g)); p = p[:n]; g = g[:n]; ann = g > 0
        for c in STATES:
            tp[c] += int(((p == c) & (g == c) & ann).sum())
            fp[c] += int(((p == c) & (g != c) & ann).sum())
            fn[c] += int(((p != c) & (g == c) & ann).sum())
    f1 = []
    for c in STATES:
        se = tp[c] / max(tp[c] + fn[c], 1); pp = tp[c] / max(tp[c] + fp[c], 1)
        f1.append(2 * se * pp / max(se + pp, 1e-9))
    return float(np.mean(f1))


recs = load_circor(DATA, limit=160)
m = SegmentationModel(sampling_frequency=4000, feature_frequency=50)
m.fit([r["signal"] for r in recs[:120]], [r["seg"] for r in recs[:120]],
      data_distribution=DataDistribution(features_frequency=50))
print("fit ok", flush=True)

preds, gts = [], []
for r in recs[120:150]:
    p = np.asarray(m.predict(r["signal"]))
    preds.append(p); gts.append(resample_seg(r["seg"], len(p)))

print("pred unique values:", np.unique(np.concatenate(preds)), flush=True)
print("gt   unique values:", np.unique(np.concatenate(gts)), flush=True)
print("pred[0][:40]:", preds[0][:40].tolist(), flush=True)
print("gt  [0][:40]:", gts[0][:40].tolist(), flush=True)

# try all cyclic rotations of pred labels (1..4)
for rot in range(4):
    rp = [(((p - 1 + rot) % 4) + 1) for p in preds]
    print(f"rotation +{rot}: macroF1={mf1(rp, gts):.4f}", flush=True)

# also try: pred is 0-indexed states 0..3 -> +1
rp = [(p + 1) for p in preds]
print("assume 0-indexed (+1):", f"macroF1={mf1(rp, gts):.4f}", flush=True)
