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
    tp={c:0 for c in STATES};fp={c:0 for c in STATES};fn={c:0 for c in STATES}
    for p,g in zip(preds,gts):
        n=min(len(p),len(g));p=p[:n];g=g[:n];ann=g>0
        for c in STATES:
            tp[c]+=int(((p==c)&(g==c)&ann).sum());fp[c]+=int(((p==c)&(g!=c)&ann).sum());fn[c]+=int(((p!=c)&(g==c)&ann).sum())
    f1=[]
    for c in STATES:
        se=tp[c]/max(tp[c]+fn[c],1);pp=tp[c]/max(tp[c]+fp[c],1);f1.append(2*se*pp/max(se+pp,1e-9))
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
print("len pred[0]", len(preds[0]), "len seg orig", len(recs[120]["seg"]),
      "sig", len(recs[120]["signal"]), "ratio", len(recs[120]["signal"])/len(preds[0]), flush=True)

# mid snippet where gt annotated
g0 = gts[0]; nz = np.where(g0 > 0)[0]
if len(nz):
    s = nz[len(nz)//2]
    print("mid pred:", preds[0][s:s+30].astype(int).tolist(), flush=True)
    print("mid gt  :", g0[s:s+30].tolist(), flush=True)

# time-shift search (shift pred relative to gt)
best = (0, -1)
for sh in range(-40, 41):
    sp, sg = [], []
    for p, g in zip(preds, gts):
        if sh >= 0:
            sp.append(p[sh:]); sg.append(g[:len(g)-sh] if sh else g)
        else:
            sp.append(p[:sh]); sg.append(g[-sh:])
    f = mf1(sp, sg)
    if f > best[1]:
        best = (sh, f)
print(f"best time-shift={best[0]} samples ({best[0]*20}ms) macroF1={best[1]:.4f}", flush=True)
