"""
Classical Springer logistic-regression HSMM baseline on CirCor, using the maintained
Python port (EchoStatements/Springer-Segmentation-Python). Same patient-level split and
same per-sample 4-state macro-F1 + AV/MV/PV/TV + murmur stratification as the deep U-Net,
so the comparison is apples-to-apples.

Gate question: does the classical HSMM hold up BETTER than deep on AV-position / murmur?
That premise is what the 'classical rescues' hybrid story rests on.

HSMM predict() returns states at feature_frequency (50 Hz). We resample GT to that rate
(nearest) and score on annotated samples only — identical metric definition to the deep run.

Usage: python run_hsmm.py --seed 0 [--limit N] [--n_jobs K]
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, "/data2/sjx/projects/pcg_seg/hsmm")
from pcg_loader import load_circor
from segmentation_model import SegmentationModel
from duration_distributions import DataDistribution

DATA = "/data2/sjx/projects/pcg_seg/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
FS = 4000
FEAT_FS = 50
STATES = [1, 2, 3, 4]
STATE_NAMES = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def load_murmur_map():
    csvp = os.path.join(os.path.dirname(DATA), "training_data.csv")
    m = {}
    if os.path.exists(csvp):
        with open(csvp) as f:
            for row in csv.DictReader(f):
                m[str(row.get("Patient ID"))] = row.get("Murmur", "Unknown")
    return m


def resample_seg(seg, target_len):
    """nearest-index resample of a per-sample int label array to target_len."""
    idx = (np.arange(target_len) * (len(seg) / target_len)).astype(int).clip(0, len(seg) - 1)
    return seg[idx]


def macro_f1(pred_segs, gt_segs):
    tp = {c: 0 for c in STATES}; fp = {c: 0 for c in STATES}; fn = {c: 0 for c in STATES}
    for pred, gt in zip(pred_segs, gt_segs):
        n = min(len(pred), len(gt)); pred = pred[:n]; gt = gt[:n]
        ann = gt > 0
        for c in STATES:
            tp[c] += int(((pred == c) & (gt == c) & ann).sum())
            fp[c] += int(((pred == c) & (gt != c) & ann).sum())
            fn[c] += int(((pred != c) & (gt == c) & ann).sum())
    f1s = []
    for c in STATES:
        se = tp[c] / max(tp[c] + fn[c], 1); pp = tp[c] / max(tp[c] + fp[c], 1)
        f1s.append(2 * se * pp / max(se + pp, 1e-9))
    return float(np.mean(f1s)), {STATE_NAMES[c]: round(f1s[i], 4) for i, c in enumerate(STATES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--noise_snr", type=float, default=None, help="add white noise to TEST signals at this SNR(dB)")
    ap.add_argument("--noise_lp", type=float, default=None, help="lowpass the added noise to this Hz (in-band noise to match deep's 1kHz-band)")
    ap.add_argument("--noise_type", default="white", choices=["white", "real"], help="real=CirCor train state-0 acquisition noise")
    args = ap.parse_args()

    raw = load_circor(DATA, limit=args.limit or None)
    murmur = load_murmur_map()
    for r in raw:
        r["pid"] = r["id"].split("_")[0]
        r["murmur"] = murmur.get(r["pid"], "Unknown")
    print(f"[load] {len(raw)} recordings, {len(set(r['pid'] for r in raw))} patients", flush=True)

    # SAME patient split as train_pcg.py (identical RNG + fraction)
    pids = sorted(set(r["pid"] for r in raw))
    rng = np.random.RandomState(args.seed); rng.shuffle(pids)
    n_test = len(pids) // 5
    test_pids = set(pids[:n_test])
    train = [r for r in raw if r["pid"] not in test_pids]
    test = [r for r in raw if r["pid"] in test_pids]
    print(f"[split seed={args.seed}] train_rec={len(train)} test_rec={len(test)}", flush=True)

    # real-noise bank at 4 kHz: state-0 (poor-quality) segments from TRAIN recordings (>=0.5 s).
    noise_bank = []
    if getattr(args, "noise_type", "white") == "real":
        for r in train:
            seg = r["seg"]; z = (seg == 0).astype(int)
            d = np.diff(np.concatenate([[0], z, [0]]))
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if b - a >= int(0.5 * FS):
                    noise_bank.append(r["signal"][a:b].astype(np.float32))
        print(f"[noise bank] {len(noise_bank)} real-noise segments (4kHz) from train", flush=True)

    model = SegmentationModel(sampling_frequency=FS, feature_frequency=FEAT_FS)
    print("[fit] training HSMM (feature extraction + logistic regression + durations)...", flush=True)
    # CRITICAL: fit expects segmentation at FEATURE rate (50 Hz) to match PCG_Features length.
    # Passing the 4 kHz seg makes the port truncate to the first ~0.4 s -> garbage emissions.
    def seg50(r):
        n = int(round(len(r["signal"]) * FEAT_FS / FS))
        return resample_seg(r["seg"], n)
    # pass an INSTANCE (port stores the class by default -> get_distributions mis-binds self).
    # default priors == Springer's fixed physiological S1/S2 durations; systole/diastole from HR.
    model.fit([r["signal"] for r in train], [seg50(r) for r in train],
              data_distribution=DataDistribution(features_frequency=FEAT_FS))
    print("[fit] done", flush=True)

    # predict per test recording; resample GT to 50 Hz to match
    def add_noise(sig, snr_db, seed=0, lp=None):
        if snr_db is None:
            return sig
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, 1.0, size=sig.shape)
        if lp is not None:
            from scipy.signal import butter, filtfilt
            b, a = butter(4, lp / (FS / 2.0), btype="low")
            noise = filtfilt(b, a, noise)
        p_sig = float(np.mean(sig.astype(np.float64) ** 2)) + 1e-12
        p_n = float(np.mean(noise ** 2)) + 1e-12
        noise *= np.sqrt((p_sig / (10 ** (snr_db / 10.0))) / p_n)
        return (sig + noise.astype(sig.dtype)).astype(np.float32)

    def add_real(sig, snr_db, idx):
        if snr_db is None or not noise_bank:
            return sig
        nb = noise_bank[idx % len(noise_bank)]
        n = np.tile(nb, int(np.ceil(len(sig) / len(nb))))[:len(sig)]
        p_sig = float(np.mean(sig.astype(np.float64) ** 2)) + 1e-12
        p_n = float(np.mean(n.astype(np.float64) ** 2)) + 1e-12
        n = n * np.sqrt((p_sig / (10 ** (snr_db / 10.0))) / p_n)
        return (sig + n).astype(np.float32)

    preds, gts, locs, mms = [], [], [], []
    for i, r in enumerate(test):
        try:
            if args.noise_type == "real":
                sig = add_real(r["signal"], args.noise_snr, i)
            else:
                sig = add_noise(r["signal"], args.noise_snr, lp=args.noise_lp)
            ps = np.asarray(model.predict(sig))
        except Exception as e:
            continue
        preds.append(ps)
        gts.append(resample_seg(r["seg"], len(ps)))
        locs.append(r["loc"]); mms.append(r["murmur"])
        if (i + 1) % 100 == 0:
            print(f"  predicted {i+1}/{len(test)}", flush=True)
    print(f"[predict] {len(preds)} recordings done", flush=True)

    def ev(mask):
        p = [preds[i] for i in range(len(preds)) if mask[i]]
        g = [gts[i] for i in range(len(preds)) if mask[i]]
        if not p:
            return None, None
        return macro_f1(p, g)

    overall, per = ev([True] * len(preds))
    print(f"[HSMM_RESULT seed={args.seed}] overall_macroF1={overall:.4f} {per}", flush=True)
    print("[STRAT by location]", flush=True)
    for loc in ["AV", "MV", "PV", "TV"]:
        m = [locs[i] == loc for i in range(len(preds))]
        f1, _ = ev(m)
        print(f"  {loc}: macroF1={f1:.4f} n={sum(m)}" if f1 else f"  {loc}: n=0", flush=True)
    print("[STRAT by murmur]", flush=True)
    for mm in ["Absent", "Present"]:
        m = [mms[i] == mm for i in range(len(preds))]
        f1, _ = ev(m)
        print(f"  {mm}: macroF1={f1:.4f} n={sum(m)}" if f1 else f"  {mm}: n=0", flush=True)
    print(f"[HSMM_DONE seed={args.seed}]", flush=True)


if __name__ == "__main__":
    main()
