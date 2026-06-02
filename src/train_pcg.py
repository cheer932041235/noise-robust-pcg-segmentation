"""
PCG heart-sound segmentation on CirCor: train 1D U-Net (reused from ECG delineation),
evaluate per-sample 4-state macro-F1, STRATIFIED by auscultation location (AV/MV/PV/TV)
and by murmur (Present/Absent). The gating goal of Phase 1: confirm the story anchor —
does the deep model degrade on AV position / murmur recordings (where classical priors
are expected to hold)? If yes -> build the hybrid; if not -> rethink.

CirCor 4 kHz -> downsample to 1 kHz (heart-sound content < ~500 Hz). 5 classes:
0=none/unannotated, 1=S1, 2=systole, 3=S2, 4=diastole.

Usage: python train_pcg.py --epochs 30 --seed 0 [--use_prior 1]
"""
import argparse, csv, glob, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly

from unet1d import UNet1D
from pcg_loader import load_circor
try:
    from signal_prior import add_prior_channels
    HAS_PRIOR = True
except Exception:
    HAS_PRIOR = False

DATA = "/data2/sjx/projects/pcg_seg/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
TARGET_FS = 1000
WIN = 4096
STATES = [1, 2, 3, 4]
STATE_NAMES = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}


def load_murmur_map():
    """patient_id -> 'Present'/'Absent'/'Unknown' from training_data.csv."""
    csvp = os.path.join(os.path.dirname(DATA), "training_data.csv")
    m = {}
    if os.path.exists(csvp):
        with open(csvp) as f:
            for row in csv.DictReader(f):
                pid = row.get("Patient ID") or row.get("patient_id")
                m[str(pid)] = row.get("Murmur", "Unknown")
    return m


def resample_rec(r):
    if r["fs"] != TARGET_FS:
        from math import gcd
        g = gcd(TARGET_FS, int(r["fs"]))
        sig = resample_poly(r["signal"], TARGET_FS // g, int(r["fs"]) // g).astype(np.float32)
        ratio = TARGET_FS / r["fs"]
        seg = np.zeros(len(sig), dtype=np.int64)
        # remap seg by ratio (nearest)
        old = r["seg"]; idx = (np.arange(len(sig)) / ratio).astype(int).clip(0, len(old) - 1)
        seg = old[idx]
    else:
        sig, seg = r["signal"], r["seg"]
    sig = (sig - sig.mean()) / (sig.std() + 1e-6)
    return sig.astype(np.float32), seg


def make_windows(recs, win=WIN, stride=None):
    stride = stride or win
    X, Y = [], []
    for sig, seg in recs:
        for s in range(0, max(1, len(sig) - win + 1), stride):
            x, y = sig[s:s + win], seg[s:s + win]
            if len(x) < win:
                x = np.pad(x, (0, win - len(x))); y = np.pad(y, (0, win - len(y)))
            if (y > 0).sum() < win * 0.3:
                continue
            X.append(x); Y.append(y)
    return np.stack(X)[:, None, :].astype(np.float32), np.stack(Y).astype(np.int64)


def _add_noise(sig, snr_db, seed=0):
    if snr_db is None:
        return sig
    rng = np.random.RandomState(seed)
    p_sig = float(np.mean(sig.astype(np.float64) ** 2)) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    return (sig + rng.normal(0, np.sqrt(p_noise), size=sig.shape).astype(sig.dtype)).astype(np.float32)


@torch.no_grad()
def macro_f1(model, recs, dev, use_prior, snr_db=None):
    """Per-sample macro-F1 over the 4 states (annotated samples only). Optional test-time noise."""
    model.eval()
    tp = {c: 0 for c in STATES}; fp = {c: 0 for c in STATES}; fn = {c: 0 for c in STATES}
    for sig, seg in recs:
        if snr_db is not None:
            sig = _add_noise(sig, snr_db)
        x = torch.from_numpy(sig[None, None, :]).to(dev)
        if use_prior:
            x = add_prior_channels(x)
        pred = model(x)[0].argmax(0).cpu().numpy()
        ann = seg > 0
        for c in STATES:
            tp[c] += int(((pred == c) & (seg == c) & ann).sum())
            fp[c] += int(((pred == c) & (seg != c) & ann).sum())
            fn[c] += int(((pred != c) & (seg == c) & ann).sum())
    f1s = []
    for c in STATES:
        se = tp[c] / max(tp[c] + fn[c], 1); pp = tp[c] / max(tp[c] + fp[c], 1)
        f1s.append(2 * se * pp / max(se + pp, 1e-9))
    return float(np.mean(f1s)), {STATE_NAMES[c]: round(f1s[i], 4) for i, c in enumerate(STATES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_prior", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test_snrs", default="", help="comma list, e.g. '15,10,5,0' for test-time noise robustness")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    use_prior = bool(args.use_prior) and HAS_PRIOR

    raw = load_circor(DATA, limit=args.limit or None)
    murmur = load_murmur_map()
    # attach metadata, resample
    recs = []
    for r in raw:
        pid = r["id"].split("_")[0]
        sig, seg = resample_rec(r)
        recs.append({"sig": sig, "seg": seg, "loc": r["loc"], "murmur": murmur.get(pid, "Unknown"), "pid": pid})
    print(f"[load] {len(recs)} recordings, {len(set(r['pid'] for r in recs))} patients", flush=True)

    # patient-level split (avoid leakage)
    pids = sorted(set(r["pid"] for r in recs))
    rng = np.random.RandomState(args.seed); rng.shuffle(pids)
    n_test = len(pids) // 5
    test_pids = set(pids[:n_test])
    train = [r for r in recs if r["pid"] not in test_pids]
    test = [r for r in recs if r["pid"] in test_pids]

    Xtr, Ytr = make_windows([(r["sig"], r["seg"]) for r in train])
    in_ch = 1 + (5 if use_prior else 0)
    model = UNet1D(in_ch=in_ch, n_classes=5, base=16, depth=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    print(f"[cfg] prior={use_prior} train_win={len(Xtr)} test_rec={len(test)} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    bs = 32
    for ep in range(1, args.epochs + 1):
        model.train(); perm = np.random.permutation(len(Xtr)); tot = 0
        for s in range(0, len(Xtr), bs):
            b = perm[s:s + bs]
            x = torch.from_numpy(Xtr[b]).to(dev); y = torch.from_numpy(Ytr[b]).to(dev)
            if use_prior:
                x = add_prior_channels(x)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  ep{ep} loss={tot/(len(Xtr)//bs+1):.4f}", flush=True)

    # overall + stratified eval
    def ev(subset, snr=None):
        if not subset:
            return None, None
        return macro_f1(model, [(r["sig"], r["seg"]) for r in subset], dev, use_prior, snr_db=snr)
    overall, per = ev(test)
    print(f"[RESULT prior={int(use_prior)} seed={args.seed}] overall_macroF1={overall:.4f} {per}", flush=True)
    print("[STRAT by location]", flush=True)
    for loc in ["AV", "MV", "PV", "TV"]:
        sub = [r for r in test if r["loc"] == loc]
        f1, _ = ev(sub)
        print(f"  {loc}: macroF1={f1:.4f} n={len(sub)}" if f1 else f"  {loc}: n=0", flush=True)
    print("[STRAT by murmur]", flush=True)
    for mm in ["Absent", "Present"]:
        sub = [r for r in test if r["murmur"] == mm]
        f1, _ = ev(sub)
        print(f"  {mm}: macroF1={f1:.4f} n={len(sub)}" if f1 else f"  {mm}: n=0", flush=True)
    if args.test_snrs:
        print("[NOISE robustness (overall macroF1 at test-time SNR)]", flush=True)
        for snr in [int(s) for s in args.test_snrs.split(",")]:
            f1, _ = ev(test, snr=snr)
            print(f"  snr={snr}dB: macroF1={f1:.4f}", flush=True)
    print(f"[PCG_DONE prior={int(use_prior)} seed={args.seed}]", flush=True)


if __name__ == "__main__":
    main()
