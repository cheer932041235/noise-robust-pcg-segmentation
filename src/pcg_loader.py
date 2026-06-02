"""
Loader for CirCor DigiScope PCG (PhysioNet 2022) heart-sound segmentation.
Each recording: a .wav (4 kHz audio) + a .tsv segmentation file with rows
    start_sec <TAB> end_sec <TAB> state
where state in {1=S1, 2=systole, 3=S2, 4=diastole, 0=unannotated}. We build a per-sample
segmentation mask (0=unannotated/none, 1..4 the four cardiac-cycle states) — directly
analogous to our ECG P/QRS/T segmentation, so the same 1D U-Net + eval pipeline transfer
(just n_classes=5: none+S1+systole+S2+diastole).

Patient metadata (auscultation location AV/MV/PV/TV, murmur present/absent) comes from the
filename (e.g. '50001_AV.wav') and training_data.csv -> used later for the position- and
murmur-conditioned robustness analysis.
"""
import os, glob
import numpy as np
try:
    import scipy.io.wavfile as wavfile
except Exception:
    wavfile = None

STATE_NAMES = {1: "S1", 2: "systole", 3: "S2", 4: "diastole"}
LOCATIONS = ["AV", "MV", "PV", "TV", "Phc"]


def load_tsv(path):
    """rows: start_sec, end_sec, state(int). Returns list[(start,end,state)]."""
    segs = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 3:
                try:
                    segs.append((float(p[0]), float(p[1]), int(float(p[2]))))
                except ValueError:
                    continue
    return segs


def tsv_to_seg(segs, fs, n):
    seg = np.zeros(n, dtype=np.int64)
    for s, e, st in segs:
        if st in (1, 2, 3, 4):
            a = max(0, int(round(s * fs))); b = min(n, int(round(e * fs)))
            seg[a:b] = st
    return seg


def location_from_id(rec_id):
    for loc in LOCATIONS:
        if rec_id.endswith("_" + loc) or ("_" + loc + "_") in rec_id:
            return loc
    return "unknown"


def load_circor_record(wav_path):
    tsv_path = wav_path[:-4] + ".tsv"
    fs, sig = wavfile.read(wav_path)
    sig = sig.astype(np.float32)
    if sig.ndim > 1:
        sig = sig[:, 0]
    segs = load_tsv(tsv_path) if os.path.exists(tsv_path) else []
    seg = tsv_to_seg(segs, fs, len(sig))
    rid = os.path.basename(wav_path)[:-4]
    return {"id": rid, "signal": sig, "fs": fs, "seg": seg,
            "loc": location_from_id(rid), "n_states": int((seg > 0).any())}


def load_circor(data_dir, limit=None, require_label=True):
    wavs = sorted(glob.glob(f"{data_dir}/**/*.wav", recursive=True))
    out = []
    for w in wavs:
        if require_label and not os.path.exists(w[:-4] + ".tsv"):
            continue
        try:
            r = load_circor_record(w)
            if require_label and not (r["seg"] > 0).any():
                continue                       # skip records with empty/no annotation
            out.append(r)
        except Exception:
            continue
        if limit and len(out) >= limit:
            break
    return out


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "/data2/sjx/projects/pcg_seg/data"
    recs = load_circor(base, limit=5)
    print(f"loaded {len(recs)} records (limit 5)")
    for r in recs[:3]:
        counts = {STATE_NAMES[c]: int((r["seg"] == c).sum()) for c in (1, 2, 3, 4)}
        print(f"  id={r['id']} loc={r['loc']} fs={r['fs']} len={len(r['signal'])}"
              f" ({len(r['signal'])/r['fs']:.1f}s) states={counts}")
    print("pcg_loader self-test done (verify against real data after download).")
