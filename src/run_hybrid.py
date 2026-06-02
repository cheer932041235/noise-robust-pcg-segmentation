"""
Phase 2 — Hybrid: deep U-Net posteriors + classical HSMM duration-Viterbi decode.
Compares, on the SAME patient split / metric, three decoders of the deep model:
  - deep-argmax  : raw per-sample argmax (what a pure deep model outputs)
  - hybrid       : deep posteriors decoded through the Springer duration-Viterbi (cyclic
                   order + physiological state-duration priors imposed)
Evaluated clean + at test-time SNRs, stratified by auscultation position and murmur,
with both per-sample macro-F1 and the +/-60ms boundary-F1 (Springer protocol).

Hypothesis: the duration prior repairs deep's fragmented/disordered predictions under
noise -> hybrid degrades far less than deep-argmax at low SNR.

Usage: python run_hybrid.py --seed 0 --epochs 30 --test_snrs 10,5,0 [--limit N]
"""
import argparse, csv, os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from math import gcd
import multiprocessing as mp

sys.path.insert(0, "/data2/sjx/projects/pcg_seg/hsmm")
from unet1d import UNet1D
from crnn1d import CRNN1D
from pcg_loader import load_circor
from eval_pcg import per_sample_macro_f1, boundary_f1, add_noise
from hybrid_decode import hybrid_decode_with_hr, estimate_hr
from duration_distributions import DataDistribution


def _decode_worker(arg):
    """pure-CPU hybrid decode, fanned out across a Pool."""
    post4, hr, sti, feat_fs = arg
    try:
        return hybrid_decode_with_hr(post4, hr, sti, feature_frequency=feat_fs)
    except Exception:
        return None

DATA = "/data2/sjx/projects/pcg_seg/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
RAW_FS = 4000
DEEP_FS = 1000
FEAT_FS = 50
WIN = 4096
STATES = [1, 2, 3, 4]


def murmur_map():
    csvp = os.path.join(os.path.dirname(DATA), "training_data.csv")
    m = {}
    if os.path.exists(csvp):
        with open(csvp) as f:
            for row in csv.DictReader(f):
                m[str(row.get("Patient ID"))] = row.get("Murmur", "Unknown")
    return m


def to_deep(sig4000):
    g = gcd(DEEP_FS, RAW_FS)
    s = resample_poly(sig4000, DEEP_FS // g, RAW_FS // g).astype(np.float32)
    return ((s - s.mean()) / (s.std() + 1e-6)).astype(np.float32)


def seg_to_deep(seg4000, n):
    idx = (np.arange(n) * (len(seg4000) / n)).astype(int).clip(0, len(seg4000) - 1)
    return seg4000[idx]


def resample_seg(seg, n):
    idx = (np.arange(n) * (len(seg) / n)).astype(int).clip(0, len(seg) - 1)
    return seg[idx]


def make_windows(pairs, win=WIN):
    X, Y = [], []
    for sig, seg in pairs:
        for s in range(0, max(1, len(sig) - win + 1), win):
            x, y = sig[s:s + win], seg[s:s + win]
            if len(x) < win:
                x = np.pad(x, (0, win - len(x))); y = np.pad(y, (0, win - len(y)))
            if (y > 0).sum() < win * 0.3:
                continue
            X.append(x); Y.append(y)
    return np.stack(X)[:, None, :].astype(np.float32), np.stack(Y).astype(np.int64)


@torch.no_grad()
def deep_posteriors(model, sig_deep, dev):
    """returns (post4[T_feat,4], argmax_feat[T_feat]) at FEAT_FS."""
    x = torch.from_numpy(sig_deep[None, None, :]).to(dev)
    logits = model(x)[0]                      # [5, N] at DEEP_FS
    prob = torch.softmax(logits, 0).cpu().numpy()
    argmax_deep = prob.argmax(0)              # [N] at DEEP_FS (0..4)
    Tf = int(round(len(sig_deep) * FEAT_FS / DEEP_FS))
    idx = (np.arange(Tf) * (len(sig_deep) / Tf)).astype(int).clip(0, len(sig_deep) - 1)
    post4 = prob[1:5, idx].T                  # [Tf,4] for S1,sys,S2,dia
    post4 = post4 / (post4.sum(1, keepdims=True) + 1e-9)
    return post4, argmax_deep[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--base", type=int, default=16, help="U-Net width (capacity); 32 = stronger baseline")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--arch", default="unet", choices=["unet", "crnn"], help="deep architecture")
    ap.add_argument("--tag", default="", help="suffix for npz/log to distinguish baselines")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--test_snrs", default="", help="e.g. 10,5,0")
    ap.add_argument("--noise_domain", default="raw", choices=["raw", "deep"],
                    help="raw=add noise at 4kHz (acquisition, band-limited by downsampling); "
                         "deep=add noise in-band at 1kHz (overlaps heart-sound band, deep is fragile)")
    ap.add_argument("--noise_type", default="white", choices=["white", "real"],
                    help="white=Gaussian; real=CirCor acquisition noise (train state-0 segments), in-band")
    ap.add_argument("--n_jobs", type=int, default=32)
    ap.add_argument("--save_ckpt", default="", help="path to save trained model state_dict")
    ap.add_argument("--train_only", type=int, default=0, help="train + save ckpt, skip eval")
    ap.add_argument("--noise_aug", type=int, default=0,
                    help="in-band noise augmentation during training (robust-training baseline): "
                         "per window, with prob 0.6 add white noise at SNR~U(-5,20) dB")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device

    raw = load_circor(DATA, limit=args.limit or None)
    mm = murmur_map()
    for r in raw:
        r["pid"] = r["id"].split("_")[0]; r["murmur"] = mm.get(r["pid"], "Unknown")
    pids = sorted(set(r["pid"] for r in raw))
    rng = np.random.RandomState(args.seed); rng.shuffle(pids)
    test_pids = set(pids[:len(pids) // 5])
    train = [r for r in raw if r["pid"] not in test_pids]
    test = [r for r in raw if r["pid"] in test_pids]
    print(f"[split seed={args.seed}] train={len(train)} test={len(test)}", flush=True)

    # real-noise bank: contiguous state-0 (unannotated/poor-quality) segments from TRAIN recordings,
    # at 1 kHz, >=0.5 s — real CirCor acquisition noise (same device/clinic), disjoint from test.
    noise_bank = []
    if args.noise_type == "real":
        for r in train:
            sd = to_deep(r["signal"]); sg = seg_to_deep(r["seg"], len(sd))
            z = (sg == 0).astype(int)
            d = np.diff(np.concatenate([[0], z, [0]]))
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                if b - a >= 500:
                    noise_bank.append(sd[a:b].astype(np.float32))
        print(f"[noise bank] {len(noise_bank)} real-noise segments from train", flush=True)

    def add_real_noise(sig, snr_db, idx):
        seg = noise_bank[idx % len(noise_bank)]
        reps = int(np.ceil(len(sig) / len(seg)))
        n = np.tile(seg, reps)[:len(sig)]
        p_sig = float(np.mean(sig.astype(np.float64) ** 2)) + 1e-12
        p_n = float(np.mean(n.astype(np.float64) ** 2)) + 1e-12
        n = n * np.sqrt((p_sig / (10 ** (snr_db / 10.0))) / p_n)
        return (sig + n).astype(np.float32)

    # train deep on clean 1000 Hz windows
    pairs = []
    for r in train:
        sd = to_deep(r["signal"]); sg = seg_to_deep(r["seg"], len(sd)); pairs.append((sd, sg))
    Xtr, Ytr = make_windows(pairs)
    if args.arch == "crnn":
        model = CRNN1D(in_ch=1, n_classes=5).to(dev)
    else:
        model = UNet1D(in_ch=1, n_classes=5, base=args.base, depth=args.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    print(f"[train] win={len(Xtr)}", flush=True)
    bs = 32
    for ep in range(1, args.epochs + 1):
        model.train(); perm = np.random.permutation(len(Xtr)); tot = 0
        for s in range(0, len(Xtr), bs):
            b = perm[s:s + bs]
            xb = Xtr[b]
            if args.noise_aug:                       # in-band noise augmentation (robust training)
                xb = xb.copy()
                for i in range(len(xb)):
                    if np.random.rand() < 0.6:
                        snr = np.random.uniform(-5, 20)
                        sig = xb[i, 0]
                        p_sig = float(np.mean(sig.astype(np.float64) ** 2)) + 1e-12
                        p_n = p_sig / (10 ** (snr / 10.0))
                        xb[i, 0] = (sig + np.random.normal(0, np.sqrt(p_n), sig.shape)).astype(np.float32)
            x = torch.from_numpy(xb).to(dev); y = torch.from_numpy(Ytr[b]).to(dev)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step(); tot += loss.item()
        if ep % 10 == 0 or ep == args.epochs:
            print(f"  ep{ep} loss={tot/(len(Xtr)//bs+1):.4f}", flush=True)
    model.eval()
    if args.save_ckpt:
        torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                    "base": args.base, "depth": args.depth, "seed": args.seed,
                    "test_pids": sorted(test_pids)}, args.save_ckpt)
        print(f"[ckpt] saved {args.save_ckpt}", flush=True)
    if args.train_only:
        print(f"[HYBRID_DONE seed={args.seed} train_only]", flush=True)
        return

    snrs = [None] + ([int(s) for s in args.test_snrs.split(",")] if args.test_snrs else [])
    pool = mp.Pool(args.n_jobs)

    outdir = "/data2/sjx/projects/pcg_seg/preds"
    os.makedirs(outdir, exist_ok=True)
    for snr in snrs:
        # Stage 1 (GPU, serial): deep forward -> posteriors + argmax; estimate HR (fast).
        posts, deep_p, gts, locs, mms, hrs, stis, confs = [], [], [], [], [], [], [], []
        for ri, r in enumerate(test):
            if snr is None:
                sd = to_deep(r["signal"]); hr_sig, hr_fs = r["signal"], RAW_FS
            elif args.noise_type == "real":   # real CirCor acquisition noise, in-band (1 kHz)
                sd = add_real_noise(to_deep(r["signal"]), snr, ri); hr_sig, hr_fs = sd, DEEP_FS
            elif args.noise_domain == "raw":
                noisy4000 = add_noise(r["signal"], snr)
                sd = to_deep(noisy4000); hr_sig, hr_fs = noisy4000, RAW_FS
            else:  # in-band: add noise to the 1kHz deep signal
                sd = add_noise(to_deep(r["signal"]), snr); hr_sig, hr_fs = sd, DEEP_FS
            post4, arg_feat = deep_posteriors(model, sd, dev)
            try:
                hr, sti = estimate_hr(hr_sig, hr_fs)
            except Exception:
                hr, sti = 75.0, 0.35
            posts.append(post4); deep_p.append(arg_feat)
            gts.append(resample_seg(r["seg"], len(arg_feat))); locs.append(r["loc"]); mms.append(r["murmur"])
            hrs.append(hr); stis.append(sti)
            confs.append(float(post4.max(1).mean()))   # deep posterior confidence (mean max-softmax)
        # Stage 2 (CPU, parallel): duration-Viterbi decode of deep posteriors.
        hyb_raw = pool.map(_decode_worker, [(posts[i], hrs[i], stis[i], FEAT_FS) for i in range(len(posts))])
        hyb_p = [hyb_raw[i] if hyb_raw[i] is not None else deep_p[i] for i in range(len(deep_p))]
        # align lengths
        for i in range(len(deep_p)):
            n = min(len(deep_p[i]), len(hyb_p[i]), len(gts[i]))
            deep_p[i] = deep_p[i][:n]; hyb_p[i] = hyb_p[i][:n]; gts[i] = gts[i][:n]

        tag = "clean" if snr is None else f"snr{snr}"
        # save predictions for offline analysis (gating sweep, metrics, figures) — no GPU re-runs
        sfx = (f"_{args.tag}" if args.tag else "")
        np.savez(f"{outdir}/preds_s{args.seed}_{args.noise_domain}_{tag}{sfx}.npz",
                 deep=np.array(deep_p, dtype=object), hyb=np.array(hyb_p, dtype=object),
                 gt=np.array(gts, dtype=object), conf=np.array(confs),
                 loc=np.array(locs), murmur=np.array(mms))
        d_f1, _ = per_sample_macro_f1(deep_p, gts)
        h_f1, _ = per_sample_macro_f1(hyb_p, gts)
        d_b, _ = boundary_f1(deep_p, gts, FEAT_FS)
        h_b, _ = boundary_f1(hyb_p, gts, FEAT_FS)
        # quick gated-hybrid sanity: per-recording, use hybrid where deep is unconfident
        confs_a = np.array(confs); tau = 0.7
        gate_use_hyb = confs_a < tau
        gated_p = [hyb_p[i] if gate_use_hyb[i] else deep_p[i] for i in range(len(deep_p))]
        g_f1, _ = per_sample_macro_f1(gated_p, gts)
        print(f"[HYB {tag} seed={args.seed}] deep_mF1={d_f1:.4f} hybrid_mF1={h_f1:.4f} "
              f"gated@.7_mF1={g_f1:.4f} (hyb_frac={gate_use_hyb.mean():.2f} conf_mean={confs_a.mean():.3f}) | "
              f"deep_bF1={d_b:.4f} hybrid_bF1={h_b:.4f}", flush=True)
        if snr is None:
            for axis, vals in [("loc", ["AV", "MV", "PV", "TV"]), ("murmur", ["Absent", "Present"])]:
                key = locs if axis == "loc" else mms
                for v in vals:
                    m = [key[i] == v for i in range(len(deep_p))]
                    if not any(m):
                        continue
                    dd, _ = per_sample_macro_f1([deep_p[i] for i in range(len(m)) if m[i]],
                                                [gts[i] for i in range(len(m)) if m[i]])
                    hh, _ = per_sample_macro_f1([hyb_p[i] for i in range(len(m)) if m[i]],
                                                [gts[i] for i in range(len(m)) if m[i]])
                    print(f"    {axis}={v}: deep={dd:.4f} hybrid={hh:.4f} n={sum(m)}", flush=True)
    pool.close(); pool.join()
    print(f"[HYBRID_DONE seed={args.seed} domain={args.noise_domain}]", flush=True)


if __name__ == "__main__":
    main()
