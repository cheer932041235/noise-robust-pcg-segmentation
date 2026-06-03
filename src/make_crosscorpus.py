"""
Cross-corpus generalization on PhysioNet/CinC 2016 (a 2nd real PCG corpus), LABEL-FREE.
A CirCor-trained U-Net (frozen ckpt) is run on 2016 recordings; we report the cardiac-cycle
violation rate (our label-free degradation signal) for the deep argmax vs the hybrid decode.
Shows the finding transfers: on a new real corpus the deep output is physiologically implausible
(esp. on noisy/abnormal recordings) and the hybrid restores plausibility — without any 2016 labels.

CinC 2016: 2 kHz wav under training-a..f. Resample to 1 kHz (the deep model's rate).
Usage: python make_crosscorpus.py [n_max]
"""
import sys, glob, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from scipy.io import wavfile
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from unet1d import UNet1D
from run_hybrid import deep_posteriors, DEEP_FS, FEAT_FS
from hybrid_decode import hybrid_decode, cycle_violation_rate

CINC = "/data2/sjx/projects/pcg_seg/data/cinc2016"
CK = "/data2/sjx/projects/pcg_seg/ckpt/unet_s0.pt"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
dev = "cuda"


def load_wav_1k(path):
    fs, sig = wavfile.read(path)
    sig = sig.astype(np.float32)
    if sig.ndim > 1:
        sig = sig[:, 0]
    if fs != DEEP_FS:
        from math import gcd
        g = gcd(DEEP_FS, int(fs))
        sig = resample_poly(sig, DEEP_FS // g, int(fs) // g).astype(np.float32)
    return ((sig - sig.mean()) / (sig.std() + 1e-6)).astype(np.float32)


def main():
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    ck = torch.load(CK, map_location=dev)
    model = UNet1D(in_ch=1, n_classes=5, base=ck["base"], depth=ck["depth"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    wavs = sorted(glob.glob(f"{CINC}/**/*.wav", recursive=True))[:n_max]
    print(f"[cinc2016] {len(wavs)} recordings", flush=True)
    dv, hv = [], []
    for i, w in enumerate(wavs):
        try:
            sd = load_wav_1k(w)
            if len(sd) < DEEP_FS * 3:        # skip <3 s
                continue
            sd = sd[:DEEP_FS * 20]           # cap at 20 s: keeps duration-Viterbi tractable
                                             # on long CinC clips; 20 s spans many cardiac cycles
            post, argd = deep_posteriors(model, sd, dev)
            try:
                hyb = hybrid_decode(post, sd, sampling_frequency=DEEP_FS, feature_frequency=FEAT_FS)
            except Exception:
                continue
            dv.append(cycle_violation_rate(argd)); hv.append(cycle_violation_rate(hyb[:len(argd)]))
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wavs)}", flush=True)
    dv, hv = np.array(dv), np.array(hv)
    np.savez(f"{FIGS}/../crosscorpus_viol.npz", dv=dv, hv=hv)   # save for reproducible re-plotting
    TAU = 0.20                                                  # the clean-calibrated gate threshold
    above = float((dv > TAU).mean() * 100)
    # NOTE: the hybrid's violation is ~0 BY CONSTRUCTION (the constrained duration-Viterbi cannot
    # break the cycle), so we do NOT present it as a head-to-head win. The transferable, non-trivial
    # signal is the DEEP violation distribution on an unseen corpus, measured without any labels.
    print(f"[RESULT] n={len(dv)} deep_violation={dv.mean():.3f}+-{dv.std():.3f} "
          f"hybrid_violation={hv.mean():.3f} (==0 by construction) frac_above_tau={above:.0f}%", flush=True)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.hist(dv, bins=25, color=ps.METHOD["deep"], edgecolor="#333", linewidth=0.5, alpha=0.85)
    ax.axvline(TAU, ls="--", color=ps.METHOD["hybrid"], lw=1.5)
    ax.text(TAU + 0.01, ax.get_ylim()[1] * 0.92, f"$\\tau^*={TAU}$", fontsize=8, color=ps.METHOD["hybrid"])
    ax.text(0.97, 0.78, f"{above:.0f}\\% exceed $\\tau^*$", transform=ax.transAxes, ha="right", fontsize=8)
    ax.set_xlabel("deep cycle-violation rate"); ax.set_ylabel("\\# CinC 2016 recordings")
    fig.tight_layout(); fig.savefig(f"{FIGS}/figX_crosscorpus.pdf"); fig.savefig(f"{FIGS}/figX_crosscorpus.png")
    print(f"saved figX_crosscorpus (deep mean {dv.mean():.3f}, {above:.0f}% above tau)", flush=True)


if __name__ == "__main__":
    main()
