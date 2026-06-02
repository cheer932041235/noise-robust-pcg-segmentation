"""
Input-gradient saliency of the deep model, clean vs 0 dB (loads a frozen checkpoint).
Clean: saliency concentrates on the S1/S2 acoustic transients (the model listens to the
right places). Under noise: saliency scatters/diffuses -> the model no longer locks onto the
sounds. Complements the calibration figure as mechanistic evidence.
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from unet1d import UNet1D
from pcg_loader import load_circor
from eval_pcg import add_noise
from run_hybrid import to_deep, seg_to_deep, DEEP_FS, FEAT_FS

DATA = "/data2/sjx/projects/pcg_seg/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
CK = "/data2/sjx/projects/pcg_seg/ckpt/unet_s0.pt"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
dev = "cuda"


def smooth(a, w=25):
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def saliency(model, sd):
    x = torch.from_numpy(sd[None, None, :]).to(dev).requires_grad_(True)
    logits = model(x)[0]                       # [5, N]
    model.zero_grad()
    logits.max(0).values.sum().backward()      # saliency of the chosen (max) class over time
    s = x.grad.abs()[0, 0].detach().cpu().numpy()
    s = smooth(s); return s / (s.max() + 1e-9)


def main():
    ck = torch.load(CK, map_location=dev)
    model = UNet1D(in_ch=1, n_classes=5, base=ck["base"], depth=ck["depth"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    raw = load_circor(DATA)
    test = [r for r in raw if r["id"].split("_")[0] in set(ck["test_pids"])]
    # pick a clean, ~6-9 s, all-4-state recording for a legible illustration
    rec = next(r for r in test if 6 <= len(r["signal"]) / r["fs"] <= 9
               and all((seg_to_deep(r["seg"], 400) == c).any() for c in (1, 2, 3, 4)))
    sd = to_deep(rec["signal"]); gt = seg_to_deep(rec["seg"], int(round(len(sd) * FEAT_FS / DEEP_FS)))
    sd0 = add_noise(sd, 0)
    sal_c, sal_n = saliency(model, sd), saliency(model, sd0)
    t = np.arange(len(sd)) / DEEP_FS
    tf = np.arange(len(gt)) / FEAT_FS
    s1 = tf[gt == 1]; s2 = tf[gt == 3]

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.4), sharex=True)
    for ax, sal, sig, ttl in [(axes[0], sal_c, sd, "clean: saliency locks onto S1/S2"),
                              (axes[1], sal_n, sd0, "0 dB: saliency scatters")]:
        ax.plot(t, sig / (np.abs(sig).max() + 1e-9), color="#999", lw=0.4, zorder=1)
        ax.fill_between(t, 0, sal, color=ps.METHOD["deep"], alpha=0.55, zorder=2, lw=0)
        for x0 in s1:
            ax.axvline(x0, color=ps.STATE[1], lw=0.5, alpha=0.5, zorder=0)
        for x0 in s2:
            ax.axvline(x0, color=ps.STATE[3], lw=0.5, alpha=0.5, zorder=0)
        ax.set_yticks([]); ax.set_ylim(-1.05, 1.05); ax.set_title(ttl, fontsize=8)
        ax.set_ylabel("saliency / PCG", fontsize=7)
    axes[1].set_xlabel("time (s)", fontsize=8); axes[1].set_xlim(0, t[-1])
    from matplotlib.lines import Line2D
    axes[0].legend([Line2D([0], [0], color=ps.STATE[1], lw=2), Line2D([0], [0], color=ps.STATE[3], lw=2)],
                   ["S1 (GT)", "S2 (GT)"], fontsize=6, loc="upper right", ncol=2)
    fig.savefig(f"{FIGS}/figI_saliency.pdf"); fig.savefig(f"{FIGS}/figI_saliency.png")
    print(f"saved figI_saliency (rec {rec['id']})")


if __name__ == "__main__":
    main()
