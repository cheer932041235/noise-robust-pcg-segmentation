"""
The 'show-don't-tell' qualitative panel (real data) for Fig 1: on a real noisy PCG recording,
the deep model fragments into a physiologically impossible state sequence, while the duration-prior
hybrid recovers a clean cardiac cycle. Trains the U-Net, picks the test recording that best
illustrates collapse->rescue, and renders:
  (A) noisy PCG waveform with three label strips: GT / deep / hybrid (+ red cycle-violation marks)
  (B) deep per-sample posterior heatmap under noise (shows confident-but-wrong)
IEEE palette, vector PDF. Also dumps the chosen recording's arrays to npz for reuse in the
full architecture figure.
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps
from run_hybrid import to_deep, seg_to_deep, make_windows, deep_posteriors, RAW_FS, DEEP_FS, FEAT_FS
from unet1d import UNet1D
from pcg_loader import load_circor
from eval_pcg import add_noise, per_sample_macro_f1
from hybrid_decode import hybrid_decode, cycle_violation_rate

DATA = "/data2/sjx/projects/pcg_seg/data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
SNR = 0
dev = "cuda"


def main():
    torch.manual_seed(0); np.random.seed(0)
    raw = load_circor(DATA)
    for r in raw:
        r["pid"] = r["id"].split("_")[0]
    pids = sorted(set(r["pid"] for r in raw))
    rng = np.random.RandomState(0); rng.shuffle(pids)
    test_pids = set(pids[:len(pids) // 5])
    train = [r for r in raw if r["pid"] not in test_pids]
    test = [r for r in raw if r["pid"] in test_pids]

    pairs = [(to_deep(r["signal"]), seg_to_deep(r["seg"], len(to_deep(r["signal"])))) for r in train]
    Xtr, Ytr = make_windows(pairs)
    model = UNet1D(in_ch=1, n_classes=5, base=16, depth=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    print("[train]", flush=True)
    bs = 32
    for ep in range(30):
        model.train(); perm = np.random.permutation(len(Xtr))
        for s in range(0, len(Xtr), bs):
            b = perm[s:s + bs]
            x = torch.from_numpy(Xtr[b]).to(dev); y = torch.from_numpy(Ytr[b]).to(dev)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    model.eval()

    # pick the recording best illustrating collapse->rescue (8-14 s, all 4 states, large hybrid gain)
    best = None
    for r in test:
        sd_clean = to_deep(r["signal"]); dur = len(sd_clean) / DEEP_FS
        if not (8 <= dur <= 14):
            continue
        gt = seg_to_deep(r["seg"], int(round(len(sd_clean) * FEAT_FS / DEEP_FS)))
        if not all((gt == c).any() for c in (1, 2, 3, 4)):
            continue
        sd_noisy = add_noise(sd_clean, SNR)
        post, argd = deep_posteriors(model, sd_noisy, dev)
        try:
            hyb = hybrid_decode(post, sd_noisy, sampling_frequency=DEEP_FS, feature_frequency=FEAT_FS)
        except Exception:
            continue
        n = min(len(argd), len(hyb), len(gt))
        df = per_sample_macro_f1([argd[:n]], [gt[:n]])[0]
        hf = per_sample_macro_f1([hyb[:n]], [gt[:n]])[0]
        gain = hf - df
        if best is None or gain > best["gain"]:
            best = dict(gain=gain, r=r, sd_clean=sd_clean, sd_noisy=sd_noisy,
                        post=post, argd=argd[:n], hyb=hyb[:n], gt=gt[:n], n=n, df=df, hf=hf)
    r = best
    print(f"[pick] {r['r']['id']} loc={r['r']['loc']} gain={r['gain']:.3f} deep={r['df']:.3f} hyb={r['hf']:.3f}", flush=True)
    np.savez(f"{FIGS}/qual_example.npz", **{k: v for k, v in r.items()
             if k in ("sd_clean", "sd_noisy", "post", "argd", "hyb", "gt")}, id=r["r"]["id"])

    # ----- render -----
    n = r["n"]; t = np.arange(n) / FEAT_FS
    wav = r["sd_noisy"][:int(n * DEEP_FS / FEAT_FS)]
    tw = np.arange(len(wav)) / DEEP_FS
    cmap = ListedColormap([ps.STATE[i] for i in (0, 1, 2, 3, 4)])

    fig = plt.figure(figsize=(7.0, 4.2))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.0, 0.5, 0.5, 0.5, 1.6], hspace=0.35)

    axw = fig.add_subplot(gs[0]); axw.plot(tw, wav, color="#333333", lw=0.6)
    axw.set_xlim(0, t[-1]); axw.set_yticks([]); axw.set_ylabel("PCG\n(0 dB)")
    axw.set_title(f"Deep fragments under noise; the duration prior restores a valid cardiac cycle "
                  f"(rec {r['r']['id']}, {r['r']['loc']})", fontsize=8)
    axw.grid(False); axw.spines["left"].set_visible(False)

    def strip(ax, seq, label, mark_viol=False):
        ax.imshow(seq[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=4,
                  extent=[0, t[-1], 0, 1], interpolation="nearest")
        ax.set_yticks([0.5]); ax.set_yticklabels([label], fontsize=7)
        ax.set_xticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        if mark_viol:
            sseq = seq[seq > 0]
            chg = np.where(np.diff(seq) != 0)[0]
            for i in chg:
                a, b = seq[i], seq[i + 1]
                if a in (1, 2, 3, 4) and b in (1, 2, 3, 4) and ({1: 2, 2: 3, 3: 4, 4: 1}[a] != b):
                    ax.plot([t[i], t[i]], [0, 1], color="#D62728", lw=1.0)

    strip(fig.add_subplot(gs[1]), r["gt"], "GT")
    strip(fig.add_subplot(gs[2]), r["argd"], f"deep ({r['df']:.2f})", mark_viol=True)
    strip(fig.add_subplot(gs[3]), r["hyb"], f"hybrid ({r['hf']:.2f})")

    axp = fig.add_subplot(gs[4])
    im = axp.imshow(r["post"][:n].T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, t[-1], 0.5, 4.5], vmin=0, vmax=1, interpolation="nearest")
    axp.set_yticks([1, 2, 3, 4]); axp.set_yticklabels([ps.STATE_NAMES[i] for i in (1, 2, 3, 4)])
    axp.set_ylabel("deep posterior"); axp.set_xlabel("time (s)"); axp.grid(False)
    cax = axp.inset_axes([1.01, 0, 0.012, 1]); fig.colorbar(im, cax=cax)

    legend = [Patch(facecolor=ps.STATE[i], label=ps.STATE_NAMES[i]) for i in (1, 2, 3, 4)]
    legend.append(plt.Line2D([0], [0], color="#D62728", lw=1.2, label="cycle violation"))
    axw.legend(handles=legend, loc="upper right", ncol=5, fontsize=6,
               bbox_to_anchor=(1.0, 1.32), handlelength=1.2, columnspacing=1.0)

    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/figQ_qualitative.{ext}")
    plt.close(fig)
    print(f"saved {FIGS}/figQ_qualitative.pdf/.png", flush=True)


if __name__ == "__main__":
    main()
