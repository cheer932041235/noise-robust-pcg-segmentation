"""
Reliability diagram + ECE for the deep model, clean vs 0 dB in-band noise (loads frozen
checkpoints). Visualizes the 'confidently wrong' finding: under noise the model stays confident
(accuracy << confidence) -> the curve drops far below the diagonal, ECE explodes.
This is the mechanistic figure motivating the plausibility gate.
"""
import sys, glob
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
CK = "/data2/sjx/projects/pcg_seg/ckpt"
FIGS = "/data2/sjx/projects/pcg_seg/figs"
dev = "cuda"
NBINS = 12


@torch.no_grad()
def conf_correct(model, recs, snr):
    """pooled per-sample (confidence=max softmax, correct?) over annotated samples."""
    conf, corr = [], []
    for r in recs:
        sd = to_deep(r["signal"])
        if snr is not None:
            sd = add_noise(sd, snr)
        gt = seg_to_deep(r["seg"], int(round(len(sd) * FEAT_FS / DEEP_FS)))
        x = torch.from_numpy(sd[None, None, :]).to(dev)
        prob = torch.softmax(model(x)[0], 0).cpu().numpy()       # [5, N]
        Tf = len(gt); idx = (np.arange(Tf) * (prob.shape[1] / Tf)).astype(int).clip(0, prob.shape[1] - 1)
        pf = prob[:, idx]                                         # [5, Tf]
        pred = pf.argmax(0); c = pf.max(0)
        ann = gt > 0
        conf.append(c[ann]); corr.append((pred[ann] == gt[ann]).astype(float))
    return np.concatenate(conf), np.concatenate(corr)


def reliability(conf, corr):
    edges = np.linspace(0, 1, NBINS + 1)
    xs, ys, ws = [], [], []
    ece = 0.0; N = len(conf)
    for i in range(NBINS):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < NBINS - 1 else conf <= 1.0001)
        if m.sum() == 0:
            continue
        c, a = conf[m].mean(), corr[m].mean()
        xs.append(c); ys.append(a); ws.append(m.sum())
        ece += m.sum() / N * abs(a - c)
    return np.array(xs), np.array(ys), ece


def main():
    raw = load_circor(DATA)
    cks = sorted(glob.glob(f"{CK}/unet_s*.pt"))
    print(f"[ckpts] {len(cks)}", flush=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=0.9, label="perfect calibration")
    for snr, col, lab in [(None, ps.METHOD["deep"], "clean"), (0, "#D62728", "0 dB (in-band)")]:
        Cc, Co = [], []
        for ckp in cks:
            ck = torch.load(ckp, map_location=dev)
            m = UNet1D(in_ch=1, n_classes=5, base=ck["base"], depth=ck["depth"]).to(dev)
            m.load_state_dict(ck["state_dict"]); m.eval()
            test = [r for r in raw if r["id"].split("_")[0] in set(ck["test_pids"])]
            c, o = conf_correct(m, test, snr)
            Cc.append(c); Co.append(o)
        conf, corr = np.concatenate(Cc), np.concatenate(Co)
        xs, ys, ece = reliability(conf, corr)
        ax.plot(xs, ys, "o-", color=col, ms=4, label=f"{lab}  (ECE={ece:.3f}, conf̄={conf.mean():.2f}, acc={corr.mean():.2f})")
    ax.set_xlabel("confidence (max softmax)", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.set_title("Deep is well-calibrated when clean,\nconfidently wrong under noise", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=6, loc="upper left")
    fig.savefig(f"{FIGS}/figH_calibration.pdf"); fig.savefig(f"{FIGS}/figH_calibration.png")
    print("saved figH_calibration", flush=True)


if __name__ == "__main__":
    main()
