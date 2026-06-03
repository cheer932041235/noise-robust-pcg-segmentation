"""Fig D: real-noise robustness (CirCor acquisition noise), 4 methods.
Reads results.json (single source of truth) — no hardcoded numbers."""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps

R = json.load(open("/data2/sjx/projects/pcg_seg/results.json"))["noise_inband_real"]
FIGS = "/data2/sjx/projects/pcg_seg/figs"
tags = ["clean", "snr10", "snr5", "snr0"]
labels = ["clean", "10 dB", "5 dB", "0 dB"]
x = np.arange(len(tags))


def curve(key):
    return [R[key].get(t, np.nan) for t in tags]


plt.figure(figsize=(6, 4))
plt.plot(x, curve("deep"), "o-", color=ps.METHOD["deep"], label="deep")
plt.plot(x, curve("hybrid"), "s--", color=ps.METHOD["hybrid"], label="hybrid")
plt.plot(x, curve("gated"), "^-", color=ps.METHOD["gated"], label="gated")
plt.plot(x, curve("hsmm"), "d:", color=ps.METHOD["HSMM"], label="HSMM")
plt.xticks(x, labels); plt.gca().invert_xaxis()
plt.xlabel("test SNR (real CirCor acquisition noise)")
plt.ylabel("per-sample macro-F1")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIGS}/figD_realnoise.pdf"); plt.savefig(f"{FIGS}/figD_realnoise.png")
print(f"saved figD (0dB: deep {R['deep']['snr0']}, hybrid {R['hybrid']['snr0']}, hsmm {R['hsmm']['snr0']})")
