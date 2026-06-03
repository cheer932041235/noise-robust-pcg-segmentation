"""Fig C: clean-data fair benchmark, deep vs Springer HSMM, overall + stratified.
Reads results.json (single source of truth) — no hardcoded numbers."""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
import pubstyle as ps

R = json.load(open("/data2/sjx/projects/pcg_seg/results.json"))["benchmark_clean"]
FIGS = "/data2/sjx/projects/pcg_seg/figs"
keys = ["overall", "AV", "MV", "PV", "TV", "Absent", "Present"]
labels = ["overall", "AV", "MV", "PV", "TV", "murmur-", "murmur+"]


def val(d, k):
    return d[k][0] if k == "overall" else d[k]


deep = [val(R["deep"], k) for k in keys]
hsmm = [val(R["hsmm"], k) for k in keys]
deep_e = [R["deep"]["overall"][1] if k == "overall" else 0 for k in keys]
hsmm_e = [R["hsmm"]["overall"][1] if k == "overall" else 0 for k in keys]

x = np.arange(len(labels)); w = 0.38
plt.figure(figsize=(7, 4))
plt.bar(x - w / 2, deep, w, yerr=deep_e, capsize=3, color=ps.METHOD["deep"], label="Deep U-Net")
plt.bar(x + w / 2, hsmm, w, yerr=hsmm_e, capsize=3, color=ps.METHOD["HSMM"], label="Springer HSMM")
plt.xticks(x, labels); plt.ylim(0.55, 0.95)
plt.ylabel("per-sample macro-F1 (clean)")
plt.legend(); plt.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIGS}/figC_benchmark.pdf"); plt.savefig(f"{FIGS}/figC_benchmark.png")
print(f"saved figC (deep overall {deep[0]:.3f}, hsmm {hsmm[0]:.3f})")
