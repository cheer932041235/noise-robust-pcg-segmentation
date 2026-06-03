# -*- coding: utf-8 -*-
"""Compute the data needed for review MUST-FIX P1/P2/P3.
P3: gate-trigger fraction by SNR (white noise), 5-seed mean.
P1-equiv: gated macro-F1 by SNR over a tau grid -> show flat (insensitivity).
P1-tau_train: recalibrate tau* on TRAIN-clean (seeds with ckpt) + confirm gated unchanged.
P2: per-seed SD for real-noise / noise-aug / cross-arch tables.
All offline from saved preds except the tau_train GPU inference.
Writes review_fixes.json."""
import os, glob, json
import numpy as np
import sys
sys.path.insert(0, "/data2/sjx/projects/pcg_seg/src")
from hybrid_decode import cycle_violation_rate
from eval_pcg import per_sample_macro_f1

PRED = "/data2/sjx/projects/pcg_seg/preds"
CKPT = "/data2/sjx/projects/pcg_seg/ckpt"
SEEDS = [0, 1, 2, 3, 4]
WHITE_TAGS = [("clean", "clean"), ("15", "snr15"), ("10", "snr10"),
              ("5", "snr5"), ("0", "snr0"), ("-5", "snr-5")]
TAU_REPORT = 0.20
OUT = {}

def f1_of(pred_list, gt_list):
    return float(per_sample_macro_f1(list(pred_list), list(gt_list))[0])

def load_ext(seed, tag, suffix="ext"):
    fs = glob.glob(f"{PRED}/preds_s{seed}_deep_{tag}_{suffix}.npz")
    if not fs:
        return None
    d = np.load(fs[0], allow_pickle=True)
    keys = set(d.keys())
    if not {"deep", "gt"} <= keys:
        return None
    deep = [np.asarray(x).astype(int) for x in d["deep"]]
    gt = [np.asarray(x).astype(int) for x in d["gt"]]
    hyb = [np.asarray(x).astype(int) for x in d["hyb"]] if "hyb" in keys else None
    return deep, hyb, gt

# ---------- P3: gate-trigger fraction by SNR (white), 5-seed ----------
trig = {}
for snr_lbl, tag in WHITE_TAGS:
    fracs = []
    for s in SEEDS:
        r = load_ext(s, tag)
        if r is None:
            continue
        deep, _, _ = r
        viol = np.array([cycle_violation_rate(x) for x in deep])
        fracs.append(float(np.mean(viol > TAU_REPORT)))
    if fracs:
        trig[snr_lbl] = {"mean_frac": round(float(np.mean(fracs)), 3),
                         "sd": round(float(np.std(fracs)), 3), "n_seeds": len(fracs)}
OUT["P3_gate_trigger_fraction_white_tau0.20"] = trig

# ---------- P1-equiv: gated macro-F1 by SNR over tau grid ----------
TAU_GRID = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
equiv = {}
for snr_lbl, tag in WHITE_TAGS:
    row = {}
    for tau in TAU_GRID:
        per_seed = []
        for s in SEEDS:
            r = load_ext(s, tag)
            if r is None or r[1] is None:
                continue
            deep, hyb, gt = r
            viol = np.array([cycle_violation_rate(x) for x in deep])
            gated = [hyb[i] if viol[i] > tau else deep[i] for i in range(len(deep))]
            per_seed.append(f1_of(gated, gt))
        if per_seed:
            row[str(tau)] = round(float(np.mean(per_seed)), 4)
    equiv[snr_lbl] = row
OUT["P1_equiv_gated_F1_vs_tau_grid"] = equiv

# ---------- P2: per-seed SD for real / noiseaug / cross-arch ----------
def per_seed_table(suffix, tags, with_gated=True, tau=TAU_REPORT):
    res = {}
    for snr_lbl, tag in tags:
        deep_v, hyb_v, gat_v = [], [], []
        for s in SEEDS:
            r = load_ext(s, tag, suffix)
            if r is None:
                continue
            deep, hyb, gt = r
            deep_v.append(f1_of(deep, gt))
            if hyb is not None:
                hyb_v.append(f1_of(hyb, gt))
                if with_gated:
                    viol = np.array([cycle_violation_rate(x) for x in deep])
                    gated = [hyb[i] if viol[i] > tau else deep[i] for i in range(len(deep))]
                    gat_v.append(f1_of(gated, gt))
        cell = {}
        if deep_v:
            cell["deep"] = [round(np.mean(deep_v), 4), round(np.std(deep_v), 4), len(deep_v)]
        if hyb_v:
            cell["hybrid"] = [round(np.mean(hyb_v), 4), round(np.std(hyb_v), 4), len(hyb_v)]
        if gat_v:
            cell["gated"] = [round(np.mean(gat_v), 4), round(np.std(gat_v), 4), len(gat_v)]
        res[snr_lbl] = cell
    return res

REAL_TAGS = [("clean", "clean"), ("10", "snr10"), ("5", "snr5"), ("0", "snr0")]
OUT["P2_real_meanSD"] = per_seed_table("real", REAL_TAGS)
NAUG_TAGS = [("clean", "clean"), ("10", "snr10"), ("5", "snr5"), ("0", "snr0"), ("-5", "snr-5")]
OUT["P2_noiseaug_meanSD"] = per_seed_table("noiseaug", NAUG_TAGS, with_gated=False)
ARCH_TAGS = [("clean", "clean"), ("0", "snr0")]
OUT["P2_arch_b32d5_meanSD"] = per_seed_table("b32d5", ARCH_TAGS)
OUT["P2_arch_crnn_meanSD"] = per_seed_table("crnn", ARCH_TAGS)

# ---------- P1-tau_train: recalibrate tau* on TRAIN-clean (ckpt seeds) ----------
tau_train = {}
try:
    import torch
    from unet1d import UNet1D
    from pcg_loader import load_circor
    import run_hybrid as RH  # to_deep, deep_posteriors, seg_to_deep, DATA
    dev = "cuda"
    raw = load_circor(RH.DATA)
    for r in raw:
        r["pid"] = r["id"].split("_")[0]
    pids = sorted(set(r["pid"] for r in raw))
    for s in [0, 1, 2]:
        ck = f"{CKPT}/unet_s{s}.pt"
        if not os.path.exists(ck):
            continue
        st = torch.load(ck, map_location=dev)
        test_pids = set(st["test_pids"])
        # sanity: reproduce split from seed and compare
        rng = np.random.RandomState(s); pp = list(pids); rng.shuffle(pp)
        repro = set(pp[:len(pp) // 5])
        split_match = (repro == test_pids)
        train = [r for r in raw if r["pid"] not in test_pids]
        model = UNet1D(in_ch=1, n_classes=5, base=st.get("base", 16), depth=st.get("depth", 4)).to(dev)
        model.load_state_dict(st["state_dict"]); model.eval()
        viols = []
        for r in train:
            sd = RH.to_deep(r["signal"])
            _, argmax_feat = RH.deep_posteriors(model, sd, dev)
            viols.append(cycle_violation_rate(np.asarray(argmax_feat).astype(int)))
        tau_tr = float(np.percentile(viols, 95))
        # confirm gated F1 on test unchanged: tau_train vs 0.20, per SNR
        gated_cmp = {}
        for snr_lbl, tag in WHITE_TAGS:
            rr = load_ext(s, tag)
            if rr is None or rr[1] is None:
                continue
            deep, hyb, gt = rr
            v = np.array([cycle_violation_rate(x) for x in deep])
            g_tr = [hyb[i] if v[i] > tau_tr else deep[i] for i in range(len(deep))]
            g_02 = [hyb[i] if v[i] > 0.20 else deep[i] for i in range(len(deep))]
            gated_cmp[snr_lbl] = [round(f1_of(g_tr, gt), 4), round(f1_of(g_02, gt), 4)]
        tau_train[f"s{s}"] = {"tau_train": round(tau_tr, 4), "n_train": len(train),
                              "split_matches_ckpt": split_match,
                              "gated_F1_[tau_train, 0.20]_by_snr": gated_cmp}
    if tau_train:
        vals = [tau_train[k]["tau_train"] for k in tau_train]
        tau_train["_mean_tau_train"] = round(float(np.mean(vals)), 4)
except Exception as e:
    tau_train["_error"] = repr(e)
OUT["P1_tau_train_recalibration"] = tau_train

with open("/data2/sjx/projects/pcg_seg/review_fixes.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("REVIEW_FIXES_DONE")
print(json.dumps(OUT, indent=1)[:1500])
