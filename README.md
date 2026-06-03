<p align="right"><b>English</b> | <a href="README.zh.md">简体中文</a></p>

# Noise-Robust Heart-Sound Segmentation via Physiology-Gated Hybrid Decoding

Code and results for the paper *"Physiology-gated hybrid decoding for noise-robust heart-sound
segmentation: a fair benchmark of deep and classical models on CirCor"* (targeting *Signal, Image
and Video Processing*, Springer). The manuscript itself is not hosted here; this repository is the
**reproducibility package** — code, every reported number (`results.json`), and the figures.

---

## TL;DR

Segmenting a phonocardiogram (PCG) into its four cardiac states (S1 → systole → S2 → diastole) is
the front end for downstream murmur analysis. We show that:

1. On a **fair, patient-disjoint benchmark**, a deep 1-D U-Net clearly beats the classical Springer
   LR-HSMM on clean data (macro-F1 **0.877 vs 0.729**).
2. But under **in-band noise the deep model collapses** — and does so *confidently* (calibration
   error 0.011 → 0.257) and *physiologically impossibly* (cardiac-cycle violation rate 0.06 → 0.22).
3. That failure is **measurable without labels**, so we turn it into a fix: a **training-free hybrid
   decoder** feeds the deep posteriors as emissions into the HSMM duration-Viterbi decoder, gated by
   the label-free violation rate. The gated hybrid is **Pareto-dominant over the clean-trained deep
   model** (never worse clean, much better under noise).

> The honest framing: this is **not** "classical beats deep". Deep wins on clean data. The value is
> a fair map of where each fails, and a training-free decoder that combines their complementary
> strengths across the whole signal-to-noise range.

---

## 1. Problem & gap

Deep PCG segmenters report strong accuracy on the large, real-world **CirCor DigiScope 2022**
dataset, but two questions were open:

- **No like-for-like comparison** of deep vs the classical HSMM existed on CirCor — prior numbers
  came from different datasets, splits and metrics, so they were not comparable.
- **Robustness was uncharacterized** — how deep segmenters behave across auscultation position,
  murmur, and (above all) acquisition noise was unknown, even though these vary in every clinic.

We address both, and turn the findings into a method.

## 2. Method

**(a) Fair benchmark.** 1-D U-Net (1.86 M params) vs Springer LR-HSMM on CirCor, patient-disjoint
splits, multi-seed CIs, stratified by the four auscultation positions and murmur status.

**(b) Failure mechanism.** Under in-band noise the deep model is *confidently wrong* (reliability
diagram / ECE) and its errors are *physiologically visible*: the cardiac-cycle **violation rate** —
a label-free measure of how often the predicted state sequence breaks the S1→systole→S2→diastole
cycle — rises monotonically with noise.

**(c) Training-free hybrid + plausibility gate.** Reuse the deep per-sample posteriors `q_t(s)` as
the emission model inside the classical HSMM duration-dependent Viterbi decoder. This imposes
physiological duration and cyclic-order priors on the deep output **without any retraining**. A gate
invokes the hybrid only when the deep output is implausible (violation rate > τ\*), with τ\*
calibrated on **training-split clean** recordings alone (τ\* = 0.20).

## 3. Results

**Clean benchmark (per-sample macro-F1, 5 seeds).** Deep dominates at every stratum.

| | Deep U-Net | Springer HSMM |
|---|---|---|
| Overall | **0.877 ± 0.007** | 0.729 ± 0.012 |

**In-band white-noise robustness (per-sample macro-F1).** Deep collapses; the hybrid beats *both*
individual methods under noise; the gate preserves clean accuracy. (Deep/hybrid/gated 5-seed, HSMM
3-seed means.)

| SNR | Deep | HSMM | Hybrid | Gated |
|---|---|---|---|---|
| clean | **0.867** | 0.729 | 0.849 | **0.867** |
| 10 dB | 0.827 | 0.697 | 0.831 | 0.830 |
| 5 dB | 0.747 | 0.657 | **0.790** | 0.758 |
| 0 dB | 0.474 | 0.580 | **0.686** | 0.587 |
| −5 dB | 0.048 | — | **0.472** | 0.276 |

- **Failure signature** at 0 dB: ECE 0.011 → 0.257 (confidence 0.65 at accuracy 0.40); cycle-violation 0.06 → 0.22.
- **Significance**: paired Wilcoxon (per recording, pooled across seeds, n = 1922) — gated vs deep at 0 dB Δ = +0.098, p ≈ 6e-149; clean Δ = +0.002, p = 0.33 (statistically indistinguishable → Pareto-dominant).
- **Gate operating point**: fires on 5.0 % of clean recordings (by the 95th-percentile design), rising to 53.3 % at 0 dB.

**Generality & honesty.**

| Check | Result |
|---|---|
| Real CirCor acquisition noise (0 dB) | hybrid 0.663 > deep 0.587 & HSMM 0.556; gate keeps clean 0.868 |
| Stronger / second architecture (0 dB) | wide U-Net 0.44 → hybrid 0.68; CNN-BiLSTM 0.42 → hybrid 0.69 |
| Noise-augmented training (0 dB) | **0.801 — beats the training-free hybrid (0.686)**; we say so plainly |
| Cross-corpus PhysioNet/CinC 2016 (label-free) | deep violation 0.254; 64 % exceed τ\* → the failure signal transfers |

We position the hybrid honestly: when retraining with representative noise is possible, **noise
augmentation is stronger**. The hybrid's orthogonal value is being **training-free and
model-agnostic** — it can be bolted onto any already-deployed segmenter with no retraining.

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | training, hybrid decoding, evaluation, ablation and figure scripts (PyTorch) |
| `figs/` | figures (PDF/PNG) |
| `results.json` | single source of truth for every number reported above |

**Key scripts** (`src/`): `train_pcg.py` / `run_hybrid.py` (train U-Net/CRNN; evaluate deep, hybrid,
gated, noise-aug across SNR), `hybrid_decode.py` (deep-posterior emissions + duration-Viterbi;
cycle-violation rate), `run_hsmm.py` (Springer LR-HSMM), `eval_pcg.py` (macro-F1 + ±60 ms
boundary-onset F1), `make_*.py` (figures and tables), `review_fixes.py` (gate-trigger fractions and
train-split τ\* recalibration).

## Data

**CirCor DigiScope 2022** and **PhysioNet/CinC 2016** are obtained from PhysioNet and are *not*
redistributed here.

## Reproduce

1-D U-Net: 1.86 M params (base width 16, depth 4, 5 classes), AdamW (lr 1e-3, wd 1e-4),
cross-entropy, batch 32, 30 epochs on 4096-sample (~4 s) windows. Splits are patient-disjoint;
deep/HSMM are 5-seed, hybrid 3-seed means (seeds 0–4).

> **Note:** scripts use absolute paths from the development server (e.g. `/data2/sjx/projects/pcg_seg`,
> conda env `structecg`). Adjust to your environment before running.

## Citation

To be added upon publication.

---

## 科研辅导 · 合作

需要**科研辅导、科研合作**，欢迎联系 **疏锦行**　微信：**shujinxing777**
