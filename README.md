# Noise-Robust Heart-Sound Segmentation via Physiology-Gated Hybrid Decoding

Code and resources for the paper
**"Physiology-gated hybrid decoding for noise-robust heart-sound segmentation: a fair benchmark of deep and classical models on CirCor"**
(targeting *Signal, Image and Video Processing*, Springer).

## Summary

- A fair, **patient-disjoint** benchmark of a 1-D U-Net vs. the classical Springer
  logistic-regression hidden semi-Markov model (LR-HSMM) on **CirCor DigiScope 2022**, with
  multi-seed confidence intervals.
- A failure-mode analysis: under **in-band noise** the deep model is *confidently wrong*
  (calibration error 0.011 → 0.257) and *physiologically implausible* (cardiac-cycle violation
  rate 0.06 → 0.22).
- A **training-free hybrid decoder** — deep per-sample posteriors used as emissions in the HSMM
  duration-dependent Viterbi decoder — plus a label-free **plausibility gate**. The gated hybrid
  is Pareto-dominant over the deep model and robust under heavy noise.
- Comparisons across **real acquisition noise**, **architectures** (U-Net, CNN-BiLSTM),
  **noise-augmented training**, and **cross-corpus** generalization (PhysioNet/CinC 2016).

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | training, hybrid decoding, evaluation, ablation and figure scripts (PyTorch) |
| `paper/` | LaTeX source (`main.tex`, Springer `sn-jnl` class), compiled `main.pdf`, `refs.bib` |
| `figs/` | figures used in the paper |
| `results.json` | single source of truth for every number reported in the paper |

## Key scripts (`src/`)

- `train_pcg.py`, `run_hybrid.py` — train the U-Net / CRNN; evaluate deep, hybrid, gated, and the
  noise-augmented baseline (`--noise_aug 1`) across SNR.
- `hybrid_decode.py` — deep-posterior emissions + duration-Viterbi decode; cycle-violation rate.
- `run_hsmm.py` — Springer LR-HSMM baseline.
- `eval_pcg.py` — per-sample macro-F1 and ±60 ms boundary-onset F1.
- `make_*.py` — figures and result tables (noise curves, calibration, ablations, cross-corpus).

## Data

- **CirCor DigiScope 2022** and **PhysioNet/CinC 2016** are obtained from PhysioNet and are *not*
  redistributed here.

## Reproduce

1-D U-Net: 1.86 M parameters (base width 16, depth 4, 5 output classes), AdamW (lr 1e-3, weight
decay 1e-4), cross-entropy, batch 32, 30 epochs on 4096-sample (~4 s) windows. All splits are
patient-disjoint; deep/HSMM results are 5-seed and hybrid 3-seed means (seeds 0–4).

> **Note:** the scripts use absolute paths from the development server (e.g.
> `/data2/sjx/projects/pcg_seg`, conda env `structecg`). Adjust these to your environment before
> running.

## Citation

To be added upon publication.
