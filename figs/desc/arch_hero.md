# MODULE LIST — Fig. 1 Full-system architecture (physiology-gated hybrid PCG decoder)

Mimic the EXACT visual language of the reference image: top row of circular ①②③④⑤ numbered
badges, one soft macaron pastel rounded-rectangle zone per stage, REAL embedded mini-diagrams
inside every stage (not empty boxes), thin grey connector arrows between stages, and a horizontal
LEGEND band at the bottom. Flat NeurIPS/ICLR style — no gradients, no shadows, no 3-D, no textures.
Sans-serif (Helvetica/Arial) labels, all horizontal. **ALL text must be in English only.**

## MODULE 1: GLOBAL CANVAS & LAYOUT GRID
- Canvas 2048×1024 px, wide aspect, 300 DPI, pure white background, 48 px margins.
- Primary flow axis: LEFT → RIGHT.
- 5-column grid, each column a rounded macaron zone with a colored header band carrying a circular
  white number badge + stage title:
  - Col ① Input (palest blue #EAF2FB / header #3B6EA5)
  - Col ② 1-D U-Net backbone (widest column, light blue #DCE9F6 / header #2F5E96) — largest & most emphasized
  - Col ③ Per-sample posteriors (mint #E7F3EA / header #3E8E54)
  - Col ④ Physiology gate (pale green #EAF7EA / header #2CA02C)
  - Col ⑤ Dual decoder → output (peach #F6E3D6 / header #C66B3D)
- Inter-column gutter 40 px; thin grey arrows (#6B7280, 2 px) connect adjacent stages.
- **IMPORTANT**: All pixel coordinates here are INTERNAL LAYOUT GUIDES only. Do NOT render any
  coordinates, pixel values, or axis numbers as visible text.
- Z-order: zone backgrounds z=0, sub-boxes z=1, arrows z=2, text z=3.

## MODULE 2: ① INPUT
- Header badge "①" + title "Noisy PCG".
- Mini-plot: a single-channel heart-sound WAVEFORM (dark blue #3B6EA5 line) with two repeating
  S1/S2 bursts, slightly noisy baseline.
- Small caption under it in italic grey: "in-band noise (SNR 0 dB)".
- Outgoing arrow to ②.

## MODULE 3: ② 1-D U-NET BACKBONE (widest, most emphasized)
- Header badge "②" + title "1-D U-Net".
- Embedded mini U-Net ARCHITECTURE diagram: a symmetric encoder–decoder of stacked 1-D feature-map
  slabs. Encoder DOWN-sampling path on the left, bottleneck at the bottom centre, decoder UP-sampling
  path on the right, drawn as a shallow "U".
  - Encoder slabs labelled with channel counts "16 → 32 → 64", tall→short (temporal length halves).
  - Bottleneck slab labelled "128".
  - Decoder slabs labelled "64 → 32 → 16", short→tall.
  - Three horizontal dashed grey SKIP-CONNECTION arrows linking matching encoder and decoder levels,
    annotated "skip (concat)".
  - Encoder slabs in blue #4C78B8, bottleneck in slate #5B4B8A, decoder slabs in light blue #9EC1E4.
- Small italic note: "Conv1d→BN→ReLU ×2 per level; ↓ stride-2 pool; ↑ transposed-conv; 1.86 M params".
- Outgoing arrow to ③, labelled "q_t(s)".

## MODULE 4: ③ PER-SAMPLE POSTERIORS
- Header badge "③" + title "Posteriors".
- Mini posteriorgram HEATMAP: 4 rows (S1, systole, S2, diastole) × time, magma colormap, one bright
  cell per column showing the soft class distribution.
- Italic note: "softmax over 4 cardiac states".
- Outgoing arrow to ④.

## MODULE 5: ④ PHYSIOLOGY GATE
- Header badge "④" + title "Physiology gate".
- A green DIAMOND decision node containing "cycle-violation  ρ > τ ?".
- Beside it a tiny horizontal bar/needle gauge showing ρ relative to a dashed threshold line τ
  (label "τ = 95th pct of clean ρ").
- Italic note: "label-free, calibrated on clean data only".
- TWO outgoing branches to ⑤: an upper GREEN arrow "ρ ≤ τ (plausible)" and a lower ORANGE arrow
  "ρ > τ (violation)".

## MODULE 6: ⑤ DUAL DECODER → OUTPUT
- Header badge "⑤" + title "Decode → segmentation".
- Upper sub-box (light blue #DCE9F6, stroke #3B6EA5): "deep argmax" — fed by the green branch.
- Lower sub-box (peach #F6E3D6, stroke #C66B3D): "HSMM duration-Viterbi" — fed by the orange branch;
  inside it a tiny duration-model / Viterbi TRELLIS mini-icon and a small Gaussian duration curve.
- Italic connector note between gate and HSMM: "deep posteriors reused as HSMM emissions".
- Both sub-boxes merge with arrows into a final OUTPUT mini-panel: a colored SEGMENTATION STRIP with
  the repeating cycle S1(dark blue)→systole(light blue)→S2(orange)→diastole(peach), labelled
  "valid cardiac cycle", with a green ✓.
- Title text near output: "training-free, model-agnostic decoder".

## FINAL MODULE: GLOBAL ARROW ROUTING & LEGEND
- Arrow ①→②→③→④ horizontal, 2 px, #6B7280.
- ④ splits into 2 colour-coded branches into ⑤ (green = plausible, orange = violation).
- ⑤ sub-boxes merge into the output strip.
- Bottom horizontal LEGEND band with small swatches: "Data flow" (grey solid arrow),
  "Posteriors q_t(s)" (mint), "Skip connection" (grey dashed), "Plausible ρ≤τ" (green),
  "Violation ρ>τ" (orange), and the 4 state colors S1 / systole / S2 / diastole.
- Flat style, thin lines, generous white space. ALL labels in English.
