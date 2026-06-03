# MODULE LIST — 1-D U-Net architecture (detailed, single-network focus)

Mimic the EXACT visual language of the reference image: clean macaron pastel palette, REAL embedded
mini-diagrams, thin connector arrows, circular number badges, and a bottom LEGEND band. Flat
NeurIPS/ICLR style — no gradients, shadows, 3-D or textures. Sans-serif labels, all horizontal.
**ALL text must be in English only.** This figure focuses on ONE network: the 1-D U-Net used for
per-sample PCG segmentation.

## MODULE 1: GLOBAL CANVAS & LAYOUT GRID
- Canvas 2048×1152 px, medium aspect, 300 DPI, white background, 56 px margins.
- Primary flow axis: LEFT → RIGHT, with a symmetric "U" depth axis (encoder descends, decoder ascends).
- Single large rounded macaron panel (very pale blue #F2F7FC) holding the whole network; a slim header
  band on top (fill #2F5E96) with white title "1-D U-Net for PCG segmentation (base 16, depth 4)".
- **IMPORTANT**: All pixel coordinates here are INTERNAL LAYOUT GUIDES only. Do NOT render any
  coordinates, pixel values, or axis numbers as visible text.
- Z-order: panel z=0, slabs z=1, arrows z=2, text z=3.

## MODULE 2: INPUT
- Left edge: a single-channel heart-sound WAVEFORM mini-plot (dark blue #3B6EA5 line, two S1/S2 bursts).
- Label "Noisy PCG  (1 × L)" in bold; italic note "in-band noise".
- Arrow into the first encoder block.

## MODULE 3: ENCODER (down-sampling path)
- Four encoder levels drawn as STACKS of 1-D feature-map slabs, descending like the left arm of a U.
  Each level = a column of thin vertical slabs whose count suggests channel depth and whose height
  shrinks with temporal length:
  - Level 0: "16 ch, L"   (tall, light blue #9EC1E4)
  - Level 1: "32 ch, L/2" (medium #6B9BD1)
  - Level 2: "64 ch, L/4" (#4C78B8)
  - Bottleneck: "128 ch, L/8" (slate #5B4B8A), at the bottom centre, labelled "bottleneck".
- Between levels: a downward arrow labelled "↓ stride-2 max-pool".
- Each level shows a tiny "Conv1d→BN→ReLU ×2" tag.

## MODULE 4: DECODER (up-sampling path)
- Three decoder levels mirroring the encoder, ascending like the right arm of the U:
  - "64 ch, L/4" → "32 ch, L/2" → "16 ch, L" (light teal/blue slabs #7FB0DE).
- Between levels: an upward arrow labelled "↑ transposed-conv (stride 2)".
- Each decoder level receives a horizontal dashed grey SKIP-CONNECTION arrow from the SAME-LEVEL
  encoder, annotated "skip: concat".

## MODULE 5: OUTPUT HEAD
- After the top decoder level: a small box "1×1 Conv → softmax (5 classes)".
- Output mini-plot: a POSTERIORGRAM heatmap (4 rows S1/systole/S2/diastole × time, magma) AND below
  it a colored SEGMENTATION STRIP showing the cycle S1(dark blue)→systole(light blue)→S2(orange)→
  diastole(peach).
- Label "per-sample posteriors q_t(s)" in bold.

## FINAL MODULE: ARROW ROUTING & LEGEND
- Solid grey flow arrows (#6B7280, 2 px) along the encoder-down / bottleneck / decoder-up path.
- Three horizontal dashed grey skip arrows across the top of the U linking matching levels.
- Bottom LEGEND band: "Conv1d→BN→ReLU block", "↓ stride-2 pool", "↑ transposed-conv",
  "skip connection (concat)", "feature map (channels × length)", and the 4 state colors
  S1 / systole / S2 / diastole.
- Flat, thin lines, generous white space. ALL labels in English. Channel counts must read exactly
  16, 32, 64, 128, 64, 32, 16.
