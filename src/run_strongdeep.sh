#!/bin/bash
# Rigor check: does a STRONGER deep baseline (4x width, deeper, longer training) still
# collapse under in-band noise and still benefit from the plausibility-gated hybrid?
# If yes -> the contribution is robust to baseline strength (preempts the obvious reviewer ask).
cd /data2/sjx/projects/pcg_seg/src
PY=/home/user/anaconda3/envs/structecg/bin/python
LOG=/data2/sjx/projects/pcg_seg/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
( for s in 0 1; do CUDA_VISIBLE_DEVICES=0 $PY run_hybrid.py --seed $s --epochs 50 --base 32 --depth 5 \
    --test_snrs 10,5,0 --noise_domain deep --tag b32d5 --n_jobs 40 > $LOG/strong_s${s}.log 2>&1; done ) &
wait
echo STRONGDEEP_DONE > $LOG/_strongdeep_done.marker
