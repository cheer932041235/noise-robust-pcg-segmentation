#!/bin/bash
# Real-noise robustness: mix real CirCor acquisition noise (train state-0 segments) into
# clean test recordings at controlled SNR, in-band. Confirms the deep-degrades / hybrid-helps
# conclusion holds under realistic (non-white, structured) noise.
cd /data2/sjx/projects/pcg_seg/src
PY=/home/user/anaconda3/envs/structecg/bin/python
LOG=/data2/sjx/projects/pcg_seg/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
( for s in 0 2; do CUDA_VISIBLE_DEVICES=0 $PY run_hybrid.py --seed $s --epochs 30 --test_snrs 10,5,0 \
    --noise_domain deep --noise_type real --tag real --n_jobs 30 > $LOG/real_s${s}.log 2>&1; done ) &
( for s in 1; do CUDA_VISIBLE_DEVICES=1 $PY run_hybrid.py --seed $s --epochs 30 --test_snrs 10,5,0 \
    --noise_domain deep --noise_type real --tag real --n_jobs 30 > $LOG/real_s${s}.log 2>&1; done ) &
wait
echo REAL_DONE > $LOG/_real_done.marker
