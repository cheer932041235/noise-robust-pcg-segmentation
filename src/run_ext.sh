#!/bin/bash
# Extended: 5-seed hybrid at more SNR points (in-band), for tighter CI + fuller noise curves.
cd /data2/sjx/projects/pcg_seg/src
PY=/home/user/anaconda3/envs/structecg/bin/python
LOG=/data2/sjx/projects/pcg_seg/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
( for s in 0 2 4; do CUDA_VISIBLE_DEVICES=0 $PY run_hybrid.py --seed $s --epochs 30 \
    --test_snrs 15,10,5,0,-5 --noise_domain deep --tag ext --n_jobs 30 > $LOG/ext_s${s}.log 2>&1; done ) &
( for s in 1 3; do CUDA_VISIBLE_DEVICES=1 $PY run_hybrid.py --seed $s --epochs 30 \
    --test_snrs 15,10,5,0,-5 --noise_domain deep --tag ext --n_jobs 30 > $LOG/ext_s${s}.log 2>&1; done ) &
wait
echo EXT_DONE > $LOG/_ext_done.marker
