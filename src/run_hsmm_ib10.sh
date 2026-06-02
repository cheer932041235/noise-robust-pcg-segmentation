#!/bin/bash
cd /data2/sjx/projects/pcg_seg/src
PY=/home/user/anaconda3/envs/structecg/bin/python
LOG=/data2/sjx/projects/pcg_seg/logs
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=6
for s in 0 1 2; do
  $PY run_hsmm.py --seed $s --noise_snr 10 --noise_lp 500 > $LOG/hsmmIB_s${s}_snr10.log 2>&1 &
done
wait
echo HSMM_IB10_DONE > $LOG/_hsmmib10_done.marker
