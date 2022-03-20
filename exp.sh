#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/METR-LA/$i.yaml &
    sleep 3
done
