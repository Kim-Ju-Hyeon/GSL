#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0310_dgx_attention/$i.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 2
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0310_dgx_attention/$i.yaml &
    sleep 3
done
