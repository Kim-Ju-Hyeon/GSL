#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/0425/attention_Gumbel.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/0425/GTS_Gumbel.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/0425/MTGNN_Gumbel.yaml &
    sleep 3
done
