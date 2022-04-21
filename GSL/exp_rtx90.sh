#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/Top_K/attention_Top_k.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/0416/attention_Gumbel.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/Top_K/GTS_Top_k.yaml &
    sleep 3
done