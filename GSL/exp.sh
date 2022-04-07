#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/GTS/config/attention_Gumbel.yaml &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/GTS/config/attention_Top_k.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/GTS/config/attention_weight.yaml &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/GTS/config/GTS_Gumbel.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/GTS/config/GTS_Top_k.yaml &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/GTS/config/GTS_weight.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/GTS/config/MTGNN_Gumbel.yaml &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/GTS/config/MTGNN_Top_k.yaml &
    sleep 3
done