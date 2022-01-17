#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1 2
do
    python3 run_exp_local.py --conf_file_path ./config/GTS/0113_dgx/$i.yaml &&
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 3 4
do
    python3 run_exp_local.py --conf_file_path ./config/GTS/0113_dgx/$i.yaml &&
    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 5 6
do
    python3 run_exp_local.py --conf_file_path ./config/GTS/0113_dgx/$i.yaml &&
    sleep 3
done