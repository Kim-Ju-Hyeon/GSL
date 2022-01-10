#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0110_dgx/exp_1.yaml &&
    sleep 3 &&
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0110_dgx/exp_2.yaml &&
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0110_dgx/exp_3.yaml &&
    sleep 3 &&
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0110_dgx/exp_4.yaml &&
    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0110_dgx/exp_5.yaml &&
    sleep 3 &&
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/0110_dgx/exp_6.yaml &&
    sleep 3
done