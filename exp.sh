#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 5
do
    python3 run_exp_CCN_Project.py --conf_file_path ./config/GTS/rtx90_1.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 5
do
    python3 run_exp_CCN_Project.py --conf_file_path ./config/GTS/rtx90_2.yaml &
    sleep 3
done