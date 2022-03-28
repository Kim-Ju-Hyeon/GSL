#!/bin/sh

#export CUDA_VISIBLE_DEVICES=0
#for i in 1
#do
#    python3 run_exp_dgx.py --conf_file_path ./config/GTS/METR-LA/$i.yaml &
#    sleep 3
#done

export CUDA_VISIBLE_DEVICES=1
for i in 2
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/METR-LA/$i.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 3
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/METR-LA/$i.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python3 run_exp_dgx.py --conf_file_path ./config/GTS/PEMS-BAY/$i.yaml &
    sleep 3
done