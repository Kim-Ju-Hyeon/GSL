#!/bin/sh


export CUDA_VISIBLE_DEVICES=0
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml - --n_stack 3 --n_block 3 --kernel_size 5 --inter_correlation_stack_length 1 &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=1
#
#export CUDA_VISIBLE_DEVICES=2
#
#export CUDA_VISIBLE_DEVICES=3
#
#export CUDA_VISIBLE_DEVICES=4
#
#export CUDA_VISIBLE_DEVICES=5
#
#export CUDA_VISIBLE_DEVICES=6
#
#export CUDA_VISIBLE_DEVICES=7




