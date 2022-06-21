#!/bin/sh


export CUDA_VISIBLE_DEVICES=0
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 3 --kernel_size 2 --inter_correlation_stack_length 1 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 9 --kernel_size 2 --inter_correlation_stack_length 1 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 3 --kernel_size 4 --inter_correlation_stack_length 1 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 9 --kernel_size 4 --inter_correlation_stack_length 1 &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 3 --kernel_size 2 --inter_correlation_stack_length 3 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 9 --kernel_size 2 --inter_correlation_stack_length 3 &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 3 --kernel_size 4 --inter_correlation_stack_length 3 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 3 --n_block 9 --kernel_size 4 --inter_correlation_stack_length 3 &
sleep 3



export CUDA_VISIBLE_DEVICES=4
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 3 --kernel_size 2 --inter_correlation_stack_length 1 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 9 --kernel_size 2 --inter_correlation_stack_length 1 &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 3 --kernel_size 4 --inter_correlation_stack_length 1 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 9 --kernel_size 4 --inter_correlation_stack_length 1 &
sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 3 --kernel_size 2 --inter_correlation_stack_length 3 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 9 --kernel_size 2 --inter_correlation_stack_length 3 &
sleep 3

export CUDA_VISIBLE_DEVICES=7
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 3 --kernel_size 4 --inter_correlation_stack_length 3 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --n_stack 9 --n_block 9 --kernel_size 4 --inter_correlation_stack_length 3 &
sleep 3






