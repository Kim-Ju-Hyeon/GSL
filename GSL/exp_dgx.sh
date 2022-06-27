#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 3 --n_block 3 --mlp_stack 64 --inter_correlation_stack_length 1 &
sleep 3
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 1 --n_block 3 --mlp_stack 64 --inter_correlation_stack_length 1 &
sleep 3


#export CUDA_VISIBLE_DEVICES=1
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 9 --n_block 3 --mlp_stack 64,64,64 --inter_correlation_stack_length 1 &
#sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 9 --n_block 3 --mlp_stack 64,64,64 --inter_correlation_stack_length 2 &
#sleep 3