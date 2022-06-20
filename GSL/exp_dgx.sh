#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --type MPGLU --n_stack 3 --n_block 3 &
sleep 30
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --type MPGLU --n_stack 3 --n_block 9 &
sleep 3



export CUDA_VISIBLE_DEVICES=1
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --type MPGLU --n_stack 9 --n_block 3 &
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_n_hits.yaml --type MPGLU --n_stack 9 --n_block 9 &
sleep 3
