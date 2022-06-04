#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --type MPNN
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --type MPGLU
sleep 3


export CUDA_VISIBLE_DEVICES=1
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --type MP_single_message
sleep 3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --type MPGLU_single_message
sleep 3
