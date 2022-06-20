#!/bin/sh


export CUDA_VISIBLE_DEVICES=2
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --type MPNN &
sleep 3



export CUDA_VISIBLE_DEVICES=3
python3 grid_search_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --type MPGLU &
sleep 3