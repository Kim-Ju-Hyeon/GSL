#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/Traffic/ic_pnbeats_traffic_96.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/Traffic/ic_pnbeats_traffic_192.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/Traffic/ic_pnbeats_traffic_336.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/Traffic/ic_pnbeats_traffic_720.yaml --stack_num 950315 &
sleep 3