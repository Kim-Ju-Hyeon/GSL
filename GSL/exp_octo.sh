#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_96.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_192.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_336.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_720.yaml --stack_num 950315 &
sleep 3