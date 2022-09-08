#!/bin/sh


export CUDA_VISIBLE_DEVICES=4
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_96.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_192.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_336.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=7
python3 run_ic_pnbeats_grid_search_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_720.yaml --stack_num 950315 &
sleep 3