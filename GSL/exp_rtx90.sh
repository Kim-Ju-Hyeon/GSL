#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_gl.yaml --stack_num 950315 &
sleep 3
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_complete_graph.yaml --stack_num 950315 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_gl2.yaml --stack_num 950315 &
sleep 3
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/WTH/ic_pnbeats_wth_no_graph.yaml --stack_num 950315 &
sleep 3
