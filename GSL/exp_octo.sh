#!/bin/sh


export CUDA_VISIBLE_DEVICES=0
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_wth.yaml --stack_num 1 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_wth.yaml --stack_num 2 &
sleep 3

#export CUDA_VISIBLE_DEVICES=2
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_exchange.yaml --stack_num 3 --mlp_num 3 &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=3
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_exchange.yaml --stack_num 3 --mlp_num 3 &
#sleep 3

export CUDA_VISIBLE_DEVICES=4
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_wth.yaml --stack_num 3 &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_wth.yaml --stack_num 4 &
sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_wth.yaml --stack_num 5 &
sleep 3

export CUDA_VISIBLE_DEVICES=7
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_wth.yaml --stack_num 10 &
sleep 3