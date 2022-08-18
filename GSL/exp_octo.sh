#!/bin/sh


export CUDA_VISIBLE_DEVICES=0
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 1 --singular_stack 1 --mlp_stack 32 --mlp_num 1 --thetas_dim 32,32 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 1 --singular_stack 1 --mlp_stack 32 --mlp_num 3 --thetas_dim 32,32 &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 1 --singular_stack 3 --mlp_stack 32 --mlp_num 1 --thetas_dim 32,32 &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 1 --singular_stack 3 --mlp_stack 32 --mlp_num 3 --thetas_dim 32,32 &
sleep 3

export CUDA_VISIBLE_DEVICES=4
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 6 --singular_stack 1 --mlp_stack 32 --mlp_num 1 --thetas_dim 32,32 &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 6 --singular_stack 1 --mlp_stack 32 --mlp_num 3 --thetas_dim 32,32 &
sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 6 --singular_stack 3 --mlp_stack 32 --mlp_num 1 --thetas_dim 32,32 &
sleep 3
#
export CUDA_VISIBLE_DEVICES=7
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ettm2.yaml --stack_num 6 --singular_stack 3 --mlp_stack 32 --mlp_num 3 --thetas_dim 32,32 &
sleep 3