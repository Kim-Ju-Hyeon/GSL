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

export CUDA_VISIBLE_DEVICES=4
python3 run_inference.py --conf_file_path ../exp/0910_Traffic_Exp/Forecast_96/stacks_1__mlp_stack_32*1__singular_stack_1_0910_151348/config.yaml --inference False &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 run_inference.py --conf_file_path ../exp/0910_Traffic_Exp/Forecast_192/stacks_1__mlp_stack_32*1__singular_stack_1_0910_151352/config.yaml --inference False &
sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 run_inference.py --conf_file_path ../exp/0910_Traffic_Exp/Forecast_336/stacks_1__mlp_stack_32*1__singular_stack_1_0910_151355/config.yaml --inference False &
sleep 3

export CUDA_VISIBLE_DEVICES=7
python3 run_inference.py --conf_file_path ../exp/0910_Traffic_Exp/Forecast_720/stacks_1__mlp_stack_32*1__singular_stack_3_0912_054511/config.yaml --inference False &
sleep 3