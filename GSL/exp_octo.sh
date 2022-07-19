#!/bin/bash

#python3 run_dataset.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml
#sleep 240

export CUDA_VISIBLE_DEVICES=0
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/bk_loss_pn_beats_ecl.yaml --stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/bk_loss_pn_beats_ecl.yaml --stack_num 6 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/bk_loss_pn_beats_ecl.yaml --stack_num 9 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 &
sleep 3


export CUDA_VISIBLE_DEVICES=4
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 6 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 &
sleep 3


export CUDA_VISIBLE_DEVICES=5
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 9 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 &
sleep 3


#export CUDA_VISIBLE_DEVICES=6
#export CUDA_VISIBLE_DEVICES=7

