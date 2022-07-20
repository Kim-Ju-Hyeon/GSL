#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 --edge_prob 0.02 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 --edge_prob 0.1 &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 --edge_prob 0.25 &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 --edge_prob 0.5 &
sleep 3


export CUDA_VISIBLE_DEVICES=4
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ECL/pn_beats_ecl.yaml --stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 4,4,2 --edge_prob 0.8 &
sleep 3


