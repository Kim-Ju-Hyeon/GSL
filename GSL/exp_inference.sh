#!/bin/sh

#python3 run_dataset.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml
#sleep 240

export CUDA_VISIBLE_DEVICES=0
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_3__singular_stack_num_1__n_pool_kernel_size_\[16\,\ 8\,\ 4\]_0713_165953/config.yaml &
sleep 3
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_15__singular_stack_num_3__n_pool_kernel_size_\[16\,\ 16\,\ 16\,\ 16\,\ 16\,\ 8\,\ 8\,\ 8\,\ 8\,\ 8\,\ 4\,\ 4\,\ 4\,\ 4\,\ 4\]_0713_170218/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_3__singular_stack_num_1__n_pool_kernel_size_\[16\,\ 8\,\ 4\]_0713_170206/config.yaml &
sleep 3
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_18__singular_stack_num_3__n_pool_kernel_size_\[16\,\ 16\,\ 16\,\ 16\,\ 16\,\ 16\,\ 8\,\ 8\,\ 8\,\ 8\,\ 8\,\ 8\,\ 4\,\ 4\,\ 4\,\ 4\,\ 4\,\ 4\]_0713_170221/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_6__singular_stack_num_1__n_pool_kernel_size_\[16\,\ 16\,\ 8\,\ 8\,\ 4\,\ 4\]_0713_165956/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_6__singular_stack_num_1__n_pool_kernel_size_\[16\,\ 16\,\ 8\,\ 8\,\ 4\,\ 4\]_0713_170209/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=4
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_9__singular_stack_num_1__n_pool_kernel_size_\[16\,\ 16\,\ 16\,\ 8\,\ 8\,\ 8\,\ 4\,\ 4\,\ 4\]_0713_165959/config.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 run_inference.py --conf_file_path ../exp/0713_ECL_PN_BEATS/stacks_12__singular_stack_num_1__n_pool_kernel_size_\[16\,\ 16\,\ 16\,\ 16\,\ 8\,\ 8\,\ 8\,\ 8\,\ 4\,\ 4\,\ 4\,\ 4\]_0713_170215/config.yaml &
sleep 3

#export CUDA_VISIBLE_DEVICES=6
#python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/pn_beats_ecl.yaml --stack_num 21 --singular_stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 8,4,2 &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=7
#python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/pn_beats_ecl.yaml --stack_num 24 --singular_stack_num 3 --n_pool_kernel_size 16,8,4 --n_stride_size 8,4,2 &
#sleep 3
