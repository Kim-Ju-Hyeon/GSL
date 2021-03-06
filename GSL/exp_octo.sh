#!/bin/sh


#export CUDA_VISIBLE_DEVICES=0
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 2 &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=1
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 3 &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=2
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 4 &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=3
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 1 &
#sleep 3

#export CUDA_VISIBLE_DEVICES=4
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 6 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 2 &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=5
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 6 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 3 &
#sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 6 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 4 &
sleep 3

#export CUDA_VISIBLE_DEVICES=7
#python3 run_ic_pnbeats_exp.py --conf_file_path ./config/IC_PN_BEATS/ECL/ic_pnbeats_ecl.yaml --stack_num 9 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 --factor 1 &
#sleep 3