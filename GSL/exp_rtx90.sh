#!/bin/sh


export CUDA_VISIBLE_DEVICES=0
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/ETT/bk_loss_pn_beats_ett_gl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/Exchange/bk_loss_pn_beats_exchange_gl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/Traffic/bk_loss_pn_beats_traffic_gl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/Weather/bk_loss_pn_beats_wth_gl.yaml --stack_num 3 --n_pool_kernel_size 8,4,2 --n_stride_size 4,2,1 &
sleep 3
