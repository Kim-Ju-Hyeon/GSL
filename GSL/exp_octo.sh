#!/bin/sh


export CUDA_VISIBLE_DEVICES=0
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_Attention.yaml --stack_num 3 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_Attention.yaml --stack_num 6 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_Attention.yaml --stack_num 9 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_MTGNN.yaml --stack_num 3 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_MTGNN.yaml --stack_num 6 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_MTGNN.yaml --stack_num 9 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3

#export CUDA_VISIBLE_DEVICES=2

#export CUDA_VISIBLE_DEVICES=3
#python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid.yaml --stack_num 3 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --edge_prob 0.05 --gl False &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=3
#python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid.yaml --stack_num 3 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --edge_prob 0.25 --gl False &
#sleep 3
#
#export CUDA_VISIBLE_DEVICES=4
#python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid.yaml --stack_num 3 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --edge_prob 0.5 --gl False &
#sleep 3

#export CUDA_VISIBLE_DEVICES=5
#export CUDA_VISIBLE_DEVICES=6

export CUDA_VISIBLE_DEVICES=7
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_GTS.yaml --stack_num 3 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_GTS.yaml --stack_num 6 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3
python3 grid_search_pnbeats_exp.py --conf_file_path ./config/PN_BEATS/COVID19/pn_beats_covid_gl_GTS.yaml --stack_num 9 --n_pool_kernel_size 2,2,1 --n_stride_size 1,1,1 --gl True &
sleep 3