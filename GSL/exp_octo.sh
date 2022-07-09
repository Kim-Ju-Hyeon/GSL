#!/bin/sh

python3 run_dataset.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml
sleep 240

export CUDA_VISIBLE_DEVICES=0
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 3 --n_block 1 --mlp_stack 32,32,32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 3 --n_block 3 --mlp_stack 32,32,32 &
#sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 6 --n_block 1 --mlp_stack 32,32,32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 3 --n_block 1 --mlp_stack 32,32,32 &
#sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 9 --n_block 1 --mlp_stack 32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 3 --n_block 3 --mlp_stack 32 &
#sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 32 --n_block 1 --mlp_stack 32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 3 --n_block 1 --mlp_stack 32 &
#sleep 3


export CUDA_VISIBLE_DEVICES=4
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 64 --n_block 1 --mlp_stack 32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 3 --n_block 3 --mlp_stack 32,16,8 &
#sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 128 --n_block 1 --mlp_stack 32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 3 --n_block 9 --mlp_stack 32,32,32 &
#sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 256 --n_block 1 --mlp_stack 32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 3 --n_block 9 --mlp_stack 32 &
#sleep 3

export CUDA_VISIBLE_DEVICES=7
python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_stack 9 --n_block 1 --mlp_stack 32,32,32 &
sleep 3
#python3 grid_search_nbeats_I_exp.py --conf_file_path ./config/N_BEATS/graph_learning_nbeats_trend_season.yaml --n_stack 9 --n_block 3 --mlp_stack 32,32,32 &
#sleep 3
