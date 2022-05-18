#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_theta_hidden [16, 16, 16] --thetas_dim [16, 16] &&
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_theta_hidden [64, 64, 64] --thetas_dim [64, 64] &&
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_trend_season.yaml --n_theta_hidden [128, 128, 128] --thetas_dim [128, 128] &
done