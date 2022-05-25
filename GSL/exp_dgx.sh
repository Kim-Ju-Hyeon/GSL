#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic --num_blocks_per_stack 3 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 18 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 30 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 3 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 18 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 30 --n_theta_hidden 128,128 --thetas_dim 64,32 &
sleep 3