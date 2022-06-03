#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic --num_blocks_per_stack 1 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 2 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 3 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 1 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 2 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3

export CUDA_VISIBLE_DEVICES=1

python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic ,generic,generic,generic --num_blocks_per_stack 3 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 1 --n_theta_hidden 128,128,128--thetas_dim 64,32 &
sleep 3
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 2 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3
python3 run_nbeats_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 3 --n_theta_hidden 128,128,128 --thetas_dim 64,32 &
sleep 3