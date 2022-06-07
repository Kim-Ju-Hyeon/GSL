#!/bin/sh


export CUDA_VISIBLE_DEVICES=3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 64,64,64,64 --thetas_dim 8,16 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 64,64,64 --thetas_dim 8,16 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 64,64 --thetas_dim 8,16 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 64 --thetas_dim 8,16 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 64,64 --thetas_dim 8,16 &
sleep 3
python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --stack_type generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic,generic  --num_blocks_per_stack 9 --n_theta_hidden 64,64 --thetas_dim 8,16 &
sleep 3


#export CUDA_VISIBLE_DEVICES=2
