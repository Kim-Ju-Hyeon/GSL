#!/bin/sh


export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode no_graph --num_blocks_per_stack 3 --n_theta_hidden 16,16,16 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode no_graph --num_blocks_per_stack 6 --n_theta_hidden 16,16,16,16,16 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode no_graph --num_blocks_per_stack 9 --n_theta_hidden 64,64 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode no_graph --num_blocks_per_stack 3 --n_theta_hidden 64,64,64,64,64 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode no_graph --num_blocks_per_stack 3 --n_theta_hidden 128,128 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode no_graph --num_blocks_per_stack 3 --n_theta_hidden 16,32,64,128,32 --thetas_dim 16,8 &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode ground_truth --num_blocks_per_stack 3 --n_theta_hidden 16,16,16 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode ground_truth --num_blocks_per_stack 6 --n_theta_hidden 16,16,16,16,16 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode ground_truth --num_blocks_per_stack 9 --n_theta_hidden 64,64 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode ground_truth --num_blocks_per_stack 3 --n_theta_hidden 64,64,64,64,64 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode ground_truth --num_blocks_per_stack 3 --n_theta_hidden 128,128 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode ground_truth --num_blocks_per_stack 3 --n_theta_hidden 16,32,64,128,32 --thetas_dim 16,8 &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode random_graph --num_blocks_per_stack 3 --n_theta_hidden 16,16,16 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode random_graph --num_blocks_per_stack 6 --n_theta_hidden 16,16,16,16,16 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode random_graph --num_blocks_per_stack 9 --n_theta_hidden 64,64 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode random_graph --num_blocks_per_stack 3 --n_theta_hidden 64,64,64,64,64 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode random_graph --num_blocks_per_stack 3 --n_theta_hidden 128,128 --thetas_dim 16,8 &
    sleep 3
    python3 run_exp.py --conf_file_path ./config/N_BEATS/nbeats_generic.yaml --graph_learning_mode random_graph --num_blocks_per_stack 3 --n_theta_hidden 16,32,64,128,32 --thetas_dim 16,8 &
    sleep 3
done