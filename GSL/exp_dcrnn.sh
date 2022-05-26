#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 48 --hidden_dim 64 &
sleep 3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 60 --hidden_dim 64 &
sleep 3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 48 --hidden_dim 128 &
sleep 3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 60 --hidden_dim 128 &
sleep 3



export CUDA_VISIBLE_DEVICES=3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 48 --hidden_dim 32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 60 --hidden_dim 32 &
sleep 3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 48 --hidden_dim 256 &
sleep 3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 60 --hidden_dim 256 &
sleep 3
