#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 12 --hidden_dim 64 &
sleep 3


export CUDA_VISIBLE_DEVICES=3
python3 run_exp.py --conf_file_path ./config/DCRNN/dcrnn.yaml --input_length 12 --hidden_dim 64 &
sleep 3
