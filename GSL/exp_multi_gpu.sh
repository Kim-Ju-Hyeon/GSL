#!/bin/sh


export CUDA_VISIBLE_DEVICES=1,5,6,7
horovodrun -np 4 python run_multi_gpu_exp.py --conf_file_path ./config/ic_pnbeats_general.yaml
sleep 3