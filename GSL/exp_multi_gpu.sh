#!/bin/sh


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
horovodrun -np 8 python run_multi_gpu_exp.py --conf_file_path ./config/IC_PN_BEATS/ic_pnbeats_ecl.yaml
sleep 3