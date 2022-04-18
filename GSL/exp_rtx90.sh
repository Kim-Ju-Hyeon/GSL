#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/Top_K_masking_zero/attention_Top_k.yaml &
    sleep 3
done

#export CUDA_VISIBLE_DEVICES=3
#for i in 1
#do
#    python3 run_exp.py --conf_file_path ./config/Top_K/GTS_Top_k.yaml &
#    sleep 3
#done

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/Top_K_masking_zero/MTGNN_Top_k.yaml &
    sleep 3
done

#export CUDA_VISIBLE_DEVICES=1
#for i in 1
#do
#    python3 run_exp.py --conf_file_path ./config/Top_K_masking_zero/GTS_Top_k.yaml &
#    sleep 3
#done