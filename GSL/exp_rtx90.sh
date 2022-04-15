#!/bin/sh

#export CUDA_VISIBLE_DEVICES=0
#for i in 1
#do
#    python3 run_exp.py --conf_file_path ./config/0416/attention_Gumbel.yaml &
#    sleep 3
#done

#export CUDA_VISIBLE_DEVICES=3
#for i in 1
#do
#    python3 run_exp.py --conf_file_path ./config/0416/attention_weight.yaml &
#    sleep 3
#done

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp.py --conf_file_path ./config/0416/GTS_weight.yaml &
    sleep 3
done

#export CUDA_VISIBLE_DEVICES=1
#for i in 1
#do
#    python3 run_exp.py --conf_file_path ./config/0416/MTGNN_Gumbel.yaml &
#    sleep 3
#done