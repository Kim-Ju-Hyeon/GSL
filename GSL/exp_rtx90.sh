#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
python3 grid_search_metr_la_exp.py --conf_file_path ./config/IC_PN_BEATS/METR_LA/ic_pnbeats_metr_la_groud_truth.yaml &
sleep 3
python3 grid_search_metr_la_exp.py --conf_file_path ./config/IC_PN_BEATS/METR_LA/ic_pnbeats_metr_la_no_graph.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 grid_search_metr_la_exp.py --conf_file_path ./config/IC_PN_BEATS/METR_LA/ic_pnbeats_metr_la_random.yaml &
sleep 3
python3 grid_search_metr_la_exp.py --conf_file_path ./config/IC_PN_BEATS/METR_LA/ic_pnbeats_metr_la_random2.yaml &
sleep 3
