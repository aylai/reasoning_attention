#!/bin/bash

gpu=$1
for i in `seq 0 20`; do
    CUDA_VISIBLE_DEVICES=$gpu python3 ./snli_reasoning_attention.py --grid_id $i --model_type word_by_word --model_name gridsearch_att_$i --train --data_dir mpe
done