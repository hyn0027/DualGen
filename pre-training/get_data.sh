#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python pretrain_data.py \
    --data-path /home/hongyining/s_link/dualEnc_virtual/gigaword/ggw_data/org_data/train.src.txt \
    --data-items 210000 \
    --data-result /home/hongyining/s_link/dualEnc_virtual/silver_data/training \
    --cluster 2000