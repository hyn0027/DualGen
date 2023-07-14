#!/bin/sh
MAX_TOKENS=2048
BART_PATH=/home/hongyining/s_link/dualEnc_virtual/fairseq/training/diffLR15/checkpoint43.pt
DATA_BIN=/home/hongyining/s_link/dualEnc_virtual/AMR2.0bin

CUDA_VISIBLE_DEVICES=0,1 fairseq-generate $DATA_BIN \
    --path $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --arch bartDualEnc_large \
    --task graph_to_seq \
    --beam 5 \
    --post-process \
    --remove-bpe \
    --results-path /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/final \
    --find-unused-parameters;


python infer.py