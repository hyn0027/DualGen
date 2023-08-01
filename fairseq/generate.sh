#!/bin/sh
MAX_TOKENS=2048
BART_PATH=/home/hongyining/s_link/dualEnc_virtual/fairseq/training/pre+fine3.0/checkpoint20.pt
DATA_BIN=/home/hongyining/s_link/dualEnc_virtual/AMR3.0bin

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_BIN \
    --path $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --arch bartDualEnc_large \
    --task graph_to_seq \
    --beam 5 \
    --post-process \
    --remove-bpe \
    --results-path /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR3.0+pretrain+bart \
    --find-unused-parameters;

python infer.py \
    --generate-result /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR3.0+pretrain+bart/generate-test.txt \
    --target /home/hongyining/s_link/dualEnc_virtual/AMR3.0/test.sequence.target \
    --output-dir /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR3.0+pretrain+bart