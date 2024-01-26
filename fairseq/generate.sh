#!/bin/sh
# MAX_TOKENS=2048
# BART_PATH=/home/hongyining/s_link/dualEnc_virtual/fairseq/training/pre+fine2.0/checkpoint_best.pt
# DATA_BIN=/home/hongyining/s_link/dualEnc_virtual/AMR2.0bin_new

# CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_BIN \
#     --path $BART_PATH \
#     --max-tokens $MAX_TOKENS \
#     --arch bartDualEnc_large \
#     --task graph_to_seq \
#     --beam 5 \
#     --post-process \
#     --remove-bpe \
#     --results-path /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR2.0+valid \
#     --find-unused-parameters;

# python infer.py \
#     --generate-result /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR2.0+valid/generate-test.txt \
#     --target /home/hongyining/s_link/dualEnc_virtual/AMR2.0/dev.sequence.target \
#     --output-dir /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR2.0+valid

MAX_TOKENS=2048
BART_PATH=/data_new/private/hongyining/dualEnc_virtual/fairseq/training/graph-no-special-pretrain-3.0/checkpoint_best.pt
DATA_BIN=/home/hongyining/s_link/dualEnc_virtual/AMR3.0bin

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_BIN \
    --path $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --arch bartDualEnc_large \
    --task graph_to_seq \
    --beam 5 \
    --post-process \
    --remove-bpe \
    --results-path /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/graph-no-special-pretrain-3.0 \
    --find-unused-parameters;

python infer.py \
    --generate-result /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/graph-no-special-pretrain-3.0/generate-test.txt \
    --data-path /home/hongyining/s_link/dualEnc_virtual/AMR3.0 \
    --output-dir /home/hongyining/s_link/dualEnc_virtual/fairseq/infer/graph-no-special-pretrain-3.0