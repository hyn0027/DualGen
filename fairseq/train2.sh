#!/bin/sh
TOTAL_NUM_UPDATES=10000  
WARMUP_UPDATES=200      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
LOG_INTERVAL=20
BART_PATH=/home/hongyining/s_link/dualEnc_virtual/bart.large/model.pt
DATA_BIN=/home/hongyining/s_link/dualEnc_virtual/AMR2.0bin

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train $DATA_BIN \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --save-dir training/test \
    --task graph_to_seq \
    --freeze 1 \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bartDualEnc_large \
    --log-interval $LOG_INTERVAL \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1}' \
    --find-unused-parameters;
