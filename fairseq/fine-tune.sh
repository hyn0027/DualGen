#!/bin/sh
TOTAL_NUM_UPDATES=8000  
WARMUP_UPDATES=200      
LR=8e-6
MAX_TOKENS=2048
MAX_EPOCH=65
UPDATE_FREQ=4
LOG_INTERVAL=20
MODEL_PATH=/home/hongyining/s_link/dualEnc_virtual/fairseq/training/SData1/checkpoint_best.pt
DATA_BIN=/home/hongyining/s_link/dualEnc_virtual/AMR2.0bin
SAVE_DIR=training/

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATA_BIN \
    --restore-file $MODEL_PATH \
    --max-tokens $MAX_TOKENS \
    --save-dir $SAVE_DIR \
    --task graph_to_seq \
    --max-epoch $MAX_EPOCH \
    --fp16 \
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
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1}' \
    --find-unused-parameters;
