INPUTFOLDER=/home/hongyining/s_link/dualEnc_virtual/AMR2.0
OUTPUTFOLDER=/home/hongyining/s_link/dualEnc_virtual/AMR2.0BPE
BINFOLDER=/home/hongyining/s_link/dualEnc_virtual/AMR2.0bin

# for SPLIT in train test dev
# do
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#     --encoder-json encoder.json \
#     --vocab-bpe vocab.bpe \
#     --inputs "$INPUTFOLDER/$SPLIT.graph.edge" \
#     --outputs "$OUTPUTFOLDER/$SPLIT.graph.edge" \
#     --workers 60 \
#     --keep-empty;
    
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#     --encoder-json encoder.json \
#     --vocab-bpe vocab.bpe \
#     --inputs "$INPUTFOLDER/$SPLIT.graph.node" \
#     --outputs "$OUTPUTFOLDER/$SPLIT.graph.node" \
#     --workers 60 \
#     --keep-empty;
 
#     python -m examples.roberta.multiprocessing_bpe_encoder \
#     --encoder-json encoder.json \
#     --vocab-bpe vocab.bpe \
#     --inputs "$INPUTFOLDER/$SPLIT.sequence.source" \
#     --outputs "$OUTPUTFOLDER/$SPLIT.sequence.source" \
#     --workers 60 \
#     --keep-empty;

#     python -m examples.roberta.multiprocessing_bpe_encoder \
#     --encoder-json encoder.json \
#     --vocab-bpe vocab.bpe \
#     --inputs "$INPUTFOLDER/$SPLIT.sequence.target" \
#     --outputs "$OUTPUTFOLDER/$SPLIT.sequence.target" \
#     --workers 60 \
#     --keep-empty;

#     cp $INPUTFOLDER/$SPLIT.graph.info $OUTPUTFOLDER/$SPLIT.graph.info 
# done

fairseq-preprocess \
--source-lang "edge" \
--trainpref "$OUTPUTFOLDER/train.graph" \
--validpref "$OUTPUTFOLDER/dev.graph" \
--testpref "$OUTPUTFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict dict.txt \
--only-source;

fairseq-preprocess \
--source-lang "node" \
--trainpref "$OUTPUTFOLDER/train.graph" \
--validpref "$OUTPUTFOLDER/dev.graph" \
--testpref "$OUTPUTFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict dict.txt \
--only-source;

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--trainpref "$OUTPUTFOLDER/train.sequence" \
--validpref "$OUTPUTFOLDER/dev.sequence" \
--testpref "$OUTPUTFOLDER/test.sequence" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict dict.txt \
--tgtdict dict.txt;

fairseq-preprocess \
--source-lang "info" \
--trainpref "$OUTPUTFOLDER/train.graph" \
--validpref "$OUTPUTFOLDER/dev.graph" \
--testpref "$OUTPUTFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict identical_dict.txt \
--only-source;
