INPUTFOLDER=/home/hongyining/s_link/dualEnc_virtual/AMR3.0
OUTPUTFOLDER=/home/hongyining/s_link/dualEnc_virtual/AMR3.0BPE
PREFOLDER=/home/hongyining/s_link/dualEnc_virtual/AMR3.0BPE_PRE
BINFOLDER=/home/hongyining/s_link/dualEnc_virtual/AMR3.0bin

# cd ../
# python preprocess.py
# cd fairseq

for SPLIT in train test dev
do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUTFOLDER/$SPLIT.graph.edge" \
    --outputs "$OUTPUTFOLDER/$SPLIT.graph.edge" \
    --workers 60 \
    --keep-empty;
    
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUTFOLDER/$SPLIT.graph.node" \
    --outputs "$OUTPUTFOLDER/$SPLIT.graph.node" \
    --workers 60 \
    --keep-empty;
 
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUTFOLDER/$SPLIT.sequence.source" \
    --outputs "$OUTPUTFOLDER/$SPLIT.sequence.source" \
    --workers 60 \
    --keep-empty;

    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUTFOLDER/$SPLIT.sequence.target" \
    --outputs "$OUTPUTFOLDER/$SPLIT.sequence.target" \
    --workers 60 \
    --keep-empty;

    cp $INPUTFOLDER/$SPLIT.graph.info $OUTPUTFOLDER/$SPLIT.graph.info 

    cp $OUTPUTFOLDER/$SPLIT.graph.info $PREFOLDER/$SPLIT.graph.info 
    cp $OUTPUTFOLDER/$SPLIT.sequence.source $PREFOLDER/$SPLIT.sequence.source
    cp $OUTPUTFOLDER/$SPLIT.sequence.target $PREFOLDER/$SPLIT.sequence.target

    python preprocess_bpe.py \
        $OUTPUTFOLDER/$SPLIT.graph.node \
        $PREFOLDER/$SPLIT.graph.node \
        $PREFOLDER/$SPLIT.graph.node.info

    python preprocess_bpe.py \
        $OUTPUTFOLDER/$SPLIT.graph.edge \
        $PREFOLDER/$SPLIT.graph.edge \
        $PREFOLDER/$SPLIT.graph.edge.info
    
done



fairseq-preprocess \
--source-lang "edge" \
--trainpref "$PREFOLDER/train.graph" \
--validpref "$PREFOLDER/dev.graph" \
--testpref "$PREFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict dict.txt \
--only-source;

fairseq-preprocess \
--source-lang "node" \
--trainpref "$PREFOLDER/train.graph" \
--validpref "$PREFOLDER/dev.graph" \
--testpref "$PREFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict dict.txt \
--only-source;

fairseq-preprocess \
--source-lang "source" \
--target-lang "target" \
--trainpref "$PREFOLDER/train.sequence" \
--validpref "$PREFOLDER/dev.sequence" \
--testpref "$PREFOLDER/test.sequence" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict dict.txt \
--tgtdict dict.txt;

fairseq-preprocess \
--source-lang "info" \
--trainpref "$PREFOLDER/train.graph" \
--validpref "$PREFOLDER/dev.graph" \
--testpref "$PREFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict identical_dict.txt \
--only-source;

fairseq-preprocess \
--source-lang "node.info" \
--trainpref "$PREFOLDER/train.graph" \
--validpref "$PREFOLDER/dev.graph" \
--testpref "$PREFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict identical_dict.txt \
--only-source;


fairseq-preprocess \
--source-lang "edge.info" \
--trainpref "$PREFOLDER/train.graph" \
--validpref "$PREFOLDER/dev.graph" \
--testpref "$PREFOLDER/test.graph" \
--destdir "$BINFOLDER/" \
--workers 60 \
--srcdict identical_dict.txt \
--only-source;
