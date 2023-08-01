#!/bin/sh
INPUTFOLDER=/home/hongyining/s_link/dualEnc_virtual/SData
OUTPUTFOLDER=/home/hongyining/s_link/dualEnc_virtual/SDataBPE
PREFOLDER=/home/hongyining/s_link/dualEnc_virtual/SDataBPE_PRE
BINFOLDER=/home/hongyining/s_link/dualEnc_virtual/SDatabin

DATA_PATH=/home/hongyining/s_link/dualEnc_virtual/silver_data
# DATA_PATH=/home/hongyining/s_link/amr_annotation_3.0/data/alignments/split
OUTPUT_PATH=/home/hongyining/s_link/dualEnc_virtual/SData
ONLY_TRAIN=true # only process data in $DATA_PATH/training

cd ../
python preprocess.py \
    --dir-path $DATA_PATH \
    --output-dir-path $OUTPUT_PATH \
    --only-train $ONLY_TRAIN
cd fairseq

if [ "$ONLY_TRAIN" = "true" ]; then
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "$INPUTFOLDER/train.graph.edge" \
        --outputs "$OUTPUTFOLDER/train.graph.edge" \
        --workers 60 \
        --keep-empty;
    
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "$INPUTFOLDER/train.graph.node" \
        --outputs "$OUTPUTFOLDER/train.graph.node" \
        --workers 60 \
        --keep-empty;
    
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUTFOLDER/train.sequence.source" \
    --outputs "$OUTPUTFOLDER/train.sequence.source" \
    --workers 60 \
    --keep-empty;

    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUTFOLDER/train.sequence.target" \
    --outputs "$OUTPUTFOLDER/train.sequence.target" \
    --workers 60 \
    --keep-empty;

    cp $INPUTFOLDER/train.graph.info $OUTPUTFOLDER/train.graph.info 

    cp $OUTPUTFOLDER/train.graph.info $PREFOLDER/train.graph.info 
    cp $OUTPUTFOLDER/train.sequence.source $PREFOLDER/train.sequence.source
    cp $OUTPUTFOLDER/train.sequence.target $PREFOLDER/train.sequence.target

    python preprocess_bpe.py \
        $OUTPUTFOLDER/train.graph.node \
        $PREFOLDER/train.graph.node \
        $PREFOLDER/train.graph.node.info

    python preprocess_bpe.py \
        $OUTPUTFOLDER/train.graph.edge \
        $PREFOLDER/train.graph.edge \
        $PREFOLDER/$SPLIT.graph.edge.info
    
    fairseq-preprocess \
    --source-lang "edge" \
    --trainpref "$PREFOLDER/train.graph" \
    --destdir "$BINFOLDER/" \
    --workers 60 \
    --srcdict dict.txt \
    --only-source;

    fairseq-preprocess \
    --source-lang "node" \
    --trainpref "$PREFOLDER/train.graph" \
    --destdir "$BINFOLDER/" \
    --workers 60 \
    --srcdict dict.txt \
    --only-source;

    fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "$PREFOLDER/train.sequence" \
    --destdir "$BINFOLDER/" \
    --workers 60 \
    --srcdict dict.txt \
    --tgtdict dict.txt;

    fairseq-preprocess \
    --source-lang "info" \
    --trainpref "$PREFOLDER/train.graph" \
    --destdir "$BINFOLDER/" \
    --workers 60 \
    --srcdict identical_dict.txt \
    --only-source;

    fairseq-preprocess \
    --source-lang "node.info" \
    --trainpref "$PREFOLDER/train.graph" \
    --destdir "$BINFOLDER/" \
    --workers 60 \
    --srcdict identical_dict.txt \
    --only-source;


    fairseq-preprocess \
    --source-lang "edge.info" \
    --trainpref "$PREFOLDER/train.graph" \
    --destdir "$BINFOLDER/" \
    --workers 60 \
    --srcdict identical_dict.txt \
    --only-source;

else
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

fi