PREDICTION=/data_new/private/hongyining/dualEnc_virtual/fairseq/infer/graph-no-special-pretrain-3.0/predictions.txt
TARGET=/data_new/private/hongyining/dualEnc_virtual/fairseq/infer/graph-no-special-pretrain-3.0/targets.txt
cd meteor-1.5
echo "evaluating METEOR"
java -Xmx2G -jar meteor-*.jar $PREDICTION $TARGET -l en -norm -lower > /home/hongyining/s_link/dualEnc_virtual/ablation/meteor.txt
cd ../

python eval.py $PREDICTION $TARGET

# graph 2.0
# sacrebleu score (corpus bleu): 22.1477
# chrf++ score: 58.6572
# meteor score: 31.4294


# graph 3.0
# sacrebleu score (corpus bleu): 22.1744
# chrf++ score: 58.2265
# meteor score: 31.5741

# seq 2.0
# sacrebleu score (corpus bleu): 43.8413
# chrf++ score: 72.1543
# meteor score: 42.1056

# seq 3.0
# sacrebleu score (corpus bleu): 45.6915
# chrf++ score: 73.3809
# meteor score: 42.8982

#-------------------------------------

# full model no pretrain
# Successfully read 1560 sentence pairs.
# sacrebleu score (corpus bleu): 30.1356
# chrf++ score: 57.3765
# meteor score: 35.3128

# full model pretrain
# sacrebleu score (corpus bleu): 30.7568
# chrf++ score: 59.6809
# meteor score: 35.7631

# seq
# Successfully read 1560 sentence pairs.
# sacrebleu score (corpus bleu): 26.1119
# chrf++ score: 49.6159
# meteor score: 31.5056

# graph
# sacrebleu score (corpus bleu): 11.1245
# chrf++ score: 37.5569
# meteor score: 23.4814


# -------------------

# full model pretrain
# Successfully read 1004 sentence pairs.
# sacrebleu score (corpus bleu): 16.4737
# chrf++ score: 54.3975
# meteor score: 23.5656

# full model nopretrain
# Successfully read 1004 sentence pairs.
# sacrebleu score (corpus bleu): 15.3531
# chrf++ score: 52.031
# meteor score: 22.8031