PREDICTION_GNN=/home/hongyining/s_link/dualEnc_virtual/ablation/DCGCN/predictions.txt
TARGET_GNN=/home/hongyining/s_link/dualEnc_virtual/ablation/DCGCN/targets.txt

PREDICTION_TRANSFORMER=/home/hongyining/s_link/dualEnc_virtual/ablation/transformer/predictions.txt
TARGET_TRANSFORMER=/home/hongyining/s_link/dualEnc_virtual/ablation/transformer/targets.txt


PREDICTION_DUALENC=/home/hongyining/s_link/dualEnc_virtual/ablation/dualEnc/predictions.txt
TARGET_DUALENC=/home/hongyining/s_link/dualEnc_virtual/ablation/dualEnc/targets.txt

GRAPH_INFO=/home/hongyining/s_link/dualEnc_virtual/AMR2.0/dev.graph.info


# cd meteor-1.5
# echo "evaluating METEOR"
# java -Xmx2G -jar meteor-*.jar $PREDICTION_GNN $TARGET_GNN -l en -norm -lower > /home/hongyining/s_link/dualEnc_virtual/ablation/meteor-gnn.txt
# java -Xmx2G -jar meteor-*.jar $PREDICTION_TRANSFORMER $TARGET_TRANSFORMER -l en -norm -lower > /home/hongyining/s_link/dualEnc_virtual/ablation/meteor-transformer.txt
# java -Xmx2G -jar meteor-*.jar $PREDICTION_DUALENC $TARGET_DUALENC -l en -norm -lower > /home/hongyining/s_link/dualEnc_virtual/ablation/meteor-dualenc.txt

# cd ../

python draw_image.py $PREDICTION_GNN $TARGET_GNN $PREDICTION_TRANSFORMER $TARGET_TRANSFORMER $PREDICTION_DUALENC $TARGET_DUALENC $GRAPH_INFO

