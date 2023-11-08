PREDICTION=/home/hongyining/s_link/dualEnc_virtual/chatGPT/4.0-15-shot/result.txt
TARGET=/home/hongyining/s_link/dualEnc_virtual/chatGPT/4.0-15-shot/target.txt
cd meteor-1.5
echo "evaluating METEOR"
java -Xmx2G -jar meteor-*.jar $PREDICTION $TARGET -l en -norm -lower > /home/hongyining/s_link/dualEnc_virtual/ablation/meteor.txt
cd ../

python eval.py $PREDICTION $TARGET

