import os
import sys
from sacrebleu import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.rcParams['savefig.dpi'] = 300
def read_arguments():
    pred_folder = sys.argv[1]
    tgt_folder = sys.argv[2]
    return pred_folder, tgt_folder

def readfile(pred_folder, tgt_folder):
    with open(file=pred_folder, mode="r") as pred:
        pred_sentences = pred.readlines()
        pred.close()
    with open(file=tgt_folder, mode="r") as tgt:
        target_sentences = tgt.readlines()
        tgt.close()

    assert len(pred_sentences) == len(target_sentences)
    print("Successfully read " + str(len(target_sentences)) + " sentence pairs.")
    return pred_sentences, target_sentences

def eval_bleu(references, predictions):
    sys.stdout = open(os.devnull, 'w')
    score = corpus_bleu(predictions, [references]).score
    sys.stdout = sys.__stdout__
    print("sacrebleu score (corpus bleu): {number}".format(number=round(score, 4)))
    return score

def eval_chrf(references, predictions):
    ref = []
    pred = []
    for i in range(len(references)):
        ref.append(references[i].split())
        pred.append(predictions[i].split())
    score = corpus_chrf(ref, pred) * 100
    print("chrf++ score: {number}".format(number=round(score, 4)))
    return score

def eval_meteor():
    with open("meteor.txt", 'r') as f:
        meteor_results = f.readlines()
    score = float(meteor_results[-1].split()[2]) * 100
    print("meteor score: {number}".format(number=round(score, 4)))
    return score

def main():
    pred_folder, tgt_folder = read_arguments()
    pred_sentences, target_sentences = readfile(pred_folder, tgt_folder)
    eval_bleu(target_sentences, pred_sentences)
    eval_chrf(target_sentences, pred_sentences)
    eval_meteor()


main()
