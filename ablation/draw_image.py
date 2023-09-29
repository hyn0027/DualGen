import os
import sys
from sacrebleu import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# import statsmodels.api as sm
# import statsmodels.formula.api as smf
def read_arguments():
    pred_gnn_folder = sys.argv[1]
    tgt_gnn_folder = sys.argv[2]
    pred_transformer_folder = sys.argv[3]
    tgt_transformer_folder = sys.argv[4]
    pred_dualenc_folder = sys.argv[5]
    tgt_dualenc_folder = sys.argv[6]
    graph_info = sys.argv[7]
    return pred_gnn_folder, tgt_gnn_folder, pred_transformer_folder, tgt_transformer_folder, pred_dualenc_folder, tgt_dualenc_folder, graph_info

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
    # print("sacrebleu score (corpus bleu): {number}".format(number=round(score, 4)))
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

def read_graph_info(path):
    with open(path, mode='r') as f:
        graph_info = f.readlines()
    data = []
    for item in graph_info:
        data.append({
            "node": int(item.split()[0]),
            "edge": int((len(item.split()) - 2) / 2)
        })
    return data

def draw_graph(graph_info, prediction, target, name):
    print(name)
    node_cnt = []
    y = []
    for graph, pred, tgt in zip(graph_info, prediction, target):
        node_cnt.append(graph['edge'] / graph['node'])
        y.append(eval_bleu([tgt], [pred]))
    # plt.scatter(node_cnt, y, marker='o', s=2)
    coefficients = np.polyfit(node_cnt, y, 1)
    slope, intercept = coefficients
    print("slope: {number}".format(number=round(slope, 4)))
    print("intercept: {number}".format(number=round(intercept, 4)))
    r2 = r2_score(y, node_cnt)
    print("r2: {number}".format(number=round(r2, 4)))
    regression_line = np.polyval(coefficients, node_cnt)
    
    plt.plot(node_cnt, regression_line, label="{}".format(name))

    print('-----------------------')
    
def draw_bar_graph(graph_info, prediction, target, name):
    section0 = {
        "tgt": [],
        "pred": []
    }
    section1 = {
        "tgt": [],
        "pred": []
    }
    section2 = {
        "tgt": [],
        "pred": []
    }
    section3 = {
        "tgt": [],
        "pred": []
    }
    section4 = {
        "tgt": [],
        "pred": []
    }
    # section5 = {
    #     "tgt": [],
    #     "pred": []
    # }
    base = 1.02
    d = 0.06
    for graph, pred, tgt in zip(graph_info, prediction, target):
        if graph['edge'] / graph['node'] < 0.9:
            section0['tgt'].append(tgt)
            section0['pred'].append(pred)
        elif graph['edge']  / graph['node'] < 0.95:
            section1['tgt'].append(tgt)
            section1['pred'].append(pred)
        elif graph['edge'] / graph['node'] <= 1.000001:
            section2['tgt'].append(tgt)
            section2['pred'].append(pred)
        elif graph['edge'] / graph['node'] <= 1.05:
            section3['tgt'].append(tgt)
            section3['pred'].append(pred)
        else:
            section4['tgt'].append(tgt)
            section4['pred'].append(pred)
        # else:
        #     section5['tgt'].append(tgt)
        #     section5['pred'].append(pred)
    print(len(section0["pred"]), len(section1["pred"]), len(section2["pred"]), len(section3["pred"]), len(section4["pred"]))
    section0_bleu = eval_bleu(section0['tgt'], section0['pred'])
    section1_bleu = eval_bleu(section1['tgt'], section1['pred'])
    section2_bleu = eval_bleu(section2['tgt'], section2['pred'])
    section3_bleu = eval_bleu(section3['tgt'], section3['pred'])
    section4_bleu = eval_bleu(section4['tgt'], section4['pred'])
    # section5_bleu = eval_bleu(section5['tgt'], section5['pred'])
    print(section0_bleu, section1_bleu, section2_bleu, section3_bleu, section4_bleu)
    return [section0_bleu, section1_bleu, section2_bleu, section3_bleu, section4_bleu]

def main():
    pred_gnn_folder, tgt_gnn_folder, pred_transformer_folder, tgt_transformer_folder, pred_dualenc_folder, tgt_dualenc_folder, graph_info = read_arguments()
    pred_gnn_sentences, tgt_gnn_sentences = readfile(pred_gnn_folder, tgt_gnn_folder)
    pred_transformer_sentences, tgt_transformer_sentences = readfile(pred_transformer_folder, tgt_transformer_folder)
    pred_dualenc_sentences, tgt_dualenc_sentences = readfile(pred_dualenc_folder, tgt_dualenc_folder)
    # eval_bleu(target_sentences, pred_sentences)
    # eval_chrf(target_sentences, pred_sentences)
    # eval_meteor()
    graph_info = read_graph_info(graph_info)
    plt.figure(figsize=(4, 2.2))
    plt.rcParams['savefig.dpi'] = 600

    draw_graph(graph_info, pred_gnn_sentences, tgt_gnn_sentences, "g2s")
    draw_graph(graph_info, pred_transformer_sentences, tgt_transformer_sentences, "s2s")
    draw_graph(graph_info, pred_dualenc_sentences, tgt_dualenc_sentences, "DualGen")
    
    plt.xlabel("Number of Edges / Number of Nodes")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score vs. Graph Complexity")
    plt.legend()
    plt.savefig("graph_complexity.png")
    plt.show()
    
    return

    plt.figure(figsize=(6, 6))
    plt.rcParams['savefig.dpi'] = 300
    
    gnn = draw_bar_graph(graph_info, pred_gnn_sentences, tgt_gnn_sentences, "gnn")
    transformer = draw_bar_graph(graph_info, pred_transformer_sentences, tgt_transformer_sentences, "transformer")
    dualenc = draw_bar_graph(graph_info, pred_dualenc_sentences, tgt_dualenc_sentences, "dualenc")

    df = pd.DataFrame(
        [
            ['section0', gnn[0], transformer[0], dualenc[0], (dualenc[0] - transformer[0]) ],
            ['section1', gnn[1], transformer[1], dualenc[1], (dualenc[1] - transformer[1]) ], 
            ['section2', gnn[2], transformer[2], dualenc[2], (dualenc[2] - transformer[2]) ], 
            ['section3', gnn[3], transformer[3], dualenc[3], (dualenc[3] - transformer[3]) ], 
            ['section4', gnn[4], transformer[4], dualenc[4], (dualenc[4] - transformer[4]) ],
        ], 
        columns=['Model', 'GNN', 'Transformer', 'DualEnc', "Minus"]
    )

    df.plot(
        x='Model', 
        y=['GNN', 'Transformer', 'DualEnc', "Minus"], 
        kind='bar',
        title="BLEU Score vs. Graph Complexity",
        stacked=False,
        figsize=(6, 6),
    )
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig("bar.png")
    plt.show()

main()
