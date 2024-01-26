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
        edge_detail = item.split()[2:]
        edge_detail = [int(id) for id in edge_detail]
        edge = [(edge_detail[i * 2], edge_detail[i * 2 + 1]) for i in range(int(len(edge_detail) / 2))]
        data.append({
            "node": int(item.split()[0]),
            "edge": int((len(item.split()) - 2) / 2),
            "root": int(item.split()[1]),
            "edge_detail": edge,
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

def fail_cases(pred, tgt, threshold=0.4):
    fail_case = []
    for i in range(len(pred)):
        score = eval_bleu([tgt[i]], [pred[i]])
        if score < threshold:
            fail_case.append(
                {
                    "case_id": i,
                    "score": score,
                    "pred": pred[i],
                    "tgt": tgt[i]
                }
            )
    return fail_case


def draw_histogram(pred, tgt, img_name):
    score = []
    for i in range(len(pred)):
        score.append(eval_bleu([tgt[i]], [pred[i]]))
    # clean plot
    plt.clf()
    plt.hist(score, bins=30)
    plt.xlabel("BLEU Score")
    plt.ylabel("Number of Sentences")
    plt.title("BLEU Score Distribution")
    plt.savefig(img_name)
    plt.show()

def depth_dfs(p, edge, visited, depth):
    visited[p] = True
    max_depth = depth
    for e in edge:
        if e[0] == p and not visited[e[1]]:
            max_depth = max(depth, depth_dfs(e[1], edge, visited, depth + 1))
    visited[p] = False
    return max_depth

def analysis_fail_cases(fail_cases, graph_info):
    edge_num = []
    node_num = []
    edge_div_node = []
    reentrance = []
    depth = []
    for item in fail_cases:
        id = item['case_id']
        graph_info_item = graph_info[id]
        edge_num.append(graph_info_item['edge'])
        node_num.append(graph_info_item['node'])
        edge_div_node.append(graph_info_item['edge'] / graph_info_item['node'])
        reentrance.append(graph_info_item['edge'] - graph_info_item['node'] + 1)
        visited = [False for i in range(graph_info_item['node'])]
        depth.append(depth_dfs(graph_info_item['root'], graph_info_item['edge_detail'], visited, 0))
    df = pd.DataFrame({
        "edge_num": edge_num,
        "node_num": node_num,
        "edge_div_node": edge_div_node,
        "reentrance": reentrance,
        "depth": depth
    })
    print(df.describe())

def main():
    pred_gnn_folder, tgt_gnn_folder, pred_transformer_folder, tgt_transformer_folder, pred_dualenc_folder, tgt_dualenc_folder, graph_info = read_arguments()
    pred_gnn_sentences, tgt_gnn_sentences = readfile(pred_gnn_folder, tgt_gnn_folder)
    pred_transformer_sentences, tgt_transformer_sentences = readfile(pred_transformer_folder, tgt_transformer_folder)
    pred_dualenc_sentences, tgt_dualenc_sentences = readfile(pred_dualenc_folder, tgt_dualenc_folder)
    graph_info = read_graph_info(graph_info)
    
    # plt.figure(figsize=(4, 2.2))
    # plt.rcParams['savefig.dpi'] = 600

    # draw_graph(graph_info, pred_gnn_sentences, tgt_gnn_sentences, "g2s")
    # draw_graph(graph_info, pred_transformer_sentences, tgt_transformer_sentences, "s2s")
    # draw_graph(graph_info, pred_dualenc_sentences, tgt_dualenc_sentences, "DualGen")
    
    # plt.xlabel("Number of Edges / Number of Nodes")
    # plt.ylabel("BLEU Score")
    # plt.title("BLEU Score vs. Graph Complexity")
    # plt.legend()
    # plt.savefig("graph_complexity.png")
    # plt.show()

    draw_histogram(pred_gnn_sentences, tgt_gnn_sentences, "gnn.png")
    draw_histogram(pred_transformer_sentences, tgt_transformer_sentences, "transformer.png")
    draw_histogram(pred_dualenc_sentences, tgt_dualenc_sentences, "dualenc.png")
    gnn_bleu = eval_bleu(tgt_gnn_sentences, pred_gnn_sentences)
    transformer_bleu = eval_bleu(tgt_transformer_sentences, pred_transformer_sentences)
    dualenc_bleu = eval_bleu(tgt_dualenc_sentences, pred_dualenc_sentences)

    gnn_severe_fail_cases = fail_cases(pred_gnn_sentences, tgt_gnn_sentences, 25)
    transformer_severe_fail_cases = fail_cases(pred_transformer_sentences, tgt_transformer_sentences, 25)
    dualenc_severe_fail_cases = fail_cases(pred_dualenc_sentences, tgt_dualenc_sentences, 25)

    print("gnn severe fail cases: {number}".format(number=len(gnn_severe_fail_cases)))
    print("transformer severe fail cases: {number}".format(number=len(transformer_severe_fail_cases)))
    print("dualenc severe fail cases: {number}".format(number=len(dualenc_severe_fail_cases)))

    analysis_fail_cases(gnn_severe_fail_cases, graph_info)
    analysis_fail_cases(transformer_severe_fail_cases, graph_info)
    analysis_fail_cases(dualenc_severe_fail_cases, graph_info)


    return

if __name__ == "__main__":
    main()
