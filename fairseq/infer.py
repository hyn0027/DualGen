import argparse
import json
import os

def parseArg():
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    return args

def addArg(parser):
    parser.add_argument("--encoder-json", default="encoder.json")
    parser.add_argument("--generate-result", default="/home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR2.0+bart/generate-test.txt")
    parser.add_argument("--data-path", default="/home/hongyining/s_link/dualEnc_virtual/AMR2.0/")
    parser.add_argument("--output-dir", default="/home/hongyining/s_link/dualEnc_virtual/fairseq/infer/AMR2.0+bart")

def load_encoder(args):
    with open(args.encoder_json) as f:
        data = json.load(f)
    encoder_dict = dict()
    for item in data:
        encoder_dict[data[item]] = item
    return encoder_dict

def load_infer_result(args):
    with open(args.generate_result) as f:
        data = f.readlines()
    result = dict()
    for item in data:
        if item[0] == "H":
            item = item.split()
            item[0] = item[0].replace("H-", "")
            result[int(item[0])] = item[2:]
    return result

def load_target(args):
    with open(os.path.join(args.data_path, "test.sequence.target")) as f:
        data = f.readlines()
    return data

def translate(infer_result, encoder_dict):
    res = ""
    for item in infer_result:
        idx = int(item)
        res = res + encoder_dict[idx]
    res = res.replace('\u0120', ' ')
    while res != "" and res[0] == ' ':
        res = res[1:]
    return res

def output(args, results, dst):
    with open(args.output_dir + '/predictions.txt', 'w') as f:
        for item in results:
            f.write(item + '\n')
    with open(args.output_dir + '/targets.txt', 'w') as f:
        for item in dst:
            f.write(item )

def convert(args, encoder_dict, infer_results, targets):
    results = []
    dst = []
    for idx in range(0, len(infer_results) + 10):
        if not idx in infer_results:
            continue
        results.append(translate(infer_results[idx], encoder_dict))
        dst.append(targets[idx])
    
    return results, dst
    with open(os.path.join(args.data_path, "test.graph.node.original")) as f:
        data = ''.join(f.readlines())
    data = data.split('\n\n')

    cnt = 0
    for i in range(len(results)):
        result = results[i]
        nodes = data[i]
        node = nodes.replace('\na ', '\n').split('\n')
        used = {}
        # print(node)
        # input("continue")
        # print(result)
        for item in node:
            if item.lower() != item:
                # print("   ", item)
                new_item = item.replace('_', ' ').replace('-', ' @-@ ')
                if new_item not in used:
                    result = result.replace(new_item.lower(), new_item)
                for word in new_item.split():
                    used[word] = True
        cnt += 1
        idx = 0
        if result[idx] >= 'a' and result[idx] <= 'z':
            result = result[:idx] + result[idx: idx + 1].upper() + result[idx + 1:]
        idx += 1
        while idx < len(result):
            new_idx = idx - 1
            while new_idx > 0 and result[new_idx] == ' ':
                new_idx -= 1
            if result[new_idx] == '.' or  result[new_idx] == '?' or result[new_idx] == '!':
                result = result[:idx] + result[idx: idx + 1].upper() + result[idx + 1:]
            idx += 1
        results[i] = result
        # print(result)
        # if (cnt == 6):
        #     exit(-1)

    return results, dst

def main():
    args = parseArg()
    encoder_dict = load_encoder(args)
    infer_results = load_infer_result(args)
    targets = load_target(args)
    results, dst = convert(args, encoder_dict, infer_results, targets)
    output(args, results, dst)

if __name__ == '__main__':
    main()