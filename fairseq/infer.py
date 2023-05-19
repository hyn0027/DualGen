import argparse
import json

def parseArg():
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    return args

def addArg(parser):
    parser.add_argument("--encoder-json", default="encoder.json")
    parser.add_argument("--generate-result", default="/home/hongyining/s_link/dualEnc_virtual/fairseq/infer/dualEnc/generate-test.txt")
    parser.add_argument("--target", default="/home/hongyining/s_link/dualEnc_virtual/AMR2.0/test.sequence.target")
    parser.add_argument("--output-dir", default="/home/hongyining/s_link/dualEnc_virtual/fairseq/infer/dualEnc")

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
    with open(args.target) as f:
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

def output(args, encoder_dict, infer_results, targets):
    results = []
    dst = []
    for idx in range(0, len(infer_results) + 10):
        if not idx in infer_results:
            continue
        results.append(translate(infer_results[idx], encoder_dict))
        dst.append(targets[idx])
    with open(args.output_dir + '/predictions.txt', 'w') as f:
        for item in results:
            f.write(item + '\n')
    with open(args.output_dir + '/targets.txt', 'w') as f:
        for item in dst:
            f.write(item )

def main():
    args = parseArg()
    encoder_dict = load_encoder(args)
    infer_results = load_infer_result(args)
    targets = load_target(args)
    output(args, encoder_dict, infer_results, targets)

if __name__ == '__main__':
    main()