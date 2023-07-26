import logging
import random
from tqdm import tqdm
import amrlib
import os
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
logger = logging.getLogger('pretrain data')

def read_file(path, encoding="utf8"):
    logger.info("start loading file {file}".format(file=path))
    with open(path, mode="r", encoding=encoding) as f:
        data = f.readlines()
    for item in data:
        item = item.replace('\n', '')
        if item == "":
            data.remove(item)
    logger.info("successfully loaded file {file}".format(file=path))
    return data

def write_file(path, content, encoding="utf8"):
    logger.info("start writing to file {file}".format(file=path))
    with open(path, mode="w", encoding=encoding) as f:
        f.write("# AMR-GigaWord 200k\n\n")
        for idx, item in enumerate(content):
            item = item.split('\n')
            result = '''# ::id {id} ::amr-annotator ISI-AMR-01 ::preferred
# ::tok {sent}
# ::alignments
{amr}

'''.format(
                id=idx,
                sent=item[0].replace('# ::snt ', ''),
                amr = '\n'.join(item[2:])
            )
            f.write(result)
    logger.info("successfully write to file {file}".format(file=path))

def addArg(parser):
    parser.add_argument("--data-path", required=True, help="data path")
    parser.add_argument("--data-items", required=True, help="how many data", type=int)
    parser.add_argument("--data-result", required=True, help="data result path")
    parser.add_argument("--cluster", default=2000, type=int, help="how many data entries per file")

def main():
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    stog = amrlib.load_stog_model(device="cuda:0", batch_size=32, num_beams=4)
    data = read_file(args.data_path)
    logger.info("{number} entries in total".format(number=len(data)))
    pre_selected_data = random.choices(data, k=args.data_items)
    selected_data = []
    while len(pre_selected_data) > 0:
        selected_data.append(pre_selected_data[:args.cluster])
        pre_selected_data = pre_selected_data[args.cluster:]
    logger.info("selected {number} entries".format(number=args.data_items))
    result = []
    logging.disable(logging.CRITICAL)
    # for item in selected_data:
    wrong_sent = 0
    cnt = 0
    for item in tqdm(selected_data):
        cnt += 1
        try:
            result += stog.parse_sents(item)
            for graph in result:
                if graph == None:
                    result.remove(graph)
                    wrong_sent += 1
            write_file(os.path.join(args.data_result, "result{id}.txt".format(id=cnt)), result)
            result = []
        except:
            result = []
            continue
    logging.disable(logging.NOTSET)
    
    logger.info("finished, {wrong}/{total} wrong".format(
        wrong=wrong_sent,
        total=args.data_items
    ))

if __name__ == "__main__":
    main()