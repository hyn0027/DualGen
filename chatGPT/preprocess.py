import os
import argparse

def break_amr_instance(example):
    example = example.strip().split('\n')
    target = example[1][8:]
    example = '\n'.join(example[3:])
    return {"input": example, "target": target}

def combine_all_files_in_dir(dir):
    amr_list = []
    files = os.listdir(dir)
    file = files[0]
    amr_example = ''
    for file in files:
        print('Begin processing file', file)
        with open(os.path.join(dir, file), 'r', encoding="utf8") as f:
            amr_example = ''
            for line in f.readlines()[1:]:
                if not line.strip():
                    if len(amr_example.replace('\n', '').replace(' ', '')) > 0:
                        amr_list.append(break_amr_instance(amr_example))
                    amr_example = ''
                amr_example += line
            if len(amr_example.replace('\n', '').replace(' ', '')) > 0:
                amr_list.append(break_amr_instance(amr_example))
            f.close()
    return amr_list

def combine_all_data(dir, output):
    amr_list = []
    for item in dir:
        amr_list += combine_all_files_in_dir(item)
    with open(os.path.join(output, "target.txt"), mode='w', encoding="utf8") as output_file:
        for item in amr_list:
            output_file.write(item['target'] + '\n')
    with open(os.path.join(output, "input.txt"), mode='w', encoding="utf8") as output_file:
        for item in amr_list:
            output_file.write(item['input'] + '\n\n')

def addArg(parser):
    parser.add_argument("--dir-path", required=True, help="data path")
    parser.add_argument("--output-path", required=True, help="output data path")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    print(args)
    test_path = [os.path.join(args.dir_path, 'dev')]
    combine_all_data(test_path, args.output_path)
