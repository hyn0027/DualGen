import argparse

def parseArg():
    parser = argparse.ArgumentParser()
    addArg(parser=parser)
    args=parser.parse_args()
    return args

def addArg(parser):
    parser.add_argument("--task", choices=["request"], required=True,
                        help="Selected from: [request]")
    parser.add_argument("--verbose", choices=["DEBUG", "INFO", "WARN"], default="INFO",
                        help="Selected from: [debug, info, warn]")
    parser.add_argument("--API-key", required=True, type=str,
                        help="OpenAI api key")
    parser.add_argument("--organization", required=True, type=str,
                        help="OpenAI organization")
    parser.add_argument("--data-path", required=True, type=str,
                        help="data path of input, one input one line")
    parser.add_argument("--target-path", required=True, type=str,
                        help="target path of input, one input one line")
    parser.add_argument("--output-path", required=True, type=str,
                        help="output path")