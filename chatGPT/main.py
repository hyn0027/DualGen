from utils.parseArgument import parseArg
from utils.log import getLogger
from request import request

def main():
    args = parseArg()
    if args.task == "request":
        request(args)

if __name__ == "__main__":
    main()