import sys

def read_file(path):
    with open(path) as f:
        data = f.readlines()
    if data[1][0:3] == "65 ":
        for i in range(len(data)):
            if data[i][0:3] == "65 ":
                data[i] = data[i][3:]
    else:
        i = 1
        while i < len(data):
            if data[i][0:3] == "64 ":
                data[i] = data[i][3:]
            else:
                i += 1
            i += 1

    return data

def write_file(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines) 
        print("successfully write to " + path)

def main():
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    infoFile = sys.argv[3]
    data = read_file(inFile)
    l = 0
    output = []
    info = []
    while l < len(data):
        out_str = ""
        info_str = ""
        while data[l] != "\n":
            out_str = out_str + data[l].replace('\n', '') + ' '
            info_str = info_str + str(len(data[l].split())) + ' '
            l += 1
        l += 1
        if len(out_str) > 0:
            out_str = out_str[:-1]
        if len(info_str) > 0:
            info_str = info_str[:-1]
        output.append(out_str + '\n')
        info.append(info_str + '\n')
    write_file(outFile, output)
    write_file(infoFile, info)

if __name__ == '__main__':
    main()