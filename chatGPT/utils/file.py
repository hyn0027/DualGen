def read_file(path, encoding="utf8"):
    with open(path, 'r', encoding=encoding) as f:
        data = f.readlines()
    return data

def write_file(path, data, encoding="utf8"):
    with open(path, 'w', encoding=encoding) as f:
        data = [item + '\n' for item in data]
        f.writelines(data)