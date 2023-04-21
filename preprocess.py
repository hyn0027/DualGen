import os
import re
import json
import copy

def break_amr_instance(example):
    example = example.strip().split('\n')
    example_id = example[0].split()[2]
    target = example[1][8:]
    example = ' '.join(example[3:])
    example = re.sub(r'\s+', ' ', example).strip() 
    example = example.lower()
    nodes = re.findall(r'(\w+\d*)\s/\s(.+?)[\s)]', example)  # extract all nodes
    nodes = dict(zip([node[0] for node in nodes], [re.sub(r'(.+?)-\d\d*', lambda x: x.group(1), node[1]) for node in
                                                   nodes]))  # convert list to dict and remove senses
    bracket_list = re.findall(r'[^\s][(][^)]*[)]', example)
    for item in bracket_list:
        example = example.replace(item, item[0] + '[' + item[2:-1] + ']')
    example = example.replace(')', ' )')
    add_bracket = re.findall(r'[:]\S+\s+[^(]\S*\s', example)
    for item in add_bracket:
        opponents = item.split()
        if opponents[1][0] ==':':
            continue
        example = example.replace(item, opponents[0] + " ( " + opponents[1] + ' ) ')
    for item in nodes:
        str = ""
        for unit in item:
            str += '[' + unit + ']'
        my_list = re.findall(r'[^/][ ]+' + str + r'[~][\S]*[ ]', example)
        my_list += re.findall(r'[^/][ ]+' + str + r'[ ]', example)
        for piece in my_list:
            example = example.replace(piece, piece[0] + ' HEREISWHEREWECHANGE' + nodes[item] + ' ')
    example = example.replace('HEREISWHEREWECHANGE', '')
    linearized_graph = break_basket(example, nodes)
    index = 0
    linearized_graph = re.sub(r'[~][e][.]\S*', '', linearized_graph)
    while linearized_graph[0] == " ":
        linearized_graph = linearized_graph[1:]
    while linearized_graph[-1] == " ":
        linearized_graph = linearized_graph[:-1]
    if linearized_graph[0] == '(' and linearized_graph[-1] == ')':
        cnt = 0
        flag = True
        for i in range(0, len(linearized_graph)):
            if linearized_graph[i] == '(':
                cnt += 1
            if linearized_graph[i] == ')':
                cnt -= 1
            if cnt < 1:
                flag = False
        if True:
            linearized_graph = linearized_graph[1:-1]
        
    while linearized_graph[0] == " ":
        linearized_graph = linearized_graph[1:]
    while linearized_graph[-1] == " ":
        linearized_graph = linearized_graph[:-1]
    linearized_graph = linearized_graph.replace('(', '( ')
    while linearized_graph.find('  ') != -1:
        linearized_graph = linearized_graph.replace('  ', ' ')
    return {"amr": linearized_graph, "sent": target.lower()}

# break basket '( )' layer by layer
def break_basket(example, nodes):
    str = ''
    match_basket = 0
    left_basket_position = -1
    right_basket_position = -1
    try:
        root_node = example[1:example.index(' /')]
        if root_node.find(' ') != -1:
            root_node = root_node[0:root_node.find(' ')]
    except:
        return example[2:-2].replace('"', "")
    example = example[1:-1]
    for (i, letter) in enumerate(example):
        if letter == '(':
            match_basket += 1
            if left_basket_position == -1 or match_basket == 1:
                left_basket_position = i
        elif letter == ')':
            match_basket -= 1
            if match_basket == 0:
                str += extract_relation(example[right_basket_position + 1: left_basket_position], nodes)
                right_basket_position = i
                str += break_basket(example[left_basket_position:right_basket_position + 1], nodes) + ' '
    if str == '':
        str += extract_relation(example, nodes)
    if str == '':
        return nodes[root_node]
    try:
        return '( ' + nodes[root_node] + ' ' + str + ') '
    except:
        return '( ' + root_node  + ' ' + str + ') '

# extract relation like ':arg1'
def extract_relation(example, nodes):
    flag = False
    str = ''
    for item in example.split():
        if flag:
            entity = re.sub(r'\"?(.*?)\"?\)*', lambda x: x.group(1), item) + ' '
            if entity in nodes:
                entity = nodes[entity]
            str += entity
            flag = False
        elif item[0] == ':':
            flag = True
            str += item + ' '
    return str


def break_amr_instance1(example):
    # print(example)
    example = example.strip().split('\n')
    example_id = example[0].split()[2]
    example = ' '.join(example[3:])
    example = re.sub(r'\s+', ' ', example).strip()
    original_example = copy.deepcopy(example)
    nodes = re.findall(r'(\w+\d*)\s/\s(.+?)[\s)]', example)  
    nodes = dict(zip([node[0] for node in nodes], [node[1] for node in nodes]))
    bracket_list = re.findall(r'[^\s][(][^)]*[)]', example)
    for item in bracket_list:
        example = example.replace(item, item[0] + '[' + item[2:-1] + ']')
    node_id = {}
    id_nodes= {}
    tmp = 0
    for item in nodes:
        node_id[item] = tmp
        id_nodes[tmp] = nodes[item]
        tmp += 1
    example = example.replace('(', ' ( ').replace(')', ' ) ')
    while example.find('  ') != -1:
        example = example.replace('  ', ' ')
    for item in nodes:
        example = example.replace(item + ' / ' + nodes[item], ' NODE' + str(node_id[item]) + ' ')
    for item in nodes:
        example = example.replace(' ' + item + ' ', ' NODE' + str(node_id[item]) + ' ')
    while example.find('  ') != -1:
        example = example.replace('  ', ' ')
    example_list = example.split()
    previous = ""
    pprevious = ""
    i = 0
    while i < len(example_list):
        item = example_list[i]
        
        flag1 = item != '(' and item != ')' and item.find("NODE") == -1 and item.find(":") == -1
        flag2 = item.find(":") != -1 and previous.find(":") != -1 and pprevious.find(":") == -1
        if flag1 or flag2:
            j = i
            item1 = example_list[j]
            item = ""
            while j < len(example_list) and item1 != '(' and item1 != ')' and item1[0] != ':':
                item += example_list[j]
                j += 1
                item1 = example_list[j]
            if j == i + 1 and item.find('~e') != -1 and item[:item.find('~e')] in nodes:
                # print("!")
                example_list[i] = 'NODE' + str(node_id[item[:item.find('~e')]])
            else:
                nodes[item] = item
                node_id[item] = tmp
                id_nodes[tmp] = item
                example_list[i] = 'NODE' + str(tmp) + ""
                tmp += 1
                example_list = example_list[:i + 1] + example_list[j:]
        pprevious = previous
        previous = item
        i += 1
    for item in example_list:
        flag1 = item != '(' and item != ')' and item.find("NODE") == -1 and item[0] != ':'
        flag2 = item.find(":") != -1 and previous.find(":") != -1 and pprevious.find(":") == -1
        if flag1 or flag2:
            print("存在非点非边的情况")
            print(example_list)
            exit(-1)
        pprevious = previous
        previous = item
    try:
        root, edge = check_edges(example_list, 1, len(example_list) - 1)
    except:
        print(example_id)
        exit(-1)
    for item in id_nodes:
        u = id_nodes[item]
        if u.find('~e') != -1:
            u = u[:u.find('~e')]
        if u.find('-0') != -1:
            u = u[:u.find('-0')]
        if u.find('-0') != -1:
            print("????", u)
        id_nodes[item] = u
    for item in edge:
        if item[1].find("~e") != -1:
            item[1] = item[1][:item[1].find("~e")]
    return {"node": tmp, "edge": edge, "root": root, "nodeName": id_nodes}

def get_id(node_name):
    return int(node_name[4:])

def check_edges(example_list, left, right):
    if (example_list[left] != '('):
        root = get_id(example_list[left])
        i = left + 1
    elif example_list[left] == '(' and example_list[left + 2] == ')':
        root = get_id(example_list[left + 1])
        i = left + 3
    else:
        print("无法识别树根")
        print(example_list)
        exit(-1)
    edge_list = []
    while i < right:
        if example_list[i][0] != ':':
            print("递归解析错误1")
            print(example_list)
            print(' '.join(example_list))
            print(left, right)
            print(' '.join(example_list[left:right]))
            exit(-1)
        start = i + 1
        if (example_list[start] != '('):
            son, son_edge_list = check_edges(example_list, start, start + 1)
            edge_list += son_edge_list
            edge_list.append([root, example_list[i], son])
            i += 2
        else:
            mark = i
            cnt = 1
            i += 2
            while i < right:
                if example_list[i] == '(':
                    cnt += 1
                elif example_list[i] == ')':
                    cnt -= 1
                if cnt == 0:
                    break
                i += 1
            if cnt != 0:
                print("递归解析错误2")
                print(example_list)
                exit(-1)
            i += 1
            son, son_edge_list = check_edges(example_list, start + 1, i - 1)
            edge_list += son_edge_list
            edge_list.append([root, example_list[mark], son])
    return root, edge_list

def combine_all_files_in_dir(dir):
    amr_list = []
    files = os.listdir(dir)
    file = files[0]
    amr_example = ''
    for file in files:
        print('Begin linearizing file', file)
        with open(os.path.join(dir, file), 'r') as f:
            amr_example = ''
            for line in f.readlines()[1:]:
                if not line.strip():
                    if len(amr_example.replace('\n', '').replace(' ', '')) > 0:
                        amr_list.append(break_amr_instance(amr_example))
                        amr_list[-1]["graph"] = break_amr_instance1(amr_example)
                    amr_example = ''
                amr_example += line
            if len(amr_example.replace('\n', '').replace(' ', '')) > 0:
                amr_list.append(break_amr_instance(amr_example))
                amr_list[-1]["graph"] = break_amr_instance1(amr_example)
            f.close()
    return amr_list

def combine_all_data(dir, output):
    amr_list = []
    for item in dir:
        amr_list += combine_all_files_in_dir(item)
    # print(amr_list[0])
    with open(output + '.sequence.source', mode='w') as output_file:
        for item in amr_list:
            output_file.write(item['amr'] + '\n')
    
    with open(output + '.sequence.target', mode='w') as output_file:
        for item in amr_list:
            output_file.write(item['sent'] + '\n')
 
    with open(output + '.graph.info', mode='w') as output_file:
        for item in amr_list:
            graph = item["graph"]
            out_str = str(graph["node"]) + ' ' + str(graph["root"])
            for edge in graph["edge"]:
                out_str += ' ' + str(edge[0]) + ' ' + str(edge[2])
            output_file.write(out_str + '\n')

    with open(output + '.graph.node', mode='w') as output_file:
        for item in amr_list:
            node = item["graph"]["nodeName"]
            out_str = ""
            for i in range(item["graph"]["node"]):
                out_str += node[i] + '\n'
            output_file.write(out_str + '\n')

    with open(output + '.graph.edge', mode='w') as output_file:
        for item in amr_list:
            edge = item["graph"]["edge"]
            out_str = ""
            for i in edge:
                out_str += i[1] + '\n'
            output_file.write(out_str + '\n')
    return amr_list

def get_edge(amr_list, output_dir):
    edge_set = set()
    output1 = ""
    output2 = ""
    output3 = ""
    for item in amr_list:
        edge = item["graph"]["edge"]
        for i in edge:
            edge_set.add(i[1])
    i = 50257
    for item in edge_set:
        output1 += '"\u0120' + item + '": ' + str(i) + ', '
        output2 += str(i) + ' 0\n'
        output3 += 'Ġ: ' + item[1:] + '\n'
        i += 1
    with open(output_dir, mode='w') as output_file:
        output_file.write(output1)
        output_file.write('\n\n\n')
        output_file.write(output2)
        output_file.write('\n\n\n')
        output_file.write(output3)

if __name__ == '__main__':
    dir_path = ['/home/hongyining/s_link/abstract_meaning_representation_amr_2.0/data/amrs/split', 
                '/home/hongyining/s_link/abstract_meaning_representation_amr_2.0/data/alignments/split']
    output_dir_path = '/home/hongyining/s_link/dualEnc_virtual/AMR2.0/'
    train_path = [os.path.join(dir_path[1], 'training')]
    dev_path = [os.path.join(dir_path[1], 'dev')]
    test_path = [os.path.join(dir_path[1], 'test')]
    train_output_path = os.path.join(output_dir_path, 'train')
    dev_output_path = os.path.join(output_dir_path, 'dev')
    test_output_path = os.path.join(output_dir_path, 'test')
    list1 = combine_all_data(train_path, train_output_path)
    list2 = combine_all_data(dev_path, dev_output_path)
    list3 = combine_all_data(test_path, test_output_path)
    # get_edge(list1 + list2 + list3, os.path.join(output_dir_path, 'edge_label'))
