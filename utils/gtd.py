import numpy as np
import os
import pdb
import random
import sys
import pickle as pkl
# from utils import load_dict

def relation2tree(childs, relations, worddicts_r, reworddicts_r):
    gtd = [[] for c in childs]
    start_relation = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    relation_stack = [start_relation]
    parent_stack = [(106, 0)]
    p_re = 0
    p_y = 106   # modify: 103->111
    p_id = 0

    for ci, c in enumerate(childs):
        gtd[ci].append(worddicts_r[c])
        gtd[ci].append(ci+1)
        
        find_flag = 0
        while relation_stack != []:
            if relation_stack[-1][:8].sum() > 0:
                for iii in range(9):
                    if relation_stack[-1][iii] != 0:
                        p_re = iii
                        p_y, p_id = parent_stack[-1]
                        relation_stack[-1][iii] = 0
                        if relation_stack[-1][:8].sum() == 0:
                            relation_stack.pop()
                            parent_stack.pop()
                        find_flag = 1
                        break
            else:
                relation_stack.pop()
                parent_stack.pop()

            if find_flag:
                break
        
        if not find_flag:
            p_y = childs[ci-1]
            p_id = ci
            p_re = 8
        gtd[ci].append(worddicts_r[p_y])
        gtd[ci].append(p_id)
        gtd[ci].append(reworddicts_r[p_re])

        relation_stack.append(relations[ci])
        parent_stack.append((c, ci+1))

    return gtd


def gen_gtd_align(gtd):
    wordNum = len(gtd)
    align = np.zeros([wordNum, wordNum], dtype='int8')
    wordindex = -1

    for i in range(len(gtd)):
        wordindex += 1
        parts = gtd[i]
        if len(parts) == 5:
            realign = parts[3]
            # import pdb;pdb.set_trace()
            realign_index = int(str(realign))
            align[realign_index, wordindex] = 1
    return align

def gen_gtd_relation_align(gtd,dict):
    wordNum = len(gtd)
    align = np.zeros([wordNum, 9], dtype='int8')
    wordindex = -1

    for i in range(len(gtd)):
        wordindex += 1
        parts = gtd[i]
        
        if len(parts) == 5:
            relation = dict[parts[-1]]
            realign = parts[3]
            # import pdb;pdb.set_trace()
            realign_index = int(str(realign))
            align[realign_index, relation] = 1
    return align

class Vocab(object):
    def __init__(self, vocfile):
        self._word2id = {}
        self._id2word = []
        with open(vocfile, 'r') as f:
            index = 0
            for line in f:
                parts = line.split()
                id = 0
                if len(parts) == 2:
                    id = int(parts[1])
                elif len(parts) == 1:
                    id = index
                    index += 1
                else:
                    print('illegal voc line %s' % line)
                    continue
                self._word2id[parts[0]] = id
                self._id2word.append(parts[0])

    def get_voc_size(self):
        return len(self._id2word)

    def get_id(self, w):
        if not w in self._word2id:
            return self._word2id['<unk>']
        return self._word2id[w]

    def get_word(self, wid):
        if wid < 0 or wid >= len(self._id2word):
            return '<unk>'
        return self._id2word[wid]

    def get_eos(self):
        return self._word2id['</s>']

    def get_sos(self):
        return self._word2id['<s>']


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            if not isparent:
                child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3], True])
            else:
                child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3], False])
            isparent = True
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['Above', 'Below']:
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['Sup']:
                    return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
                elif child_list[i][2] not in ['Right', 'Above', 'Below', 'Sup']:
                    return_string += ['illegal']
        elif gtd_list[nodeid][0] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                     '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix']:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):

                if child_list[i][2] in ['Rstart']:
                    if child_list[i][3]:
                        return_string += convert(child_list[i][1], gtd_list)
                    else:
                        return_string += ['\\\\'] + convert(child_list[i][1], gtd_list)
                elif child_list[i][2] == 'Right':
                    if gtd_list[nodeid][0] in ['\\begincases']:
                        return_string += ['\\end' + gtd_list[nodeid][0][6:]] + convert(child_list[i][1], gtd_list)
                    else:
                        return_string += convert(child_list[i][1], gtd_list)
                        # elif child_list[i][2] not in ['Right', 'Above', 'Below']:
                        #     return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Leftsup':
                    return_string += ['\\['] + convert(child_list[i][1], gtd_list) + ['\\]']
                elif child_list[i][2] in ['Inside', 'boxed', 'textcircled']:
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['Sub', 'Below']:
                    if gtd_list[nodeid][0] in ['\\underline', '\\xrightarrow', '\\underrightarrow', '\\underbrace'] and \
                                    child_list[i][2] in ['Below']:
                        return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                    else:
                        return_string += ['_', '{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['Sup', 'Above']:
                    if gtd_list[nodeid][0] in ['\\overline', '\\widehat', '\\hat', '\\widetilde',
                                               '\\dot', '\\oversetfrown', '\\overrightarrow'] and child_list[i][2] in [
                        'Above']:
                        return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
                    else:
                        return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
                elif child_list[i][2] in ['\\Nextline']:
                    return_string += ['\\\\'] + convert(child_list[i][1], gtd_list)
                elif child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)

        return return_string


def gtd2latex(cap):
    try:
        gtd_list = []
        gtd_list.append(['<s>', 0, -1, 'root'])
        for i in range(len(cap)):
            parts = cap[i]
            sym = parts[0]
            childid = int(parts[1])
            parentid = int(parts[3])
            relation = parts[4]
            gtd_list.append([sym, childid, parentid, relation])
        bool_endcases = False

        idx = -1
        for i in range(len(gtd_list)):
            if gtd_list[i][0] == '\\begincases':
                idx = i
        if idx != -1:
            bool_endcases = True
            for i in range(idx + 1, len(gtd_list)):
                if gtd_list[i][2] == idx and gtd_list[i][3] == 'Right':
                    bool_endcases = False

        latex_list = convert(1, gtd_list)
        if bool_endcases:
            latex_list += ['\\endcases']

        latex_list = np.array(latex_list)
        latex_list[latex_list == '\\space'] = '&'
        latex_list = list(latex_list)
        if 'illegal' in latex_list:
            latex_string = 'error3*3'
        else:
            latex_string = ' '.join(latex_list)
        return latex_string
    except:
        return ('error3*3')
        pass


def latex2gtd(cap):
    # cap = parts[1:]
    try:
        gtd_label = []
        # cap = '\\begincases a \\\\ b \\\\ c \\endcases = 1'
        # cap = '\\begincases  a + b  \\\\ c + d \\\\  e + f \\endcases = 1'
        cap = cap.split()
        gtd_stack = []
        idx = 0
        outidx = 1
        error_flag = False
        while idx < len(cap):
            # if idx == 16:
            #     print cap[idx]
            #     pdb.set_trace()
            if idx == 0:
                if cap[0] in ['{', '}']:
                    return ('error2*2: {} should NOT appears at START')

                if cap[0] not in ['\\beginaligned']:
                    string = cap[0] + '\t' + str(outidx) + '\t<s>\t0\tStart'
                    gtd_label.append(string.split('\t'))
                    outidx += 1
                else:
                    gtd_stack.append([cap[idx], str(outidx), 'Align', True])
                    idx += 1
                    string = cap[idx] + '\t' + str(outidx) + '\t<s>\t0\tStart'
                    gtd_label.append(string.split('\t'))
                    outidx += 1

                idx += 1

            else:
                # print(cap[idx])
                # pdb.set_trace()
                if cap[idx] == '{':
                    if cap[idx - 1] == '{':
                        return ('error2*2: double { appears')

                    elif cap[idx - 1] == '}' and gtd_stack:
                        if gtd_stack[-1][0] != '\\frac':
                            return ('error2*2: } { not follows frac ...')
                        else:
                            gtd_stack[-1][2] = 'Below'
                            idx += 1
                    else:
                        if cap[idx - 1] in ['\\frac', '\\overline', '\\widehat', '\\hat',
                                            '\\widetilde', '\\dot', '\\oversetfrown', '\\overrightarrow']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Above', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\sqrt']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Inside', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\underline', '\\xrightarrow', '\\underrightarrow', '\\underbrace']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Below', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\boxed', '\\textcircled']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Inside', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\bcancel']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Insert', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\begincases']:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Rstart', True])
                            idx += 1
                        elif cap[idx - 1] in ['\\\\']:
                            idx += 1
                        elif cap[idx - 1] == '_':
                            if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                return ('error2*2: ^ _ follows wrong math symbols')
                            # elif cap[idx - 2] in ['\\sum', '\\int', '\\lim', '\\bigcup', '\\bigcap']:
                            elif cap[idx - 2] in ['\\lim', '\\bigcup', '\\bigcap']:
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Below', True])
                                idx += 1
                            # elif gtd_stack and gtd_stack[-1][0] in ['\\sum', '\\int', '\\lim']:
                            elif gtd_stack and gtd_stack[-1][0] in ['\\lim']:
                                if gtd_stack[-1][2] != 'Below' and gtd_stack[-1][3]:
                                    gtd_stack[-1][2] = 'Below'
                                    gtd_stack[-1][3] = False
                                else:
                                    gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sub', True])
                                idx += 1
                            elif cap[idx - 2] == '}' and gtd_stack:
                                gtd_stack[-1][2] = 'Sub'
                                idx += 1
                            else:
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sub', True])
                                idx += 1
                        elif cap[idx - 1] == '^':

                            if cap[idx - 2] in ['_', '^', '\\frac', '\\sqrt']:
                                return ('error2*2: ^ _ follows wrong math symbols')
                            # elif cap[idx - 2] in ['\\sum', '\\int', '\\lim']:  # 只能先尝试把int删掉了
                            elif cap[idx - 2] in ['\\lim']:  # 只能先尝试把int删掉了
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Above', True])
                                idx += 1
                            # elif gtd_stack and gtd_stack[-1][0] in ['\\sum', '\\int', '\\lim'] and cap[idx - 2] == '}':
                            elif gtd_stack and gtd_stack[-1][0] in ['\\lim'] and cap[idx - 2] == '}':
                                if gtd_stack[-1][2] != 'Above' and gtd_stack[-1][3]:
                                    gtd_stack[-1][2] = 'Above'
                                    gtd_stack[-1][3] = False
                                else:
                                    gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sup', True])
                                idx += 1
                            elif cap[idx - 2] == '}' and gtd_stack:
                                gtd_stack[-1][2] = 'Sup'
                                idx += 1
                            else:
                                gtd_stack.append([cap[idx - 2], str(outidx - 1), 'Sup', True])
                                idx += 1
                        elif cap[idx - 1] == ']':
                            if gtd_stack and gtd_stack[-1][0] == '\\sqrt' and gtd_stack[-1][3]:
                                gtd_stack[-1][2] = 'Inside'
                                idx += 1
                                gtd_stack[-1][3] = False
                            else:
                                return ('error2*2: { follows unknown math symbols ...')
                        else:
                            return ('error2*2: { follows unknown math symbols ...')

                elif cap[idx] == '}':
                    if cap[idx - 1] in ['}', '\\endcases', '\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                        '\\endvmatrix', '\\endVmatrix'] and gtd_stack:
                        del (gtd_stack[-1])
                    idx += 1
                elif cap[idx] in ['\\endcases', '\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                  '\\endvmatrix', '\\endVmatrix']:
                    if cap[idx - 1] in ['\\endcases'] and gtd_stack:
                        del (gtd_stack[-1])
                    elif cap[idx - 1] in ['\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix', '\\endvmatrix',
                                          '\\endVmatrix'] and gtd_stack:
                        string = cap[idx - 1] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        del (gtd_stack[-1])
                        outidx += 1

                    if idx == len(cap) - 1 and cap[idx] not in ['\\endcases'] and gtd_stack:
                        string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        del (gtd_stack[-1])
                        outidx += 1
                    idx += 1
                elif cap[idx] in ['\\\\']:
                    if cap[idx-1] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                        '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix']:
                        return ('error2*2')
                    idx += 1
                elif cap[idx] in [']'] and gtd_stack and cap[idx - 1] in ['}'] and len(gtd_stack) > 1 and gtd_stack[-2][
                    0] == '\\sqrt' and gtd_stack[-2][2] == 'Leftsup':
                    del (gtd_stack[-1])
                    idx += 1
                elif cap[idx] in ['_', '^']:
                    if idx == len(cap) - 1:
                        return ('error2*2: ^ _ appers at end ...')
                    if cap[idx + 1] != '{':
                        return ('error2*2: ^ _ not follows { ...')
                    else:
                        idx += 1
                elif cap[idx] in ['\limits']:
                    return ('error2*2: \limits happens')

                elif cap[idx] == '[' and cap[idx - 1] != '\\\\' and cap[idx -1] not in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                        '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix'] and not (cap[idx - 1] in ['{'] and gtd_stack):
                    if cap[idx - 1] == '\\sqrt':
                        if cap[idx + 1] != ']':
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Leftsup', True])
                        else:
                            gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Leftsup', True])
                            idx += 1
                    elif cap[idx - 1] == '}' and gtd_stack:
                        string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        del (gtd_stack[-1])
                    else:
                        parts = string.split('\t')
                        string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                    idx += 1

                # elif idx == len(cap) - 1 and cap[idx] == ']' and not gtd_stack:
                #
                else:
                    if cap[idx - 1] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix', '\\beginbmatrix',
                                        '\\beginBmatrix', '\\beginvmatrix', '\\beginVmatrix']:

                        gtd_stack.append([cap[idx - 1], str(outidx - 1), 'Rstart', True])

                    if cap[idx - 1] == '{' or (
                                            cap[idx - 1] == '[' and gtd_stack and gtd_stack[-1][0] == '\\sqrt' and
                                    gtd_stack[-1][
                                        2] == 'Leftsup') or cap[idx - 1] in ['\\begincases', '\\\\', '\\beginmatrix',
                                                                             '\\beginpmatrix',
                                                                             '\\beginbmatrix', '\\beginBmatrix',
                                                                             '\\beginvmatrix',
                                                                             '\\beginVmatrix']:
                        if cap[idx - 1] == '\\\\' and cap[idx - 2] == '}' and gtd_stack:
                            del (gtd_stack[-1])
                        if cap[idx - 1] == '\\\\' and gtd_stack == []:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + cap[idx - 2] + '\t' + \
                                     str(outidx - 1) + '\t\\Nextline'
                        elif cap[idx - 1] == '\\\\' and (
                                    gtd_stack and gtd_stack[-1][0] not in ['\\begincases', '\\beginmatrix',
                                                                           '\\beginpmatrix',
                                                                           '\\beginbmatrix', '\\beginBmatrix',
                                                                           '\\beginvmatrix',
                                                                           '\\beginVmatrix']):
                            string = cap[idx] + '\t' + str(outidx) + '\t' + cap[idx - 2] + '\t' + \
                                     str(outidx - 1) + '\t\\Nextline'
                        elif gtd_stack:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + \
                                     gtd_stack[-1][1] + '\t' + gtd_stack[-1][2]

                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
                    elif cap[idx - 1] == '}' and gtd_stack:
                        if cap[idx] not in ['\\endcases', '\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                            '\\endvmatrix', '\\endVmatrix']:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                                1] + '\tRight'
                            gtd_label.append(string.split('\t'))
                            outidx += 1
                            del (gtd_stack[-1])
                        idx += 1
                    elif cap[idx] == ']' and (gtd_stack and gtd_stack[-1][0] == '\\sqrt' and gtd_stack[-1][2] == 'Leftsup'):
                        idx += 1
                    elif cap[idx - 1] in ['\\endcases']:
                        while gtd_stack and gtd_stack[-1][0] != '\\begincases':
                            del (gtd_stack[-1])
                        string = cap[idx] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
                        del (gtd_stack[-1])
                    elif gtd_stack and cap[idx - 1] in ['\\endmatrix', '\\endpmatrix', '\\endbmatrix', '\\endBmatrix',
                                                        '\\endvmatrix', '\\endVmatrix']:
                        string = cap[idx - 1] + '\t' + str(outidx) + '\t' + gtd_stack[-1][0] + '\t' + gtd_stack[-1][
                            1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1

                        parts = string.split('\t')
                        string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + \
                                 parts[1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
                        del (gtd_stack[-1])
                    else:
                        parts = string.split('\t')
                        if cap[idx] == '&' and (
                                    gtd_stack and gtd_stack[-1][0] in ['\\begincases', '\\beginmatrix', '\\beginpmatrix',
                                                                       '\\beginbmatrix', '\\beginBmatrix', '\\beginvmatrix',
                                                                       '\\beginVmatrix']):
                            string = '\space' + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[1] + '\tRight'
                        else:
                            string = cap[idx] + '\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[
                                1] + '\tRight'
                        gtd_label.append(string.split('\t'))
                        outidx += 1
                        idx += 1
        parts = string.split('\t')
        string = '</s>\t' + str(outidx) + '\t' + parts[0] + '\t' + parts[1] + '\tEnd'
        gtd_label.append(string.split('\t'))
        return gtd_label
    except:
        return ('error2*2')
        pass

def save_gtd_label():

    bfs1_path = 'CROHME/'
    gtd_root_path = ''
    gtd_paths = ['train','14_test']

    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_label_gtd.pkl'
        out_label_fp = open(outpkl_label_file, 'wb')
        label_lines = {}
        process_num = 0
        origin_label = bfs1_path + gtd_path + '_labels.txt'
        
        with open(origin_label) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                key = line[0]
                latex = line[1]
                gtd = latex2gtd(latex)
                # print(key)
                # print(gtd)
                # if('frac' in latex):
                #     print(gtd)
                label_lines[key] = gtd

        

        print ('process files number ', process_num)

        pkl.dump(label_lines, out_label_fp)
        print ('save file done')
        out_label_fp.close()

def save_gtd_align():

    bfs1_path = 'CROHME/'
    gtd_root_path = ''
    gtd_paths = ['train','14_test']

    from utils import load_dict
    dict = load_dict(bfs1_path + 'dictionary_relation_9.txt')
    for gtd_path in gtd_paths:
        outpkl_label_file = bfs1_path + gtd_path + '_relations.pkl'
        
        out_label_fp = open(outpkl_label_file, 'wb')
        label_aligns = {}
        process_num = 0
        gtd_label = bfs1_path + gtd_path + '_label_gtd.pkl'
        fp_label=open(gtd_label, 'rb')
        gtds=pkl.load(fp_label)
        for uid, label_lines in gtds.items():
            # print(uid)
            # print(label_lines)
            align = gen_gtd_relation_align(label_lines,dict)
            label_aligns[uid] = align
            # print(align)
            # break
            

        print ('process files number ', process_num)

        pkl.dump(label_aligns, out_label_fp)
        print ('save file done')
        out_label_fp.close() 

# def to_gtd(C_y, P_y, P_re, worddicts_r, reworddicts_r):
#     C_node = C_y[1::2]
#     P_node = P_y.reshape(-1)[0::2]
#     relation = P_re[0::2]
#     for i in range(C_node.shape[0]):
#         if(P_node[i] == 0):
#             P_node[i] == 106

#         gtd += [[worddicts_r[C_node[i]],0,]] 
if __name__ == '__main__':
    # latex = 'a = \\frac { x } { y } + \sqrt [ c ] { b }'
    # gtd = latex2gtd(latex)
    # for item in gtd:
    #     print('\t\t'.join(item))
    # align = gen_gtd_align(gtd)
    # print(align)
    #     # print('\n')
    # # print(gtd)
    save_gtd_label()
    save_gtd_align()

    # gtd = latex2gtd('\\int _ { 2 } ^ { b } f d \\alpha')
    # print(gtd)

    # # gtd = [['\\frac', '1', '<s>', '0', 'Start'], ['y', '3', '\\frac', '1', 'Below'], ['x', '2', '\\frac', '1', 'Above'], ['</s>', '4', 'y', '3', 'End']]
    # latex = gtd2latex(gtd)
    # print(latex)
