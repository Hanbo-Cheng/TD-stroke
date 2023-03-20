import time
import os
import re
import numpy as np 
import random
import argparse
import pickle as pkl
import torch
from torch import optim, nn
from utils.utils import load_dict, prepare_data, gen_sample, weight_init, compute_wer, compute_sacc, cmp_result
from model.encoder_decoder import Encoder_Decoder
from model.beam_test import beam_test
from utils.data_iterator import dataIterator, BatchBucket
from utils.gtd import gtd2latex, relation2tree


parser = argparse.ArgumentParser()
parser.add_argument('beam_size', type=int, default=10)
parser.add_argument('data_path', type=str)
parser.add_argument('output_path', type=str)
parser.add_argument('pretrained_models', type=str)
parser.add_argument('mode', type=str)
args = parser.parse_args()

# load configurations
# root_paths
data_path    = args.data_path
output_path  = args.output_path
if not os.path.exists(output_path):
    print('Creating ', output_path)
    os.mkdir(output_path)
dictionaries =   [data_path + 'dictionary_object.txt', 
                  data_path + 'dictionary_relation.txt']
valid_datasets = [data_path + 'valid_images.pkl', 
                  data_path + 'valid_labels_change.pkl', 
                  data_path + 'valid_relations_change.pkl']
valid_result =   [output_path+'/greedy_valid.result', 
                  output_path+'/beam_valid.result']
if args.mode == 'test':
    valid_datasets =  [data_path + 'test_images.pkl', 
                       data_path + 'test_labels_change.pkl', 
                       data_path + 'test_relations_change.pkl']
    valid_result =    [output_path+'/greedy_test.result', 
                       output_path+'/beam_test.result']
pretrained_models = args.pretrained_models.split(',')

# test settings
maxlen = 200
beam_size = args.beam_size
# model architecture
params = {}
params['n'] = 256
params['m'] = 256
params['dim_attention'] = 512
params['D'] = 936
params['K'] = 104
params['Kre'] = 9
params['mre'] = 256
params['maxlen'] = maxlen
params['growthRate'] = 24
params['reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 1
params['lc_lambda'] = 1.
params['lr_lambda'] = 1

# load dictionary
worddicts = load_dict(dictionaries[0])
print ('total chars',len(worddicts))
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

reworddicts = load_dict(dictionaries[1])
print ('total relations',len(reworddicts))
reworddicts_r = [None] * len(reworddicts)
for kk, vv in reworddicts.items():
    reworddicts_r[vv] = kk

# load valid gtd
with open(valid_datasets[1], 'rb') as fp:
    valid_gtds = pkl.load(fp)

valid, valid_uid = dataIterator(valid_datasets[0], valid_datasets[1], 
                    valid_datasets[2], worddicts, reworddicts,
                    1, 8000000, 200, 8000000)     
# valid = valid[:100]
# model
WAP_models = []
for pretrained_model in pretrained_models:
    WAP_model = Encoder_Decoder(params)
    WAP_model.load_state_dict(
        torch.load(pretrained_model, map_location=lambda storage,loc:storage))
    WAP_model.cuda()
    WAP_model.eval()
    WAP_models.append(WAP_model)


fp_results = open(valid_result[1], 'w')
with torch.no_grad():
    number_right = 0
    latex_right = 0
    total_distance = 0
    total_length = 0
    total_latex_distance = 0
    total_latex_length = 0
    total_number = 0
    print('begin sampling')
    ud_epoch = time.time()
    valid_count_idx = 0
    iii = 0
    for x, ly, ry, re, ma, lp, rp in valid:
        if (total_number + 1) % 50 == 0:
            print(total_number + 1)
        x, x_mask, C_y, y_mask, P_y, P_re, C_re, C_re_mask, lp, rp = \
            prepare_data(params, x, ly, ry, re, ma, lp, rp)

        L, B = C_y.shape[:2]
        x = torch.from_numpy(x).cuda()  # (batch,1,H,W)
        x_mask = torch.from_numpy(x_mask).cuda()  # (batch,H,W)
        #C_y = torch.from_numpy(C_y).to(torch.long).cuda()  # (seqs_y,batch)
        lengths_gt = (y_mask > 0.5).sum(0)
        y_mask = torch.from_numpy(y_mask).cuda()  # (seqs_y,batch)
        P_y = torch.from_numpy(P_y).to(torch.long).cuda()  # (seqs_y,batch)
        P_re = torch.from_numpy(P_re).to(torch.long).cuda()  # (seqs_y,batch)
        #C_re = torch.from_numpy(C_re).cuda()  # (batch,seqs_y,seqs_y)

        object_predict, relation_predict \
            = beam_test(WAP_models, x, x_mask, L+1, P_y[0], P_re[0], y_mask[0], beam_size)

        gtd = relation2tree(object_predict, relation_predict, worddicts_r, reworddicts_r)
        latex = gtd2latex(gtd)
        
        uid = valid_uid[total_number]
        groud_truth_gtd = valid_gtds[uid]
        groud_truth_latex = gtd2latex(groud_truth_gtd)

        child = C_y[:, 0]
        distance, length = cmp_result(object_predict, child)
        total_number += 1

        if distance == 0:
            number_right += 1
            fp_results.write(uid + '\tObject True\t')
        else:
            fp_results.write(uid + '\tObject False\t')
        
        latex_distance, latex_length = cmp_result(groud_truth_latex, latex)
        if latex_distance == 0:
            latex_right += 1
            fp_results.write('Latex True\n')
        else:
            fp_results.write('Latex False\n')


        total_distance += distance
        total_length += length
        total_latex_distance += latex_distance
        total_latex_length += latex_length

        fp_results.write(groud_truth_latex+'\n')
        fp_results.write(latex+'\n')

        for c in child:
            fp_results.write(worddicts_r[c] + ' ')
        fp_results.write('\n')

        for c in object_predict:
            fp_results.write(worddicts_r[c] + ' ')
        fp_results.write('\n')

    wer = total_distance / total_length * 100
    sacc = number_right / total_number * 100
    latex_wer = total_latex_distance / total_latex_length * 100
    latex_acc = latex_right / total_number * 100
    fp_results.close()
                
    print('valid set decode done')
    ud_epoch = (time.time() - ud_epoch) / 60.
    print('WER', wer, 'SACC', sacc, 'Latex', latex_wer, 'Latex Acc', latex_acc, 'epoch cost time ... ', ud_epoch)


