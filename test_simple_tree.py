import time
import os
import re
from tqdm import tqdm
import numpy as np 
import random
import pickle as pkl
import torch
from torch import optim, nn
from utils.utils import load_dict, prepare_data, gen_sample, weight_init, compute_wer, compute_sacc, cmp_result
from model.encoder_decoder import Encoder_Decoder
from utils.data_iterator import dataIterator, BatchBucket
from utils.gtd import gtd2latex, relation2tree


# load configurations
bfs2_path = '/yrfs2/cv9/cjwu4/009_off_HMER/jianshu/'  
work_path = './'
dictionaries = [bfs2_path + 'dictionary_object.txt', bfs2_path + 'dictionary_relation.txt']
datasets = []
# datasets.append([bfs2_path + 'jiaming-test19-py3.pkl', bfs2_path + 'test_19_labels.pkl', bfs2_path + 'test_19_relations.pkl'])
# datasets.append([bfs2_path + 'jiaming-test16-py3.pkl', bfs2_path + 'test_16_labels.pkl', bfs2_path + 'test_16_relations.pkl'])
datasets.append([bfs2_path + 'jiaming-test14-py3.pkl', bfs2_path + 'test_14_labels.pkl', bfs2_path + 'test_14_relations.pkl'])

# test settings
maxlen = 200
beam_size = 10

params = {}
params['n'] = 256
params['m'] = 256
params['re_m'] = 64
params['dim_attention'] = 512
params['D'] = 936
params['K'] = 107
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
for di in range(len(datasets)):
    with open(datasets[di][1], 'rb') as fp:
        valid_gtds = pkl.load(fp)
    valid, valid_uid = dataIterator(datasets[di][0], datasets[di][1], 
                    datasets[di][2], worddicts, reworddicts,
                    1, 8000000, 200, 8000000)     

    for i in range(1,4):
        valid_result = [work_path+'results00' + str(i) +'/2014_test.result', work_path+'results00' + str(i) +'/beam_test.result']
        saveto = work_path+'results00' + str(i) +'/WAP_params.pkl'
        # model
        WAP_model = Encoder_Decoder(params)
        print('Loading', i , 'model params....', end='\t')
        WAP_model.load_state_dict(torch.load(saveto,map_location=lambda storage,loc:storage))
        print('Done!')
        WAP_model.cuda()


        fp_results = open(valid_result[0], 'w')
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
            WAP_model.eval()
            valid_count_idx = 0
            iii = 0
            for x, ly, ry, re, ma, lp, rp in tqdm(valid):
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

                object_predicts, P_masks, relation_table_static, _ \
                    = WAP_model.greedy_inference(x, x_mask, L+1, P_y[0], P_re[0], y_mask[0])
                object_predicts, P_masks = object_predicts.cuda().numpy(), P_masks.cuda().numpy()
                relation_table_static = relation_table_static.numpy()
                for bi in range(B):
                    length_predict = min((P_masks[bi, :] > 0.5).sum() + 1, P_masks.shape[1])
                    object_predict = object_predicts[:int(length_predict), bi]
                    relation_predict = relation_table_static[bi, :int(length_predict), :]
                    gtd = relation2tree(object_predict[1::2], relation_predict[1::2], worddicts_r, reworddicts_r)
                    latex = gtd2latex(gtd)
                    
                    uid = valid_uid[total_number]

                    groud_truth_gtd = valid_gtds[uid]
                    groud_truth_latex = gtd2latex(groud_truth_gtd)
                    #print(gtd)
                    #print(groud_truth_gtd)
                    
                    child = C_y[:int(lengths_gt[bi]), bi]
                    distance, length = cmp_result(object_predict, child)
                    total_number += 1

                    if distance == 0:
                        number_right += 1
                        fp_results.write(uid + 'Object True\t')
                    else:
                        fp_results.write(uid + 'Object False\t')
                    
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

                    for li in range(lengths_gt[bi]):
                        fp_results.write(worddicts_r[child[li]] + ' ')
                    fp_results.write('\n')

                    for li in range(length_predict):
                        fp_results.write(worddicts_r[object_predict[li]] + ' ')
                    fp_results.write('\n')

            wer = total_distance / total_length * 100
            sacc = number_right / total_number * 100
            latex_wer = total_latex_distance / total_latex_length * 100
            latex_acc = latex_right / total_number * 100
            fp_results.close()
                        
            print('valid set decode done')
            ud_epoch = (time.time() - ud_epoch) / 60.
            print('WER', wer, 'SACC', sacc, 'Latex', latex_wer, 'Latex Acc', latex_acc, 'epoch cost time ... ', ud_epoch)
