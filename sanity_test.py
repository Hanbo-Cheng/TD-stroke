import time
import os
import re
import numpy as np 
import random
import pickle as pkl
import torch
from torch import optim, nn
from utils.utils import load_dict, prepare_data, gen_sample, weight_init, compute_wer, compute_sacc, cmp_result
from model.encoder_decoder import Encoder_Decoder
from utils.data_iterator import dataIterator, BatchBucket
from utils.gtd import gtd2latex, relation2tree
from model.beam_test import beam_test
import copy

# whether use multi-GPUs
multi_gpu_flag = False
# whether init params
init_param_flag = True
# whether reload params
reload_flag = False

# load configurations
# root_paths
bfs2_path = './CROHME/'
work_path = './'
dictionaries = [bfs2_path + 'dictionary_107.txt', bfs2_path + 'dictionary_relation_9.txt']
datasets = [bfs2_path + 'train_images.pkl', bfs2_path + 'train_label_gtd.pkl', bfs2_path + 'train_relations.pkl']
valid_datasets = [bfs2_path + '14_test_images.pkl', bfs2_path + '14_test_label_gtd.pkl', bfs2_path + '14_test_relations.pkl']
valid_output = [work_path+'results001'+'/symbol_relation/', work_path+'results001'+'/memory_alpha/']
valid_result = [work_path+'results001'+'/valid.cer', work_path+'results001'+'/valid.exprate']
saveto = work_path+'results001'+'/WAP_params.pkl'
last_saveto = work_path+'results001'+'/WAP_params_last.pkl'

# training settings
maxlen = 200
max_epochs = 5000
lrate = 1
my_eps = 1e-6
decay_c = 1e-4
clip_c = 100.

# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 15
validStart = 0
finish_after = 10000000

# model architecture
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
params['lr_lambda'] = 1.
params['lc_lambda_pix'] = 0.5


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
with open(bfs2_path + "14_test_label_gtd.pkl", 'rb') as fp:
    train_gtds = pkl.load(fp)

train_dataIterator = BatchBucket(600, 2100, 200, 800000, 1,  #batch size 
                    datasets[0], datasets[1], datasets[2], 
                    dictionaries[0], dictionaries[1])
train, train_uid = train_dataIterator.get_batches()

valid, valid_uid = dataIterator(valid_datasets[0], valid_datasets[1], 
                    valid_datasets[2], worddicts, reworddicts,
                    1, 8000000, 200, 8000000)  




# statistics
history_errs = []

for eidx in range(1):
    n_samples = 0
    ud_epoch = time.time()

    # train_dataIterator._reset()
    # train, train_uid = train_dataIterator.get_batches()
    # random.shuffle(train)
    total_number = 0
    cnt = 0
    for x, ly, ry, re, ma, lp, rp in valid:
        

        ud_start = time.time()
        n_samples += len(x)

        x, x_mask, C_y, y_mask, P_y, P_re, C_re, C_re_mask, lp, rp = \
                prepare_data(params, x, ly, ry, re, ma, lp, rp)

        C_y = C_y.reshape(-1)
        C_re = C_re.reshape(-1,9)
        gtd = relation2tree(C_y[1::2], copy.deepcopy(C_re[1::2]) , worddicts_r, reworddicts_r)

        
        latex = gtd2latex(gtd)

        uid = valid_uid[total_number]
        groud_truth_gtd = train_gtds[uid]
        groud_truth_latex = gtd2latex(groud_truth_gtd)

        latex_distance, latex_length = cmp_result(groud_truth_latex, latex)
        total_number += 1
        # if('frac' in latex):
        #     print(latex)
        if(latex_distance != 0):
            print("GT: ",groud_truth_latex)
            print("Trans: ",latex)
            cnt +=1
            



        

        


    print(cnt)
    print('Seen %d samples' % total_number)

    # early stop
    if estop:
        break
