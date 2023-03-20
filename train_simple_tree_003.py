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

# whether use multi-GPUs
multi_gpu_flag = False
# whether init params
init_param_flag = True
# whether reload params
reload_flag = False

# load configurations
# root_paths
bfs2_path = '/yrfs2/cv9/cjwu4/009_off_HMER/data_offline/'
work_path = './'
dictionaries = [bfs2_path + 'dictionary_object.txt', bfs2_path + 'dictionary_relation.txt']
datasets = [bfs2_path + 'train_images.pkl', bfs2_path + 'train_labels.pkl', bfs2_path + 'train_relations.pkl']
valid_datasets = [bfs2_path + 'valid_images.pkl', bfs2_path + 'valid_labels.pkl', bfs2_path + 'valid_relations.pkl']
valid_output = [work_path+'results003'+'/symbol_relation/', work_path+'results003'+'/memory_alpha/']
valid_result = [work_path+'results003'+'/valid.cer', work_path+'results003'+'/valid.exprate']
saveto = work_path+'results003'+'/WAP_params.pkl'
last_saveto = work_path+'results003'+'/WAP_params_last.pkl'

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
with open(bfs2_path + 'valid_labels.pkl', 'rb') as fp:
    valid_gtds = pkl.load(fp)

train_dataIterator = BatchBucket(600, 2100, 200, 800000, 16,
                    datasets[0], datasets[1], datasets[2], 
                    dictionaries[0], dictionaries[1])
train, train_uid = train_dataIterator.get_batches()

valid, valid_uid = dataIterator(valid_datasets[0], valid_datasets[1], 
                    valid_datasets[2], worddicts, reworddicts,
                    1, 8000000, 200, 8000000)  



# display
uidx = 0  # count batch
lpred_loss_s = 0.  # count loss
repred_loss_s = 0.
loss_s = 0.

ud_s = 0  # time for training an epoch
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100
WER = 100


# initialize model
WAP_model = Encoder_Decoder(params)
if init_param_flag:
    WAP_model.apply(weight_init)
if multi_gpu_flag:
    WAP_model = nn.DataParallel(WAP_model, device_ids=[0, 1, 2, 3])
if reload_flag:
    WAP_model.load_state_dict(torch.load(last_saveto,map_location=lambda storage,loc:storage))
WAP_model.cuda()

# print model's parameters
# model_params = WAP_model.named_parameters()
# for k, v in model_params:
#     print(k)

# loss function
# criterion = torch.nn.CrossEntropyLoss(reduce=False)
# optimizer
optimizer = optim.Adadelta(WAP_model.parameters(), lr=lrate, eps=my_eps, weight_decay=decay_c)

print('Optimization')

# statistics
history_errs = []

for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()

    train_dataIterator._reset()
    train, train_uid = train_dataIterator.get_batches()
    random.shuffle(train)
    if validFreq == -1:
        validFreq = len(train)
    if saveFreq == -1:
        saveFreq = len(train)
    if sampleFreq == -1:
        sampleFreq = len(train)
    for x, ly, ry, re, ma, lp, rp in train:
        
        WAP_model.train()
        ud_start = time.time()
        n_samples += len(x)
        uidx += 1
        x, x_mask, C_y, y_mask, P_y, P_re, C_re, C_re_mask, lp, rp = \
                prepare_data(params, x, ly, ry, re, ma, lp, rp)

        length = C_y.shape[0]
        x = torch.from_numpy(x).cuda()  # (batch,1,H,W)
        x_mask = torch.from_numpy(x_mask).cuda()  # (batch,H,W)
        C_y = torch.from_numpy(C_y).to(torch.long).cuda()  # (seqs_y,batch)
        y_mask = torch.from_numpy(y_mask).cuda()  # (seqs_y,batch)
        P_y = torch.from_numpy(P_y).to(torch.long).cuda()  # (seqs_y,batch)
        P_re = torch.from_numpy(P_re).to(torch.long).cuda()  # (seqs_y,batch)
        C_re = torch.from_numpy(C_re).cuda()  # (batch,seqs_y,seqs_y)
        C_re_mask = torch.from_numpy(C_re_mask).cuda()

        # forward
        loss, object_loss, relation_loss =  WAP_model(params, x, x_mask, 
            C_y, P_y, C_re, P_re, y_mask, C_re_mask, length)


        lpred_loss_s += object_loss.item()
        repred_loss_s += relation_loss.item()
        loss_s += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_c > 0.:
            torch.nn.utils.clip_grad_norm_(WAP_model.parameters(), clip_c)

        # update
        optimizer.step()

        ud = time.time() - ud_start
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            lpred_loss_s /= dispFreq
            repred_loss_s /= dispFreq
            print ('Epoch', eidx, ' Update', uidx, ' Cost_object %.7f, Cost_relation %.7f' % \
                (np.float(lpred_loss_s),  np.float(repred_loss_s) ), \
                ' UD %.3f' % ud_s, ' lrate', lrate, ' eps', my_eps, ' bad_counter', bad_counter)
            ud_s = 0
            loss_s = 0.
            lpred_loss_s = 0.
            repred_loss_s = 0.

        if np.mod(uidx, saveFreq) == 0:
            print('Saving latest model params ... ')
            torch.save(WAP_model.state_dict(), last_saveto)
        
        # validation
        if np.mod(uidx, sampleFreq) == 0 and (eidx % 2) == 0:
            number_right = 0
            total_distance = 0
            total_length = 0
            latex_right = 0
            total_latex_distance = 0
            total_latex_length = 0
            total_number = 0

            print('begin sampling')
            ud_epoch_train = (time.time() - ud_epoch) / 60.
            print('epoch training cost time ... ', ud_epoch_train)
            WAP_model.eval()
            
            fp_results = open('reuslts-debug.txt', 'w')
            with torch.no_grad():
                valid_count_idx = 0
                for x, ly, ry, re, ma, lp, rp in valid:
                    x, x_mask, C_y, y_mask, P_y, P_re, C_re, C_re_mask, lp, rp = \
                    prepare_data(params, x, ly, ry, re, ma, lp, rp)
            
                    L, B = C_y.shape[:2]
                    x = torch.from_numpy(x).cuda()  # (batch,1,H,W)
                    x_mask = torch.from_numpy(x_mask).cuda()  # (batch,H,W)
                    lengths_gt = (y_mask > 0.5).sum(0)
                    y_mask = torch.from_numpy(y_mask).cuda()  # (seqs_y,batch)
                    P_y = torch.from_numpy(P_y).to(torch.long).cuda()  # (seqs_y,batch)
                    P_re = torch.from_numpy(P_re).to(torch.long).cuda()  # (seqs_y,batch)

                    object_predicts, P_masks, relation_table_static, _ \
                        = WAP_model.greedy_inference(x, x_mask, L+1, P_y[0], P_re[0], y_mask[0])
                    object_predicts, P_masks = object_predicts.cpu().numpy(), P_masks.cpu().numpy()
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
            print('WER', wer, 'SACC', sacc, 'Latex WER', latex_wer, 'Latex SACC', latex_acc,  'epoch cost time ... ', ud_epoch)

            # the first time validation or better model
            if latex_wer <= WER:
                WER = latex_wer
                bad_counter = 0
                print('Saving best model params ... ')
                torch.save(WAP_model.state_dict(), saveto)
            else:
                bad_counter += 1
                if bad_counter > patience:
                    if halfLrFlag == 2:
                        print('Early Stop!')
                        estop = True
                        break
                    else:
                        print('Lr decay and retrain!')
                        bad_counter = 0
                        lrate = lrate / 10.
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lrate
                        halfLrFlag += 1

        # finish after these many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            estop = True
            break

    print('Seen %d samples' % n_samples)

    # early stop
    if estop:
        break
