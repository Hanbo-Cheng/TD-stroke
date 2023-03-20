import random
import numpy as np
import copy
import sys
import pickle as pkl
import torch
from torch import nn

# load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    i = 0
    for l in stuff:
        w = l.split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon

class Node:
    def __init__(self, id):
        self.id = id
        self.path = []

def random_order(root, ids):
    ids.append(root.id)
    if len(root.path) >= 2:
        random.shuffle(root.path)
    for p in root.path:
        random_order(p, ids)
    return ids

def random_label(relation):
    id = 0
    root = Node(id)
    id += 1
    node = root
    stack_relation = [relation[0]]
    stack_node = [node]
    while stack_relation:
        re = stack_relation[-1]
        node = stack_node[-1]
        if sum(re[:-1]) > 0:    # CHB：修改了原本是re[:-1]
            for ri, r in enumerate(re[:-1]):  # CHB：修改了原本是re[:-1]
                if r:
                    c_node = Node(id)
                    stack_relation.append(relation[id])
                    stack_node.append(c_node)
                    node.path.append(c_node)
                    re[ri] = 0
                    id += 1
                    break
        else:
            stack_relation.pop()
            stack_node.pop()
    random_ids = random_order(root, [])
    return random_ids



# create batch
def prepare_data(params, images_x, seqs_ly, seqs_ry, seqs_re, seqs_ma, seqs_lp, seqs_rp):
    images_x, online_x = images_x
    heights_x = [s.shape[0] for s in images_x]
    widths_x = [s.shape[1] for s in images_x]
    lengths_ly = [len(s)-1 for s in seqs_ly]
    lengths_ry = [len(s)-1 for s in seqs_ry]

    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_ly = np.max(lengths_ly) * 2
    maxlen_ry = np.max(lengths_ry) * 2

    x = np.zeros((n_samples, params['input_channels'], max_height_x, max_width_x)).astype(np.float32) - 1
    ly = np.zeros((maxlen_ly, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    ry = np.zeros((maxlen_ry, n_samples)).astype(np.int64)
    re = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
    ma = np.zeros((maxlen_ly, n_samples, 9)).astype(np.float32)
    lp = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
    rp = np.zeros((maxlen_ry, n_samples)).astype(np.int64)

    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
    ry_mask = np.zeros((maxlen_ry, n_samples)).astype(np.float32)
    re_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
    ma_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)

    for idx, [s_x, s_online_x, s_ly, s_ry, s_re, s_ma, s_lp, s_rp] in enumerate(zip(images_x, online_x, seqs_ly, seqs_ry, seqs_re, seqs_ma, seqs_lp, seqs_rp)):
        x[idx, 0, :heights_x[idx], :widths_x[idx]] = (255 - s_x) / 255.
        x[idx, 1:, :heights_x[idx], :widths_x[idx]] = s_online_x.transpose(2,0,1)
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        s_ma = s_ma[1:, :]
        # rand_idxs = random_label(copy.deepcopy(s_ma))
        rand_idxs = range(s_ma.shape[0])

        
        for i in range(lengths_ry[idx]):
            #child
            ly[2*i,   idx] = 106  # CHB:根据使用的dict修改
            ly[2*i+1, idx] = s_ly[rand_idxs[i]]
            ma[2*i,   idx, 0] = 1
            ma[2*i+1, idx, :s_ma.shape[-1]] = s_ma[rand_idxs[i], :]

            #parent
            ry[2*i,   idx] = s_ry[rand_idxs[i]]
            ry[2*i+1, idx] = 106
            re[2*i,   idx] = s_re[rand_idxs[i]]
            re[2*i+1, idx] = 0
            #lp[2*i, idx] = 2*s_lp[rand_idxs[i]]
            #rp[2*i,   idx] = s_rp[rand_idxs[i]]
            #rp[2*i+1, idx] = s_rp[rand_idxs[i]]
        ry[0, idx] = 0
        ma_mask[:lengths_ly[idx]*2, idx] = 1.
        y_mask[:lengths_ly[idx]*2, idx] = 1.

    return x, x_mask, ly, y_mask, ry, re, ma, ma_mask, lp, rp

def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30, rpos_beam=3):
    
    sample = []
    sample_score = []
    rpos_sample = []
    # rpos_sample_score = []
    relation_sample = []

    live_k = 1
    dead_k = 0  # except init, live_k = k - dead_k

    # current living paths and corresponding scores(-log)
    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)
    hyp_rpos_samples = [[]] * live_k
    hyp_relation_samples = [[]] * live_k
    # get init state, (1,n) and encoder output, (1,D,H,W)
    next_state, ctx0 = model.f_init(x)
    next_h1t = next_state
    # -1 -> My_embedding -> 0 tensor(1,m)
    next_lw = -1 * torch.ones(1, dtype=torch.int64).cuda()
    next_calpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()  # (live_k,H,W)
    next_palpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    nextemb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()
    nextePmb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()    

    for ii in range(maxlen):
        ctxP = ctx0.repeat(live_k, 1, 1, 1)  # (live_k,D,H,W)
        next_lpos = ii * torch.ones(live_k, dtype=torch.int64).cuda()
        next_h01, next_ma, next_ctP, next_pa, next_palpha_past, nextemb_memory, nextePmb_memory = \
                    model.f_next_parent(params, next_lw, next_lpos, ctxP, next_state, next_h1t, next_palpha_past, nextemb_memory, nextePmb_memory, ii)
        next_ma = next_ma.cuda().numpy()
        # next_ctP = next_ctP.cuda().numpy()
        next_palpha_past = next_palpha_past.cuda().numpy()
        nextemb_memory = nextemb_memory.cuda().numpy()
        nextePmb_memory = nextePmb_memory.cuda().numpy()

        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2)) # batch * Matt * dim
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        
        next_rpos = next_ma.argsort(axis=1)[:,-rpos_beam:] # topK parent index; batch * topK
        n_gaps = nextemb_memory.shape[1]
        n_batch = nextemb_memory.shape[0]
        next_rpos_gap = next_rpos + n_gaps * np.arange(n_batch)[:, None]
        next_remb_memory = nextemb_memory.reshape([n_batch*n_gaps, nextemb_memory.shape[-1]])
        next_remb = next_remb_memory[next_rpos_gap.flatten()] # [batch*rpos_beam, emb_dim]
        rpos_scores = next_ma.flatten()[next_rpos_gap.flatten()] # [batch*rpos_beam,]

        # next_ctPC = next_ctP.repeat(1, 1, rpos_beam)
        # next_ctPC = torch.reshape(next_ctPC, (-1, next_ctP.shape[1]))
        ctxC = ctx0.repeat(live_k*rpos_beam, 1, 1, 1)
        next_ctPC = torch.zeros(next_ctP.shape[0]*rpos_beam, next_ctP.shape[1]).cuda()
        next_h01C = torch.zeros(next_h01.shape[0]*rpos_beam, next_h01.shape[1]).cuda()
        next_calpha_pastC = torch.zeros(next_calpha_past.shape[0]*rpos_beam, next_calpha_past.shape[1], next_calpha_past.shape[2]).cuda()
        for bidx in range(next_calpha_past.shape[0]):
            for ridx in range(rpos_beam):
                next_ctPC[bidx*rpos_beam+ridx] = next_ctP[bidx]
                next_h01C[bidx*rpos_beam+ridx] = next_h01[bidx]
                next_calpha_pastC[bidx*rpos_beam+ridx] = next_calpha_past[bidx]
        next_remb = torch.from_numpy(next_remb).cuda()

        next_lp, next_rep, next_state, next_h1t, next_ca, next_calpha_past, next_re = \
                    model.f_next_child(params, next_remb, next_ctPC, ctxC, next_h01C, next_calpha_pastC)

        next_lp = next_lp.cuda().numpy()
        next_state = next_state.cuda().numpy()
        next_h1t = next_h1t.cuda().numpy()
        next_calpha_past = next_calpha_past.cuda().numpy()
        next_re = next_re.cuda().numpy()

        hyp_scores = np.tile(hyp_scores[:, None], [1, rpos_beam]).flatten()
        cand_scores = hyp_scores[:, None] - np.log(next_lp+1e-10)- np.log(rpos_scores+1e-10)[:,None]
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        voc_size = next_lp.shape[1]
        trans_indices = ranks_flat // voc_size
        trans_indicesP = ranks_flat // (voc_size*rpos_beam)
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        # update paths
        new_hyp_samples = []
        new_hyp_scores = np.zeros(k-dead_k).astype('float32')
        new_hyp_rpos_samples = []
        new_hyp_relation_samples = []
        new_hyp_states = []
        new_hyp_h1ts = []
        new_hyp_calpha_past = []
        new_hyp_palpha_past = []
        new_hyp_emb_memory = []
        new_hyp_ePmb_memory = []
        
        for idx, [ti, wi, tPi] in enumerate(zip(trans_indices, word_indices, trans_indicesP)):
            new_hyp_samples.append(hyp_samples[tPi]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_rpos_samples.append(hyp_rpos_samples[tPi]+[next_rpos.flatten()[ti]])
            new_hyp_relation_samples.append(hyp_relation_samples[tPi]+[next_re[ti]])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_h1ts.append(copy.copy(next_h1t[ti]))
            new_hyp_calpha_past.append(copy.copy(next_calpha_past[ti]))
            new_hyp_palpha_past.append(copy.copy(next_palpha_past[tPi]))
            new_hyp_emb_memory.append(copy.copy(nextemb_memory[tPi]))
            new_hyp_ePmb_memory.append(copy.copy(nextePmb_memory[tPi]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_rpos_samples = []
        hyp_relation_samples = []
        hyp_states = []
        hyp_h1ts = []
        hyp_calpha_past = []
        hyp_palpha_past = []
        hyp_emb_memory = []
        hyp_ePmb_memory = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0: # <eol>
                sample_score.append(new_hyp_scores[idx])
                sample.append(new_hyp_samples[idx])
                rpos_sample.append(new_hyp_rpos_samples[idx])
                relation_sample.append(new_hyp_relation_samples[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_scores.append(new_hyp_scores[idx])
                hyp_samples.append(new_hyp_samples[idx])
                hyp_rpos_samples.append(new_hyp_rpos_samples[idx])
                hyp_relation_samples.append(new_hyp_relation_samples[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_h1ts.append(new_hyp_h1ts[idx])
                hyp_calpha_past.append(new_hyp_calpha_past[idx])
                hyp_palpha_past.append(new_hyp_palpha_past[idx])
                hyp_emb_memory.append(new_hyp_emb_memory[idx])
                hyp_ePmb_memory.append(new_hyp_ePmb_memory[idx])   
                    
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_lw = np.array([w[-1] for w in hyp_samples])  # each path's final symbol, (live_k,)
        next_state = np.array(hyp_states)  # h2t, (live_k,n)
        next_h1t = np.array(hyp_h1ts)
        next_calpha_past = np.array(hyp_calpha_past)  # (live_k,H,W)
        next_palpha_past = np.array(hyp_palpha_past)
        nextemb_memory = np.array(hyp_emb_memory)
        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2))
        nextePmb_memory = np.array(hyp_ePmb_memory)
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        next_lw = torch.from_numpy(next_lw).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_h1t = torch.from_numpy(next_h1t).cuda()
        next_calpha_past = torch.from_numpy(next_calpha_past).cuda()
        next_palpha_past = torch.from_numpy(next_palpha_past).cuda()
        nextemb_memory = torch.from_numpy(nextemb_memory).cuda()
        nextePmb_memory = torch.from_numpy(nextePmb_memory).cuda()

    return sample_score, sample, rpos_sample, relation_sample

def prepare_data_stroke(params, images_x, x_mask, seqs_ly, seqs_ry, seqs_re, seqs_ma, seqs_lp, seqs_rp, mod = 'train'):
    images_x, online_x = images_x
    off_mask, on_mask = x_mask

    seqs_ma =  seqs_ma
    heights_x = [s.shape[0] for s in images_x]
    widths_x = [s.shape[1] for s in images_x]
    lengths_ly = [len(s)-1 for s in seqs_ly]
    lengths_ry = [len(s)-1 for s in seqs_ry]

    lengths_x = [len(s) for s in online_x]  # each sample's n-points in a batch
    n_strokes = [m.shape[0] for m in on_mask]
    max_on_len = np.max(lengths_x) + 1 

    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_ly = np.max(lengths_ly) * 2
    maxlen_ry = np.max(lengths_ry) * 2

    x = np.zeros((n_samples, params['input_channels'], max_height_x, max_width_x)).astype(np.float32) 
    on_x = np.zeros((max_on_len, n_samples, params['online_input_channels'])).astype('float32')
    ly = np.zeros((maxlen_ly, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    ry = np.zeros((maxlen_ry, n_samples)).astype(np.int64)
    re = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
    ma = np.zeros((maxlen_ly, n_samples, 9)).astype(np.float32)
    lp = np.zeros((maxlen_ly, n_samples)).astype(np.int64)
    rp = np.zeros((maxlen_ry, n_samples)).astype(np.int64)

    # TODO: mask
    on_stroke_masks = []  
    max_num_stroke = np.max(n_strokes)
    maxlen_y = np.max(lengths_ly) + 1
    # stroke_a = np.zeros((max_num_stroke, n_samples, maxlen_y)).astype(np.float32)
    on_stroke_a_mask = np.zeros((max_num_stroke, n_samples, maxlen_y)).astype(np.float32)


    on_x_mask = np.zeros((max_on_len, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
    ry_mask = np.zeros((maxlen_ry, n_samples)).astype(np.float32)
    re_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
    ma_mask = np.zeros((maxlen_ly, n_samples)).astype(np.float32)
    
    
    for idx, s_x in enumerate(online_x):
        on_x[:lengths_x[idx], idx, :] = s_x
        on_x_mask[:lengths_x[idx] + 1, idx] = 1.

        on_stroke_mask = np.zeros((n_strokes[idx], max_on_len)).astype(np.float32)
        on_stroke_mask[:, :lengths_x[idx]] = on_mask[idx] * 1.
        on_stroke_masks.append(torch.from_numpy(on_stroke_mask).cuda()) # TODO: stroke mask处理
            # stroke_a[:n_strokes[idx], idx, :lengths_ly[idx]] = stroke_seqs_a[idx] * 1.
            # on_stroke_a_mask[:n_strokes[idx], idx, :lengths_ly[idx]] = 1.


    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    off_x = np.zeros((n_samples, params['input_channels'], max_height_x, max_width_x)).astype(
        np.float32)  # (batch,1,H,W))
    off_x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)  # (batch,H,W))
    off_stroke_masks = []  # 元素数=batch, 每个元素的size为(n_strokes, max_height_x, max_width_x), 补充部分为0
    for idx, s_x in enumerate(images_x):
        off_x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        off_x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        off_stroke_mask = np.zeros((off_mask[idx].shape[0], max_height_x, max_width_x)).astype(np.float32)
        off_stroke_mask[:, :heights_x[idx], :widths_x[idx]] = off_mask[idx] * 1.
        off_stroke_masks.append(torch.from_numpy(off_stroke_mask).cuda())
    for idx, [ s_ly, s_ry, s_re, s_ma, s_lp, s_rp] in enumerate(zip( seqs_ly, seqs_ry, seqs_re, seqs_ma, seqs_lp, seqs_rp)):
        # x[idx, 0, :heights_x[idx], :widths_x[idx]] = (255 - s_x) / 255.

        # x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        s_ma = s_ma[1:, :]
        # rand_idxs = random_label(copy.deepcopy(s_ma))
        rand_idxs = range(s_ma.shape[0])

        
        for i in range(lengths_ry[idx]):
            #child
            ly[2*i,   idx] = 106  # CHB:根据使用的dict修改
            ly[2*i+1, idx] = s_ly[rand_idxs[i]]
            ma[2*i,   idx, 0] = 1
            ma[2*i+1, idx, :s_ma.shape[-1]] = s_ma[rand_idxs[i], :]

            #parent
            ry[2*i,   idx] = s_ry[rand_idxs[i]]
            ry[2*i+1, idx] = 106
            re[2*i,   idx] = s_re[rand_idxs[i]]
            re[2*i+1, idx] = 0
            #lp[2*i, idx] = 2*s_lp[rand_idxs[i]]
            #rp[2*i,   idx] = s_rp[rand_idxs[i]]
            #rp[2*i+1, idx] = s_rp[rand_idxs[i]]
        ry[0, idx] = 0
        ma_mask[:lengths_ly[idx]*2, idx] = 1.
        y_mask[:lengths_ly[idx]*2, idx] = 1.

    return off_x ,on_x, off_x_mask, on_x_mask, off_stroke_masks, on_stroke_masks, ly, y_mask, ry, re, ma, ma_mask, lp, rp

def gen_sample_stroke(model, off_x, on_x, off_m, on_m, params, gpu_flag, k=1, maxlen=30, rpos_beam=3):
    
    sample = []
    sample_score = []
    rpos_sample = []
    # rpos_sample_score = []
    relation_sample = []

    live_k = 1
    dead_k = 0  # except init, live_k = k - dead_k

    # current living paths and corresponding scores(-log)
    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)
    hyp_rpos_samples = [[]] * live_k
    hyp_relation_samples = [[]] * live_k
    # get init state, (1,n) and encoder output, (1,D,H,W)
    next_state, ctx0 = model.f_init(on_x, on_m, off_x, off_m)
    next_h1t = next_state
    # -1 -> My_embedding -> 0 tensor(1,m)
    next_lw = -1 * torch.ones(1, dtype=torch.int64).cuda()
    next_calpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()  # (live_k,H,W)
    next_palpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    nextemb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()
    nextePmb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()    

    for ii in range(maxlen):
        ctxP = ctx0.repeat(live_k, 1, 1)  # (live_k,D,H,W)
        next_lpos = ii * torch.ones(live_k, dtype=torch.int64).cuda()
        next_h01, next_ma, next_ctP, next_pa, next_palpha_past, nextemb_memory, nextePmb_memory = \
                    model.f_next_parent(params, next_lw, next_lpos, ctxP, next_state, next_h1t, next_palpha_past, nextemb_memory, nextePmb_memory, ii)
        next_ma = next_ma.cuda().numpy()
        # next_ctP = next_ctP.cuda().numpy()
        next_palpha_past = next_palpha_past.cuda().numpy()
        nextemb_memory = nextemb_memory.cuda().numpy()
        nextePmb_memory = nextePmb_memory.cuda().numpy()

        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2)) # batch * Matt * dim
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        
        next_rpos = next_ma.argsort(axis=1)[:,-rpos_beam:] # topK parent index; batch * topK
        n_gaps = nextemb_memory.shape[1]
        n_batch = nextemb_memory.shape[0]
        next_rpos_gap = next_rpos + n_gaps * np.arange(n_batch)[:, None]
        next_remb_memory = nextemb_memory.reshape([n_batch*n_gaps, nextemb_memory.shape[-1]])
        next_remb = next_remb_memory[next_rpos_gap.flatten()] # [batch*rpos_beam, emb_dim]
        rpos_scores = next_ma.flatten()[next_rpos_gap.flatten()] # [batch*rpos_beam,]

        # next_ctPC = next_ctP.repeat(1, 1, rpos_beam)
        # next_ctPC = torch.reshape(next_ctPC, (-1, next_ctP.shape[1]))
        ctxC = ctx0.repeat(live_k*rpos_beam, 1, 1, 1)
        next_ctPC = torch.zeros(next_ctP.shape[0]*rpos_beam, next_ctP.shape[1]).cuda()
        next_h01C = torch.zeros(next_h01.shape[0]*rpos_beam, next_h01.shape[1]).cuda()
        next_calpha_pastC = torch.zeros(next_calpha_past.shape[0]*rpos_beam, next_calpha_past.shape[1], next_calpha_past.shape[2]).cuda()
        for bidx in range(next_calpha_past.shape[0]):
            for ridx in range(rpos_beam):
                next_ctPC[bidx*rpos_beam+ridx] = next_ctP[bidx]
                next_h01C[bidx*rpos_beam+ridx] = next_h01[bidx]
                next_calpha_pastC[bidx*rpos_beam+ridx] = next_calpha_past[bidx]
        next_remb = torch.from_numpy(next_remb).cuda()

        next_lp, next_rep, next_state, next_h1t, next_ca, next_calpha_past, next_re = \
                    model.f_next_child(params, next_remb, next_ctPC, ctxC, next_h01C, next_calpha_pastC)

        next_lp = next_lp.cuda().numpy()
        next_state = next_state.cuda().numpy()
        next_h1t = next_h1t.cuda().numpy()
        next_calpha_past = next_calpha_past.cuda().numpy()
        next_re = next_re.cuda().numpy()

        hyp_scores = np.tile(hyp_scores[:, None], [1, rpos_beam]).flatten()
        cand_scores = hyp_scores[:, None] - np.log(next_lp+1e-10)- np.log(rpos_scores+1e-10)[:,None]
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        voc_size = next_lp.shape[1]
        trans_indices = ranks_flat // voc_size
        trans_indicesP = ranks_flat // (voc_size*rpos_beam)
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        # update paths
        new_hyp_samples = []
        new_hyp_scores = np.zeros(k-dead_k).astype('float32')
        new_hyp_rpos_samples = []
        new_hyp_relation_samples = []
        new_hyp_states = []
        new_hyp_h1ts = []
        new_hyp_calpha_past = []
        new_hyp_palpha_past = []
        new_hyp_emb_memory = []
        new_hyp_ePmb_memory = []
        
        for idx, [ti, wi, tPi] in enumerate(zip(trans_indices, word_indices, trans_indicesP)):
            new_hyp_samples.append(hyp_samples[tPi]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_rpos_samples.append(hyp_rpos_samples[tPi]+[next_rpos.flatten()[ti]])
            new_hyp_relation_samples.append(hyp_relation_samples[tPi]+[next_re[ti]])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_h1ts.append(copy.copy(next_h1t[ti]))
            new_hyp_calpha_past.append(copy.copy(next_calpha_past[ti]))
            new_hyp_palpha_past.append(copy.copy(next_palpha_past[tPi]))
            new_hyp_emb_memory.append(copy.copy(nextemb_memory[tPi]))
            new_hyp_ePmb_memory.append(copy.copy(nextePmb_memory[tPi]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_rpos_samples = []
        hyp_relation_samples = []
        hyp_states = []
        hyp_h1ts = []
        hyp_calpha_past = []
        hyp_palpha_past = []
        hyp_emb_memory = []
        hyp_ePmb_memory = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0: # <eol>
                sample_score.append(new_hyp_scores[idx])
                sample.append(new_hyp_samples[idx])
                rpos_sample.append(new_hyp_rpos_samples[idx])
                relation_sample.append(new_hyp_relation_samples[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_scores.append(new_hyp_scores[idx])
                hyp_samples.append(new_hyp_samples[idx])
                hyp_rpos_samples.append(new_hyp_rpos_samples[idx])
                hyp_relation_samples.append(new_hyp_relation_samples[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_h1ts.append(new_hyp_h1ts[idx])
                hyp_calpha_past.append(new_hyp_calpha_past[idx])
                hyp_palpha_past.append(new_hyp_palpha_past[idx])
                hyp_emb_memory.append(new_hyp_emb_memory[idx])
                hyp_ePmb_memory.append(new_hyp_ePmb_memory[idx])   
                    
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_lw = np.array([w[-1] for w in hyp_samples])  # each path's final symbol, (live_k,)
        next_state = np.array(hyp_states)  # h2t, (live_k,n)
        next_h1t = np.array(hyp_h1ts)
        next_calpha_past = np.array(hyp_calpha_past)  # (live_k,H,W)
        next_palpha_past = np.array(hyp_palpha_past)
        nextemb_memory = np.array(hyp_emb_memory)
        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2))
        nextePmb_memory = np.array(hyp_ePmb_memory)
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        next_lw = torch.from_numpy(next_lw).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_h1t = torch.from_numpy(next_h1t).cuda()
        next_calpha_past = torch.from_numpy(next_calpha_past).cuda()
        next_palpha_past = torch.from_numpy(next_palpha_past).cuda()
        nextemb_memory = torch.from_numpy(nextemb_memory).cuda()
        nextePmb_memory = torch.from_numpy(nextePmb_memory).cuda()

    return sample_score, sample, rpos_sample, relation_sample

def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30, rpos_beam=3):
    
    sample = []
    sample_score = []
    rpos_sample = []
    # rpos_sample_score = []
    relation_sample = []

    live_k = 1
    dead_k = 0  # except init, live_k = k - dead_k

    # current living paths and corresponding scores(-log)
    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)
    hyp_rpos_samples = [[]] * live_k
    hyp_relation_samples = [[]] * live_k
    # get init state, (1,n) and encoder output, (1,D,H,W)
    next_state, ctx0 = model.f_init(x)
    next_h1t = next_state
    # -1 -> My_embedding -> 0 tensor(1,m)
    next_lw = -1 * torch.ones(1, dtype=torch.int64).cuda()
    next_calpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()  # (live_k,H,W)
    next_palpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3]).cuda()
    nextemb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()
    nextePmb_memory = torch.zeros(params['maxlen'], live_k, params['m']).cuda()    

    for ii in range(maxlen):
        ctxP = ctx0.repeat(live_k, 1, 1, 1)  # (live_k,D,H,W)
        next_lpos = ii * torch.ones(live_k, dtype=torch.int64).cuda()
        next_h01, next_ma, next_ctP, next_pa, next_palpha_past, nextemb_memory, nextePmb_memory = \
                    model.f_next_parent(params, next_lw, next_lpos, ctxP, next_state, next_h1t, next_palpha_past, nextemb_memory, nextePmb_memory, ii)
        next_ma = next_ma.cuda().numpy()
        # next_ctP = next_ctP.cuda().numpy()
        next_palpha_past = next_palpha_past.cuda().numpy()
        nextemb_memory = nextemb_memory.cuda().numpy()
        nextePmb_memory = nextePmb_memory.cuda().numpy()

        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2)) # batch * Matt * dim
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        
        next_rpos = next_ma.argsort(axis=1)[:,-rpos_beam:] # topK parent index; batch * topK
        n_gaps = nextemb_memory.shape[1]
        n_batch = nextemb_memory.shape[0]
        next_rpos_gap = next_rpos + n_gaps * np.arange(n_batch)[:, None]
        next_remb_memory = nextemb_memory.reshape([n_batch*n_gaps, nextemb_memory.shape[-1]])
        next_remb = next_remb_memory[next_rpos_gap.flatten()] # [batch*rpos_beam, emb_dim]
        rpos_scores = next_ma.flatten()[next_rpos_gap.flatten()] # [batch*rpos_beam,]

        # next_ctPC = next_ctP.repeat(1, 1, rpos_beam)
        # next_ctPC = torch.reshape(next_ctPC, (-1, next_ctP.shape[1]))
        ctxC = ctx0.repeat(live_k*rpos_beam, 1, 1, 1)
        next_ctPC = torch.zeros(next_ctP.shape[0]*rpos_beam, next_ctP.shape[1]).cuda()
        next_h01C = torch.zeros(next_h01.shape[0]*rpos_beam, next_h01.shape[1]).cuda()
        next_calpha_pastC = torch.zeros(next_calpha_past.shape[0]*rpos_beam, next_calpha_past.shape[1], next_calpha_past.shape[2]).cuda()
        for bidx in range(next_calpha_past.shape[0]):
            for ridx in range(rpos_beam):
                next_ctPC[bidx*rpos_beam+ridx] = next_ctP[bidx]
                next_h01C[bidx*rpos_beam+ridx] = next_h01[bidx]
                next_calpha_pastC[bidx*rpos_beam+ridx] = next_calpha_past[bidx]
        next_remb = torch.from_numpy(next_remb).cuda()

        next_lp, next_rep, next_state, next_h1t, next_ca, next_calpha_past, next_re = \
                    model.f_next_child(params, next_remb, next_ctPC, ctxC, next_h01C, next_calpha_pastC)

        next_lp = next_lp.cuda().numpy()
        next_state = next_state.cuda().numpy()
        next_h1t = next_h1t.cuda().numpy()
        next_calpha_past = next_calpha_past.cuda().numpy()
        next_re = next_re.cuda().numpy()

        hyp_scores = np.tile(hyp_scores[:, None], [1, rpos_beam]).flatten()
        cand_scores = hyp_scores[:, None] - np.log(next_lp+1e-10)- np.log(rpos_scores+1e-10)[:,None]
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        voc_size = next_lp.shape[1]
        trans_indices = ranks_flat // voc_size
        trans_indicesP = ranks_flat // (voc_size*rpos_beam)
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        # update paths
        new_hyp_samples = []
        new_hyp_scores = np.zeros(k-dead_k).astype('float32')
        new_hyp_rpos_samples = []
        new_hyp_relation_samples = []
        new_hyp_states = []
        new_hyp_h1ts = []
        new_hyp_calpha_past = []
        new_hyp_palpha_past = []
        new_hyp_emb_memory = []
        new_hyp_ePmb_memory = []
        
        for idx, [ti, wi, tPi] in enumerate(zip(trans_indices, word_indices, trans_indicesP)):
            new_hyp_samples.append(hyp_samples[tPi]+[wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_rpos_samples.append(hyp_rpos_samples[tPi]+[next_rpos.flatten()[ti]])
            new_hyp_relation_samples.append(hyp_relation_samples[tPi]+[next_re[ti]])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_h1ts.append(copy.copy(next_h1t[ti]))
            new_hyp_calpha_past.append(copy.copy(next_calpha_past[ti]))
            new_hyp_palpha_past.append(copy.copy(next_palpha_past[tPi]))
            new_hyp_emb_memory.append(copy.copy(nextemb_memory[tPi]))
            new_hyp_ePmb_memory.append(copy.copy(nextePmb_memory[tPi]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_rpos_samples = []
        hyp_relation_samples = []
        hyp_states = []
        hyp_h1ts = []
        hyp_calpha_past = []
        hyp_palpha_past = []
        hyp_emb_memory = []
        hyp_ePmb_memory = []

        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0: # <eol>
                sample_score.append(new_hyp_scores[idx])
                sample.append(new_hyp_samples[idx])
                rpos_sample.append(new_hyp_rpos_samples[idx])
                relation_sample.append(new_hyp_relation_samples[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_scores.append(new_hyp_scores[idx])
                hyp_samples.append(new_hyp_samples[idx])
                hyp_rpos_samples.append(new_hyp_rpos_samples[idx])
                hyp_relation_samples.append(new_hyp_relation_samples[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_h1ts.append(new_hyp_h1ts[idx])
                hyp_calpha_past.append(new_hyp_calpha_past[idx])
                hyp_palpha_past.append(new_hyp_palpha_past[idx])
                hyp_emb_memory.append(new_hyp_emb_memory[idx])
                hyp_ePmb_memory.append(new_hyp_ePmb_memory[idx])   
                    
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_lw = np.array([w[-1] for w in hyp_samples])  # each path's final symbol, (live_k,)
        next_state = np.array(hyp_states)  # h2t, (live_k,n)
        next_h1t = np.array(hyp_h1ts)
        next_calpha_past = np.array(hyp_calpha_past)  # (live_k,H,W)
        next_palpha_past = np.array(hyp_palpha_past)
        nextemb_memory = np.array(hyp_emb_memory)
        nextemb_memory = np.transpose(nextemb_memory, (1, 0, 2))
        nextePmb_memory = np.array(hyp_ePmb_memory)
        nextePmb_memory = np.transpose(nextePmb_memory, (1, 0, 2))
        next_lw = torch.from_numpy(next_lw).cuda()
        next_state = torch.from_numpy(next_state).cuda()
        next_h1t = torch.from_numpy(next_h1t).cuda()
        next_calpha_past = torch.from_numpy(next_calpha_past).cuda()
        next_palpha_past = torch.from_numpy(next_palpha_past).cuda()
        nextemb_memory = torch.from_numpy(nextemb_memory).cuda()
        nextePmb_memory = torch.from_numpy(nextePmb_memory).cuda()

    return sample_score, sample, rpos_sample, relation_sample
# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

# compute metric
def cmp_result(rec,label):
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def compute_wer(rec_mat, label_mat):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        # label = list(map(int,label))
        # rec = list(map(int,rec))
        dist, llen = cmp_result(rec, label)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    wer = float(total_dist)/total_label
    sacc = float(total_line_rec)/total_line
    return wer, sacc

def cmp_sacc_result(rec_list,label_list,rec_ridx_list,label_ridx_list,rec_re_list,label_re_list,chdict,redict):
    rec = True
    out_sym_pdict = {}
    label_sym_pdict = {}
    out_sym_pdict['0'] = '<s>'
    label_sym_pdict['0'] = '<s>'
    for idx, sym in enumerate(rec_list):
        out_sym_pdict[str(idx+1)] = chdict[sym]
    for idx, sym in enumerate(label_list):
        label_sym_pdict[str(idx+1)] = chdict[sym]

    if len(rec_list) != len(label_list):
        rec = False
    else:
        for idx in range(len(rec_list)):
            out_sym = chdict[rec_list[idx]]
            label_sym = chdict[label_list[idx]]
            out_repos = int(rec_ridx_list[idx])
            label_repos = int(label_ridx_list[idx])
            out_re = redict[rec_re_list[idx]]
            label_re = redict[label_re_list[idx]]
            if out_repos in out_sym_pdict:
                out_resym_s = out_sym_pdict[out_repos]
            else:
                out_resym_s = 'unknown'
            if label_repos in label_sym_pdict:
                label_resym_s = label_sym_pdict[label_repos]
            else:
                label_resym_s = 'unknown'

            # post-processing only for math recognition
            if (out_resym_s == '\lim' and label_resym_s == '\lim') or \
            (out_resym_s == '\int' and label_resym_s == '\int') or \
            (out_resym_s == '\sum' and label_resym_s == '\sum'):
                if out_re == 'Above':
                    out_re = 'Sup'
                if out_re == 'Below':
                    out_re = 'Sub'
                if label_re == 'Above':
                    label_re = 'Sup'
                if label_re == 'Below':
                    label_re = 'Sub'

            # if out_sym != label_sym or out_pos != label_pos or out_repos != label_repos or out_re != label_re:
            # if out_sym != label_sym or out_repos != label_repos:
            if out_sym != label_sym or out_repos != label_repos or out_re != label_re:
                rec = False
                break
    return rec

def compute_sacc(rec_mat, label_mat, rec_ridx_mat, label_ridx_mat, rec_re_mat, label_re_mat, chdict, redict):
    total_num = len(rec_mat)
    correct_num = 0
    for key_rec in rec_mat:
        rec_list = rec_mat[key_rec]
        label_list = label_mat[key_rec]
        rec_ridx_list = rec_ridx_mat[key_rec]
        label_ridx_list = label_ridx_mat[key_rec]
        rec_re_list = rec_re_mat[key_rec]
        label_re_list = label_re_mat[key_rec]
        rec_result = cmp_sacc_result(rec_list,label_list,rec_ridx_list,label_ridx_list,rec_re_list,label_re_list,chdict,redict)
        if rec_result:
            correct_num += 1
    correct_rate = 1. * correct_num / total_num
    return correct_rate


def load_encoder_params(model_params, pre_params, is_tap):
    if is_tap:
        tap_pretrained_dict = {'tap_' + k: v for k, v in pre_params.items() if 'tap_' + k in model_params}
        print(tap_pretrained_dict.keys())
        # update params of tap_encoder
        model_params.update(tap_pretrained_dict)
    else:
        wap_pretrained_dict = {'wap_' + k: v for k, v in pre_params.items() if 'wap_' + k in model_params}
        print(wap_pretrained_dict.keys())
        # update params of wap_encoder
        model_params.update(wap_pretrained_dict)
    return model_params

# array([[0, 0, 1, 0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 1, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 1, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int8)

if __name__ == '__main__':
    align = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]])   
    res = random_label(align)
    print(res)
    
