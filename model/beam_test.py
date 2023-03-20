import torch
import torch.nn.functional as F
import copy


def beam_test(models, x, x_mask, max_len, p_y, p_re, p_mask, beam_sise):

    # init state
    number_models = len(models)
    ctx_s = []
    ctx_s_pool = []
    ht_s = []
    ctx_key_object_s = []
    ctx_key_relation_s = []
    attention_past_s = []

    for model in models:
        ctx, ctx_mask = model.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) \
                    / ctx_mask.sum(2).sum(1)[:, None]         # (batch,D)
        ht = torch.tanh(model.init_context(ctx_mean))
        ctx_key_object = model.decoder.conv_key_object(ctx).permute(0, 2, 3, 1)
        ctx_pool = F.avg_pool2d(ctx, (2,2), stride=(2,2), ceil_mode=True)
        ctx_mask_pool = ctx_mask[:, 0::2, 0::2]
        ctx_key_relation = model.decoder.conv_key_relation(ctx_pool).permute(0, 2, 3, 1)
        B, H, W = ctx_mask.shape
        attention_past = torch.zeros(B, 1, H, W).cuda()
        
        ctx_s.append(ctx)
        ctx_s_pool.append(ctx_pool)
        ht_s.append(ht)
        ctx_key_object_s.append(ctx_key_object)
        ctx_key_relation_s.append(ctx_key_relation)
        attention_past_s.append(attention_past)

    # decoding
    ctx_input_s = []
    ctx_input_s_pool = []
    ctx_key_object_input_s = []
    ctx_key_relation_input_s = []
    ctx_mask_input_s = []
    ctx_mask_input_s_pool = []
    ht_input_s = []

    for mi in range(number_models):
        ctx_input = ctx_s[mi].repeat(1, 1, 1, 1)
        ctx_input_pool = ctx_s_pool[mi].repeat(1, 1, 1, 1)
        ctx_key_object_input = ctx_key_object_s[mi].repeat(1, 1, 1, 1)
        ctx_key_relation_input = ctx_key_relation_s[mi].repeat(1, 1, 1, 1)
        ctx_mask_input = ctx_mask.repeat(1, 1, 1)
        ctx_mask_input_pool = ctx_mask_pool.repeat(1, 1, 1)
        ht_input = ht_s[mi]

        ctx_input_s.append(ctx_input)
        ctx_input_s_pool.append(ctx_input_pool)
        ctx_key_object_input_s.append(ctx_key_object_input)
        ctx_key_relation_input_s.append(ctx_key_relation_input)
        ctx_mask_input_s.append(ctx_mask_input)
        ctx_mask_input_s_pool.append(ctx_mask_input_pool)
        ht_input_s.append(ht_input)

    y_input = p_y
    p_mask_input = p_mask.repeat(1)
    re_input = p_re

    # all we want
    labels = torch.zeros(beam_sise, max_len+1, dtype=torch.long).cuda()
    relation_tables = torch.zeros(beam_sise, max_len+1, 9, dtype=torch.long).cuda()
    len_labels = torch.zeros(beam_sise).cuda()
    logs = torch.zeros(beam_sise).cuda()
    labels_stack = [[] for bi in range(beam_sise)]
    relation_stack = [[] for bi in range(beam_sise)]
    # attention_list = [[] for i in range(beam_sise)]
    
    
    N = beam_sise
    end_N = 0
    for ei in range(max_len):
        score_s = []
        ct_s = []
        ht_s = []
        for mi in range(number_models):
            model = models[mi]
            ctx_input = ctx_input_s[mi]
            ctx_key_object_input = ctx_key_object_input_s[mi]
            ctx_mask_input = ctx_mask_input_s[mi]
            attention_past = attention_past_s[mi]
            ht_input = ht_input_s[mi]

            score, ct, ht, attention = model.decoder.get_child(ctx_input, 
                ctx_key_object_input, ctx_mask_input, 
                attention_past, y_input, p_mask_input, re_input, ht_input)
            attention_past = attention[:, None, :, :] + attention_past

            score_s.append(score)
            ct_s.append(ct)
            ht_s.append(ht)
            attention_past_s[mi] = attention_past

  
        # 复制当前没结束的前N项
        t_labels = copy.deepcopy(labels[:N, :])
        t_relation_tables = copy.deepcopy(relation_tables[:N, :, :])
        t_len_labs = copy.deepcopy(len_labels[:N])
        t_logs = copy.deepcopy(logs[:N])
        t_labels_stack = copy.deepcopy(labels_stack[:N])
        t_relation_stack = copy.deepcopy(relation_stack[:N])
        # t_alpha_list = copy.deepcopy(attention_list)

        #创建下次循环需要的变量：
        ys = torch.LongTensor(N).cuda()
        res = torch.LongTensor(N).cuda()
        hts_s = []
        atts_s = []
        for mi in range(number_models):
            hts = torch.Tensor(N, ht.shape[1]).cuda()
            atts = torch.Tensor(N, 1, H, W).cuda()

            hts_s.append(hts)
            atts_s.append(atts)

        # 计算出此次综合概率前N项
        mean_score = torch.zeros_like(score_s[0]).cuda()
        for score in score_s:
            mean_score = mean_score + F.softmax(score, 1)
        mean_score = mean_score / number_models
        # mean_score = F.softmax(score_s[0], 1)

        log_prob_y = torch.log(mean_score) #(N,K)
        max_logs, max_ys = torch.topk(log_prob_y, N, 1) #(N,N) (N,N)
        if ei == 0:
            t_all_logs = max_logs
        else:
            t_all_logs = max_logs + t_logs[:,None] #(N,N)
        t_logs, t_max_indexs = torch.topk(t_all_logs.view(-1), N) #(N.) (N.)
        
        # 得到此次log最大的前N项的predict
        t_ys = torch.LongTensor(N).cuda()
        t_ct_s = []
        t_ht_s = []
        for mi in range(number_models):
            t_ct = torch.zeros(N, ct.shape[1]).cuda()
            t_ht = torch.zeros(N, ht.shape[1]).cuda()
            t_ct_s.append(t_ct)
            t_ht_s.append(t_ht)

        column_len = N
        for yi in range(column_len):
            index = t_max_indexs[yi].item()
            row = int(index/column_len)
            column = index%column_len
            t_ys[yi] = max_ys[row][column].item()
            for mi in range(number_models):
                t_ct_s[mi][yi] = ct_s[mi][row]
                t_ht_s[mi][yi] = ht_s[mi][row]

        predict_relation_s = []
        for mi in range(number_models):
            model = models[mi]
            ctx_input_pool = ctx_input_s_pool[mi]
            ctx_key_relation_input = ctx_key_relation_input_s[mi]
            ctx_mask_input_pool = ctx_mask_input_s_pool[mi]
            attention_past = attention_past_s[mi]
            t_ct = t_ct_s[mi]
            t_ht = t_ht_s[mi]
            predict_relation = model.decoder.get_relation(ctx_input_pool, 
                ctx_key_relation_input, ctx_mask_input_pool,
                t_ct, t_ys, t_ht)
            predict_relation_s.append(predict_relation)
        
        sum_predict_relation = torch.zeros_like(predict_relation_s[0]).cuda()
        
        for mi in range(number_models):
            sum_predict_relation = sum_predict_relation + torch.sigmoid(predict_relation_s[mi])
        sum_predict_relation =  ( (sum_predict_relation / number_models) > 0.5 )
        # for mi in range(number_models):
        #     sum_predict_relation = sum_predict_relation + predict_relation_s[mi]
        # sum_predict_relation =   (sum_predict_relation > 0 )
        # sum_predict_relation = (predict_relation_s[0] > 0)
        # t_predict_relation = copy.deepcopy(predict_relation)

        t_end = 0 #本次label终止的个数
        column_len = N
        for yi in range(column_len):
            index = t_max_indexs[yi].item()
            row = int(index/column_len)
            column = index%column_len
            t_y = t_ys[yi]

            tt_relation_stack = copy.deepcopy(t_relation_stack[row])
            tt_label_stack = copy.deepcopy(t_labels_stack[row])
            tt_relation_stack.append(copy.deepcopy(sum_predict_relation[row]))
            tt_label_stack.append(t_y)
            t_p_re, t_p = models[0].find_parent(tt_relation_stack, tt_label_stack)
            
            if t_p_re == 8:
                end_N += 1
                N = beam_sise - end_N
                logs[N] = t_logs[yi]
                labels[N,:] = t_labels[row,:]
                labels[N, ei] = t_y
                labels[N, ei+1] = 0
                len_labels[N] = t_len_labs[row] + 1
                relation_tables[N] = t_relation_tables[row]
                relation_tables[N, ei] = sum_predict_relation[row]
                # attention_list[N] = t_alpha_list[row]
                t_end += 1
            else:
                #print(t_y)
                ni = yi - t_end
                labels[ni,:] = t_labels[row,:]
                labels[ni, ei] = t_y
                #print(ni, ei, labels[:,ei])
                # attention_list[ni] = t_alpha_list[row].copy()
                # attention_list[ni].append(alpha_np[row])
                len_labels[ni] = t_len_labs[row] + 1 
                logs[ni] = t_logs[yi]
                relation_tables[ni] = t_relation_tables[row]
                relation_tables[ni, ei] = sum_predict_relation[row]
                relation_stack[ni] = tt_relation_stack
                labels_stack[ni] = tt_label_stack

                #继续跟新下个输入。
                res[ni], ys[ni] = t_p_re, t_p
                for mi in range(number_models):
                    hts_s[mi][ni, :] = ht_s[mi][row,:]
                    atts_s[mi][ni, :] = attention_past_s[mi][row, :]

                #print(labels.cuda().numpy())
        # print(labels.cuda().numpy())
        # print(torch.exp(logs).cuda().numpy())
        # print(len_labels.cuda().numpy())
        # print(end_N)
        # N = B - end_N
        if N < 1:
            break

        #如果还没有结束，更新下个循环需要输入的变量。
        y_input = ys[:N]
        re_input = res[:N]
        p_mask_input = p_mask.repeat(N)
        for mi in range(number_models):
            ht_input_s[mi] = hts_s[mi][:N, :]
            attention_past_s[mi] = atts_s[mi][:N, :]
            ctx_input_s[mi] = ctx_s[mi].repeat(N, 1, 1, 1)
            ctx_input_s_pool[mi] = ctx_s_pool[mi].repeat(N, 1, 1, 1)
            ctx_key_object_input_s[mi] = ctx_key_object_s[mi].repeat(N, 1, 1, 1)
            ctx_key_relation_input_s[mi] = ctx_key_relation_s[mi].repeat(N, 1, 1, 1)
            ctx_mask_input_s[mi] = ctx_mask.repeat(N, 1, 1)
            ctx_mask_input_s_pool[mi] = ctx_mask_pool.repeat(N, 1, 1)


    logs = logs / len_labels
    _, index = torch.max(logs.unsqueeze(0),1)
    
    predict_object = labels[index[0],:].cuda().numpy()
    predict_relation = relation_tables[index[0]].cuda().numpy()
    len_predict = int(len_labels[index[0]].cuda().item())
    # print(labels.cuda().numpy())
    # print(logs.cuda().numpy())
    # print(len_labels.cuda().numpy())
    return predict_object[:len_predict+1], predict_relation[:len_predict+1]