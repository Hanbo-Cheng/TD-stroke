import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# DenseNet-B
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, use_dropout, pool):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)
        self.pool = pool

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        if self.pool:
            out = F.avg_pool2d(out, (1, 2), ceil_mode=True)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, reduction, bottleneck, use_dropout):
        super(DenseNet, self).__init__()
        nDenseBlocks = 3
        nChannels = 128
        self.conv1 = nn.Conv2d(9, nChannels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout, False)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout, False)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans3 = Transition(nChannels, nOutChannels, use_dropout, True)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans4 = Transition(nChannels, nOutChannels, use_dropout, False)

        nChannels = nOutChannels
        self.dense5 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        out_mask = x_mask[:, :, 0::2]
        out = self.dense4(out)
        out = self.trans4(out)
        out = self.dense5(out)
        out = F.avg_pool2d(out, (1, 2), ceil_mode=True)
        out_mask = out_mask[:, :, 0::2]
        return out, out_mask


class TAP_Encoder(nn.Module):
    def __init__(self, growthRate, reduction, bottleneck, use_dropout, n_feature, hidden_size):
        super(TAP_Encoder, self).__init__()
        self.densenet = DenseNet(growthRate, reduction, bottleneck, use_dropout)
        self.gru0 = nn.GRU(n_feature, hidden_size, bidirectional=True)
        # self.ln0 = nn.LayerNorm(512)
        # self.gru_r0 = nn.GRU(n_feature, hidden_size)
        self.gru1 = nn.GRU(512, hidden_size, bidirectional=True)
        # self.ln1 = nn.LayerNorm(512)
        # self.gru_r1 = nn.GRU(512, hidden_size)
        self.change_dim = nn.Linear(512, 500, bias=False)

    def forward(self, x, x_mask, stroke_mask):
        ctx, ctx_mask = self.densenet(x, x_mask)
        shapes = ctx.shape
        ctx = ctx.permute(3, 0, 1, 2).view(shapes[3], shapes[0], shapes[1])
        ctx_mask = ctx_mask.permute(2, 0, 1).view(shapes[3], shapes[0])

        h = torch.nn.utils.rnn.pack_padded_sequence(ctx, lengths=list(ctx_mask.sum(0).int()), enforce_sorted=False)
        proj, proj_n = self.gru0(h)
        proj = torch.nn.utils.rnn.pad_packed_sequence(proj, padding_value=0.0)[0]
        # proj = self.ln0(proj)
        proj = torch.nn.utils.rnn.pack_padded_sequence(proj, lengths=list(ctx_mask.sum(0).int()), enforce_sorted=False)
        sequence, sequence_n = self.gru1(proj)
        h = torch.nn.utils.rnn.pad_packed_sequence(sequence, padding_value=0.0)[0]
        # h = self.ln1(h)
        h = self.change_dim(h)
        h = h * ctx_mask[:, :, None]
        n_strokes = [m.shape[0] for m in stroke_mask]
        max_num_stroke = np.max(n_strokes)
        out_feats = torch.zeros((max_num_stroke, h.shape[1], h.shape[2])).cuda()  # (max_num_strokes,batch,D)
        out_feats_mask = torch.zeros((max_num_stroke, h.shape[1])).cuda()  # (max_num_strokes,batch)

        for idx in range(h.shape[1]):
            f = h[:, idx, :]
            stroke_m = stroke_mask[idx][:, None, None, :]
            for j in range(2):
                stroke_m = F.max_pool2d(stroke_m, (1, 2), ceil_mode=True)  # (n_strokes,1,1,L)
            stroke_m = stroke_m.permute(0, 3, 1, 2).view(-1, f.shape[0], 1)  # (n_strokes,L,1)
            feats = f[None, :, :] * stroke_m  # # (n_strokes,L,D)
            feat = feats.sum(1) / stroke_m.sum(1)  # (n_strokes, D)
            out_feats[:n_strokes[idx], idx, :] = feat
            out_feats_mask[:n_strokes[idx], idx] = 1.

        return out_feats, out_feats_mask
