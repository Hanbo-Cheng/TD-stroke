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
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
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
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class WAP_Encoder(nn.Module):
    def __init__(self, growthRate, reduction, bottleneck, use_dropout):
        super(WAP_Encoder, self).__init__()
        nDenseBlocks = 16
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=7, padding=3, stride=2, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        self.conv_change_dim = nn.Conv2d(684, 500, 1, bias=False)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def get_stroke_feature(self, out, stroke_mask, pooling_time):
        n_strokes = [m.shape[0] for m in stroke_mask]
        max_num_stroke = np.max(n_strokes)
        out_feats = torch.zeros((max_num_stroke, out.shape[0], out.shape[1])).cuda()  # (max_num_strokes,batch,D)
        out_feats_mask = torch.zeros((max_num_stroke, out.shape[0])).cuda()  # (max_num_strokes,batch)
        for idx, f in enumerate(out):
            stroke_m = stroke_mask[idx][:, None, :, :]
            for j in range(pooling_time):
                stroke_m = F.max_pool2d(stroke_m, 2, ceil_mode=True)  # (n_strokes,1,H,W)
            feats = f[None, :, :, :] * stroke_m
            feat = feats.sum(3).sum(2) / stroke_m.sum(3).sum(2)  # (n_strokes, D)
            out_feats[:n_strokes[idx], idx, :] = feat
            out_feats_mask[:n_strokes[idx], idx] = 1.
        return out_feats, out_feats_mask

    def forward(self, x, x_mask, stroke_mask):
        out = self.conv1(x)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        out = self.conv_change_dim(out)
        n_strokes = [m.shape[0] for m in stroke_mask]
        max_num_stroke = np.max(n_strokes)
        out_feats = torch.zeros((max_num_stroke, out.shape[0], out.shape[1])).cuda()  # (max_num_strokes,batch,D)
        out_feats_mask = torch.zeros((max_num_stroke, out.shape[0])).cuda()  # (max_num_strokes,batch)
        for idx, f in enumerate(out * out_mask[:, None, :, :]):
            stroke_m = stroke_mask[idx][:, None, :, :]
            for j in range(4):
                stroke_m = F.max_pool2d(stroke_m, 2, ceil_mode=True)  # (n_strokes,1,H,W)
            feats = f[None, :, :, :] * stroke_m
            feat = feats.sum(3).sum(2) / stroke_m.sum(3).sum(2)  # (n_strokes, D)
            out_feats[:n_strokes[idx], idx, :] = feat
            out_feats_mask[:n_strokes[idx], idx] = 1.
        return out_feats, out_feats_mask
