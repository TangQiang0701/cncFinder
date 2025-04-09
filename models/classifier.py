# -- coding: utf-8 --
# author : TangQiang
# time   : 2025/3/15
# email  : tangqiang.0701@gmail.com
# file   : classifier.py

import dgl
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn as dglnn

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_heads, device, drop):
        super(Classifier, self).__init__()
        self.n_heads = n_heads
        self.conv1 = dglnn.GraphConv(in_feats=in_dim, out_feats=hidden_dim, norm='none',
                                     allow_zero_in_degree=True).to(device)
        self.conv2 = dglnn.GraphConv(in_feats=hidden_dim, out_feats=hidden_dim, norm='none',
                                     allow_zero_in_degree=True).to(device)
        self.conv3 = dglnn.GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=n_heads, allow_zero_in_degree=True,
                                   feat_drop=drop, attn_drop=drop).to(device)
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Linear(128, 2),
        ).to(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, g, h):
        self.h1 = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
        self.h2 = F.relu(self.conv2(g, self.h1, edge_weight=g.edata['weight']))
        h, self.att1 = self.conv3(g, self.h2, edge_weight=g.edata['weight'], get_attention=True)
        self.h = F.relu(h.sum(dim=1) / self.n_heads)
        with g.local_scope():
            g.ndata['h'] = self.h
            hg = dgl.sum_nodes(g, 'h')
            logists = self.classifier(hg)
            return F.softmax(logists)

    def get_weights(self):
        return self.h1, self.h2,self.h, self.att1
