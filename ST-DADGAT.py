import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
path = 'C:/Users/hbu/Desktop/seed_channel.xlsx'


def normalize_A(A, symmetry=False):

    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Selfattention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size, channels, height = inputs.shape
        inputs = torch.flatten(inputs, 1, 2)
        q = self.query(inputs).view(batch_size, channels, -1)
        k = self.key(inputs).view(batch_size, channels, -1).permute(0, 2, 1)
        dim_k = q.size(2)
        dim_k = torch.tensor(dim_k)
        v = self.value(inputs).view(batch_size, channels, height)
        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix/torch.sqrt(dim_k))
        out = torch.bmm(v.permute(0, 2, 1), attn_matrix).permute(0, 2, 1)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads,  mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attentions = [Selfattention(in_channels, out_channels) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.PN1 = nn.LayerNorm(5)
        self.PN2 = nn.LayerNorm(5)
        self.ffn = FeedForward(in_channels, mlp_dim, dropout)
        self.linear1 = nn.Linear(in_channels * n_heads, in_channels)
        self.ch = in_channels
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.PN1(x)
        x2 = torch.cat([att(x1) for att in self.attentions], dim=2)
        x2 = self.dropout(x2)
        x20 = torch.flatten(x2, 1, 2)
        x21 = self.linear1(x20).view(-1, 62, 5) + x
        x3 = self.PN2(x21)
        x31 = torch.flatten(x3, 1, 2)
        x4 = self.ffn(x31).view(-1, 62, 5) + x21
        return x4


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.m = 0.8

    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)

        a_input1 = self.leakyrelu(a_input)
        e = torch.matmul(a_input1, self.a).squeeze(3)
        dj = self.m * adj + (1 - self.m) * e
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(dj > 0, dj, zero_vec)
        attention1 = F.softmax(attention, dim=1)
        attention = F.dropout(attention1, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.relu(h_prime)
        else:
            return F.relu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)
        self.lam = 0.5
        self.linear = nn.Linear(62*n_hid*n_heads, 62*n_hid)

    def forward(self, x, adj):
        batch_size, channel, f = x.shape
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        out_att = self.out_att(x, adj)
        x = F.elu(out_att)
        return F.log_softmax(x, dim=2)


def calculate_adjacency_matrix(files_path):
    df = pd.read_excel(files_path, header=None)
    coordinates = df.iloc[1:63, 1:4].values
    m = 0.2
    distances = distance_matrix(coordinates, coordinates)
    adjacency_matrix = np.exp(-distances**2 / (2 * m**2))
    return adjacency_matrix


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=62, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class conv3(nn.Module):
    def __init__(self):
        super(conv3, self).__init__()
        self.cnn1 = nn.Conv1d(62, 62, 3)
        self.cnn2 = nn.Conv1d(62, 62, 3)
        self.cnn3 = nn.Conv1d(62, 62, 3)

        self.cnn4 = nn.Conv1d(62, 62, 5)
        self.cnn5 = nn.Conv1d(62, 62, 5)

        self.cnn6 = nn.Conv1d(62, 62, 3)
        self.cnn7 = nn.Conv1d(62, 62, 5)

        self.bn1 = nn.BatchNorm1d(62)
        self.bn2 = nn.BatchNorm1d(62)
        self.bn3 = nn.BatchNorm1d(62)
        self.bn4 = nn.BatchNorm1d(62)
        self.bn5 = nn.BatchNorm1d(62)
        self.bn6 = nn.BatchNorm1d(62)
        self.bn7 = nn.BatchNorm1d(62)
        self.sig = nn.SELU()
        self.att1 = ChannelAttention()
        self.att2 = ChannelAttention()

    def forward(self, x):
        x1 = self.sig(self.bn1(self.cnn1(x)))
        x2 = torch.cat([x1, x], 2)
        x2 = self.att1(x2)

        x3 = self.sig(self.bn2(self.cnn2(x2)))
        x4 = torch.cat([x2, x3], 2)
        x5 = self.sig(self.bn3(self.cnn3(x4)))

        x6 = self.sig(self.bn4(self.cnn4(x2)))
        x7 = torch.cat([x6, x2], 2)
        x8 = self.sig(self.bn5(self.cnn5(x7)))

        x9 = self.sig(self.bn6(self.cnn6(x2)))
        x10 = torch.cat([x9, x2], 2)
        x11 = self.sig(self.bn7(self.cnn7(x10)))

        x12 = torch.cat([x5, x8, x11], 2)
        x12 = self.att2(x12)
        return x12


class models3(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropoutg, alpha, gn_heads):
        super(models3, self).__init__()
        self.gat = GAT(n_feat, n_hid, n_class, dropoutg, alpha, gn_heads)
        self.fc1 = nn.Linear(620, 256)      #  620
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 4)
        self.dropout = nn.Dropout(0.1)
        self.A = torch.tensor(calculate_adjacency_matrix(path), dtype=torch.float32)
        self.conv = conv3()
        self.transformer = TransformerBlock(310, 124, 6, 186, 0.3)

    def forward(self, x):
        L = normalize_A(self.A.cuda())
        xx1 = self.conv(x)
        xt = self.transformer(x)
        xx = torch.cat([xx1, xt], 2)
        x1 = self.gat(xx, L)
        x11 = torch.flatten(x1, 1, 2)
        x2 = self.dropout((self.dropout(self.fc1(x11))))
        x3 = self.dropout((self.dropout(self.fc2(x2))))
        x4 = self.dropout((self.dropout(self.fc3(x3))))

        return x11, x4



