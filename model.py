import torch
import torch.nn as nn
import torch.nn.functional as F

# PBFL 官方 WordCNN+Attention
class WordCNNAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Conv1d(embed_dim, hidden_dim, 3, padding=1)
        self.att = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        att = F.softmax(self.att(x), dim=1)
        return (x * att).sum(1)

# APH 官方偏置模块
class APHBias(nn.Module):
    def __init__(self, user_num, item_num):
        super().__init__()
        self.global_mean = nn.Parameter(torch.tensor(3.0))
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)

    def forward(self, u, i):
        bu = self.user_bias(u).squeeze(-1)
        bi = self.item_bias(i).squeeze(-1)
        return self.global_mean + bu + bi

# 最终模型：PBFL + APH + 讽刺修正 ŷ = r̂ + ĉ·s
class FinalModel(nn.Module):
    def __init__(self, user_num, item_num, vocab_size, embed_dim=64):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.user_emb = nn.Embedding(user_num, embed_dim)
        self.item_emb = nn.Embedding(item_num, embed_dim)

        self.encoder = WordCNNAttention(embed_dim, embed_dim)
        self.interact = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU()
        )

        self.r_pred = nn.Linear(embed_dim, 1)
        self.mlp_s = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.c_pred = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

        self.aph = APHBias(user_num, item_num)

    def forward(self, u, i, rev):
        emb = self.word_emb(rev)
        feat = self.encoder(emb)
        ue = self.user_emb(u)
        ie = self.item_emb(i)

        e = self.interact(torch.cat([ue, ie, feat], dim=-1))
        r = self.r_pred(e).squeeze(-1)
        s = self.mlp_s(e).squeeze(-1)
        c = self.c_pred(feat).squeeze(-1)

        y = r + c * s
        return self.aph(u, i) + y
