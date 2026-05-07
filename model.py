import torch
import torch.nn as nn
import torch.nn.functional as F

class WordLevelConvAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cnn = nn.Conv1d(dim, dim, 3, padding=1)
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        att = F.softmax(self.att(x), dim=1)
        return (x * att).sum(1)

class FinalModel(nn.Module):
    def __init__(self, user_num, item_num, vocab_size, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num, dim)
        self.word_emb = nn.Embedding(vocab_size, dim, padding_idx=0)

        self.review_encoder = WordLevelConvAttention(dim)
        self.interact = nn.Sequential(nn.Linear(dim*4, dim), nn.ReLU())

        self.r_pred = nn.Linear(dim, 1)
        self.c_pred = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.s_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, u, i, user_rev, item_rev):
        u_emb = self.user_emb(u)
        i_emb = self.item_emb(i)

        u_rev_emb = self.word_emb(user_rev)
        i_rev_emb = self.word_emb(item_rev)

        u_rev_feat = self.review_encoder(u_rev_emb)
        i_rev_feat = self.review_encoder(i_rev_emb)

        feat = self.interact(torch.cat([u_emb, i_emb, u_rev_feat, i_rev_feat], -1))
        r = self.r_pred(feat).squeeze(-1)
        c = self.c_pred(u_rev_feat).squeeze(-1)
        s = self.s_mlp(u_rev_feat).squeeze(-1)

        y = r + c * s
        return y
