import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意力CNN编码器
class WordCNNAttention(nn.Module):
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

# PBFL 主模型
class FinalModel(nn.Module):
    def __init__(self, user_num, item_num, vocab_size, dim=64):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num, dim)
        self.encoder = WordCNNAttention(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, u, i, t):
        te = self.word_emb(t)
        tf = self.encoder(te)
        uf = self.user_emb(u)
        i_f = self.item_emb(i)
        return self.fc(torch.cat([uf, i_f, tf], dim=-1)).squeeze(-1)
