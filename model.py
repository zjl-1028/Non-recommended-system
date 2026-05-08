import torch
import torch.nn as nn
import torch.nn.functional as F

class WordLevelConvAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cnn = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.att = nn.Linear(dim, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        att_weight = torch.softmax(self.att(x), dim=1)
        x = torch.sum(x * att_weight, dim=1)
        return x

class FinalModel(nn.Module):
    def __init__(self, user_num, item_num, vocab_size, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num, dim)
        self.word_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.encoder = WordLevelConvAttention(dim)
        self.sarc_cls = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2)
        )
        self.interact = nn.Sequential(
            nn.Linear(dim * 4 + 2, dim),
            nn.ReLU()
        )
        self.r_pred = nn.Linear(dim, 1)
        self.c_pred = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.s_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, u, i, user_rev, item_rev):
        u_emb = self.user_emb(u)
        i_emb = self.item_emb(i)
        u_rev_emb = self.word_emb(user_rev)
        i_rev_emb = self.word_emb(item_rev)

        u_feat = self.encoder(u_rev_emb)
        i_feat = self.encoder(i_rev_emb)

        sarc_logits = self.sarc_cls(u_feat)
        sarc_prob = torch.softmax(sarc_logits, dim=-1)

        cat_feat = torch.cat([u_emb, i_emb, u_feat, i_feat, sarc_prob], -1)
        fusion = self.interact(cat_feat)

        r = self.r_pred(fusion).squeeze(-1)
        c = self.c_pred(fusion).squeeze(-1)
        s = self.s_mlp(fusion).squeeze(-1)

        y = r + c * s
        return y, sarc_logits
