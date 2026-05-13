import torch
import torch.nn as nn
import torch.nn.functional as F

class CSHNetFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, out_dim, dropout=0.2):
        super().__init__()
        self.conv2 = nn.Conv1d(embed_dim, out_dim, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, out_dim, kernel_size=4, padding=2)

        self.bilstm = nn.LSTM(input_size=out_dim * 3, hidden_size=out_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.mha = nn.MultiheadAttention(embed_dim=out_dim * 2, num_heads=4, batch_first=True, dropout=dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))
        c4 = F.relu(self.conv4(x))

        c_cat = torch.cat([c2, c3, c4], dim=1)
        c_cat = c_cat.transpose(1, 2)

        lstm_out, _ = self.bilstm(c_cat)
        att_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        feat = torch.mean(att_out, dim=1)
        return feat

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
        self.csh_net = CSHNetFeatureExtractor(dim, dim)
        
        self.sarc_cls = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.interact = nn.Sequential(nn.Linear(dim * 4 + 2, dim), nn.ReLU())
        
        self.r_pred = nn.Linear(dim, 1)
        self.c_pred = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.s_pred = nn.Linear(dim, 1)

    def forward(self, u, i, user_rev, item_rev):
        u_emb = self.user_emb(u)
        i_emb = self.item_emb(i)
        u_rev_emb = self.word_emb(user_rev)
        i_rev_emb = self.word_emb(item_rev)

        u_feat = self.encoder(u_rev_emb)
        i_feat = self.encoder(i_rev_emb)
        sarc_feat = self.csh_net(u_rev_emb)

        sarc_logits = self.sarc_cls(sarc_feat)
        sarc_prob = torch.softmax(sarc_logits, dim=-1)

        cat_feat = torch.cat([u_emb, i_emb, u_feat, i_feat, sarc_prob], -1)
        fusion = self.interact(cat_feat)

        r = self.r_pred(fusion).squeeze(-1)
        c = self.c_pred(fusion).squeeze(-1)
        s = self.s_pred(fusion).squeeze(-1)

        y = r + c * s
        return y, sarc_logits
