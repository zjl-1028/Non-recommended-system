import torch
import torch.nn as nn
import torch.nn.functional as F

class CSHNetFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, out_dim, max_len=30, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.conv2 = nn.Conv1d(embed_dim, out_dim, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, out_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(embed_dim, out_dim, kernel_size=4, padding=2)
        self.bilstm = nn.LSTM(input_size=out_dim * 3, hidden_size=out_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.mha = nn.MultiheadAttention(embed_dim=out_dim * 2, num_heads=4, batch_first=True, dropout=dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        c2 = F.relu(self.conv2(x))[:, :, :self.max_len]
        c3 = F.relu(self.conv3(x))[:, :, :self.max_len]
        c4 = F.relu(self.conv4(x))[:, :, :self.max_len]
        c_cat = torch.cat([c2, c3, c4], dim=1)
        c_cat = c_cat.transpose(1, 2)
        lstm_out, _ = self.bilstm(c_cat)
        att_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        feat = torch.mean(att_out, dim=1)
        return feat

class PBFLSarcModel(nn.Module):
    def __init__(self, user_num, item_num, vocab_size, dim=64, max_len=30):
        super().__init__()
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num, dim)
        self.word_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.feature_net = CSHNetFeatureExtractor(dim, dim, max_len)
        self.comment_type_layer = nn.Linear(dim * 2, dim)
        self.s_layer = nn.Linear(dim, 1)
        self.c_layer = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, user_id, item_id, review):
        u_emb = self.user_emb(user_id)
        i_emb = self.item_emb(item_id)
        w_emb = self.word_emb(review)
        text_feat = self.feature_net(w_emb)
        type_feat = F.relu(self.comment_type_layer(text_feat))
        s = self.s_layer(type_feat).squeeze(-1)
        c = self.c_layer(type_feat).squeeze(-1)
        r = (u_emb * i_emb).sum(-1)
        y = r + c * s
        return y, r, s, c
