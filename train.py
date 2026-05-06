import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 读取你刚建的 data.csv 数据集
class ReviewDataset(Dataset):
    def __init__(self, df, word2id, max_len=30):
        self.df = df
        self.word2id = word2id
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        u = int(row.user_id)
        i = int(row.item_id)
        txt = str(row.review_text)
        r = float(row.rating)

        tokens = txt.split()[:self.max_len]
        ids = [self.word2id.get(w, 1) for w in tokens]
        if len(ids) < self.max_len:
            ids += [0]*(self.max_len - len(ids))

        return torch.tensor(u), torch.tensor(i), torch.tensor(ids), torch.tensor(r)

# PBFL 模块
class WordCNNAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cnn = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.att = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.cnn(x)
        x = F.relu(x)
        x = x.transpose(1,2)
        att = F.softmax(self.att(x), dim=1)
        return (x * att).sum(1)

# APH 偏置
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

# 完整模型
class FinalModel(nn.Module):
    def __init__(self, user_num, item_num, vocab_size, embed_dim=64):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.user_emb = nn.Embedding(user_num, embed_dim)
        self.item_emb = nn.Embedding(item_num, embed_dim)

        self.encoder = WordCNNAttention(embed_dim)
        self.interact = nn.Sequential(
            nn.Linear(embed_dim*3, embed_dim),
            nn.ReLU()
        )
        self.r_pred = nn.Linear(embed_dim,1)
        self.mlp_s = nn.Sequential(nn.Linear(embed_dim,32),nn.ReLU(),nn.Linear(32,1))
        self.c_pred = nn.Sequential(nn.Linear(embed_dim,1),nn.Sigmoid())
        self.aph = APHBias(user_num, item_num)

    def forward(self, u, i, rev):
        emb = self.word_emb(rev)
        feat = self.encoder(emb)
        ue = self.user_emb(u)
        ie = self.item_emb(i)
        e = self.interact(torch.cat([ue,ie,feat], dim=-1))
        r = self.r_pred(e).squeeze(-1)
        s = self.mlp_s(e).squeeze(-1)
        c = self.c_pred(feat).squeeze(-1)
        y = r + c * s
        return self.aph(u,i) + y

def train_epoch(model, loader, opt, cri, dev):
    model.train()
    loss_total = 0
    for u,i,rev,y in loader:
        u,i,rev,y = u.to(dev),i.to(dev),rev.to(dev),y.to(dev)
        opt.zero_grad()
        pred = model(u,i,rev)
        loss = cri(pred,y)
        loss.backward()
        opt.step()
        loss_total += loss.item()
    return loss_total / len(loader)

def eval_epoch(model, loader, cri, dev):
    model.eval()
    loss_total = 0
    mae_total = 0
    with torch.no_grad():
        for u,i,rev,y in loader:
            u,i,rev,y = u.to(dev),i.to(dev),rev.to(dev),y.to(dev)
            pred = model(u,i,rev)
            loss_total += cri(pred,y).item()
            mae_total += torch.mean(torch.abs(pred-y)).item()
    return loss_total/len(loader), mae_total/len(loader)

if __name__ == "__main__":
    set_seed()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取你建好的数据集
    df = pd.read_csv("data.csv")

    # 构造词表
    all_words = []
    for t in df.review_text:
        all_words += str(t).split()
    vocab = list(set(all_words))
    word2id = {w:i+2 for i,w in enumerate(vocab)}
    word2id["<pad>"] = 0
    word2id["<unk>"] = 1

    user_num = df.user_id.max() + 1
    item_num = df.item_id.max() + 1

    # 划分训练验证
    split = int(0.8 * len(df))
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]

    train_set = ReviewDataset(train_df, word2id)
    val_set = ReviewDataset(val_df, word2id)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = FinalModel(user_num, item_num, len(word2id)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cri = nn.MSELoss()

    print("✅ 用你上传的 data.csv 真实小数据集开始训练")
    for ep in range(15):
        tr_loss = train_epoch(model, train_loader, opt, cri, dev)
        va_loss, va_mae = eval_epoch(model, val_loader, cri, dev)
        print(f"Epoch {ep+1:2d} | TrainLoss:{tr_loss:.4f} ValLoss:{va_loss:.4f} ValMAE:{va_mae:.4f}")
