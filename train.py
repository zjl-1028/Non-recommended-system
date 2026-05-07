import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_amazon_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append([d['reviewerID'], d['asin'], d.get('reviewText', ''), d['overall']])
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'review_text', 'rating'])
    df['user_id'] = pd.factorize(df['user_id'])[0]
    df['item_id'] = pd.factorize(df['item_id'])[0]
    return df

class PBFLDataset(Dataset):
    def __init__(self, df, word2id, max_len=150):
        self.df = df
        self.word2id = word2id
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = str(row.review_text).split()[:self.max_len]
        ids = [self.word2id.get(w, 1) for w in tokens] + [0]*(self.max_len - len(tokens))
        return (
            torch.tensor(row.user_id),
            torch.tensor(row.item_id),
            torch.tensor(ids),
            torch.tensor(float(row.rating), dtype=torch.float32)
        )

class WordCNNAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cnn = nn.Conv1d(dim, dim, 3, padding=1)
        self.att = nn.Linear(dim, 1)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.cnn(x)
        x = F.relu(x)
        x = x.transpose(1,2)
        att = F.softmax(self.att(x), dim=1)
        return (x * att).sum(1)

class FinalModel(nn.Module):
    def __init__(self, user_num, item_num, vocab_size, dim=64):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num, dim)
        self.encoder = WordCNNAttention(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim*3, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
    def forward(self, u, i, t):
        te = self.word_emb(t)
        tf = self.encoder(te)
        uf = self.user_emb(u)
        i_f = self.item_emb(i)
        return self.fc(torch.cat([uf, i_f, tf], dim=-1)).squeeze(-1)

if __name__ == "__main__":
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = load_amazon_data("/mnt/ZQL/PBFL/Musical_Instruments_5.json")

    all_words = []
    for t in df.review_text:
        all_words += str(t).split()
    vocab = list(set(all_words))
    word2id = {w:i+2 for i,w in enumerate(vocab)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1

    user_num = df.user_id.max() + 1
    item_num = df.item_id.max() + 1

    split = int(0.8 * len(df))
    train_loader = DataLoader(PBFLDataset(df.iloc[:split], word2id), batch_size=256, shuffle=True)
    test_loader = DataLoader(PBFLDataset(df.iloc[split:], word2id), batch_size=256)

    model = FinalModel(user_num, item_num, len(word2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cri = nn.MSELoss()

    best_mae = 999
    best_rmse = 999

    print("="*60)
    print(" PBFL 模型训练与测试（真实数据集）")
    print("="*60)

    for ep in range(10):
        model.train()
        t_loss = 0
        for u,i,t,y in train_loader:
            u,i,t,y = u.to(device),i.to(device),t.to(device),y.to(device)
            opt.zero_grad()
            pred = model(u,i,t)
            pred = pred.float()
            loss = cri(pred, y)
            loss.backward()
            opt.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        mae_sum = 0
        rmse_sum = 0
        with torch.no_grad():
            for u,i,t,y in test_loader:
                u,i,t,y = u.to(device),i.to(device),t.to(device),y.to(device)
                pred = model(u,i,t)
                pred = pred.float()
                mae_sum += torch.abs(pred - y).mean().item()
                rmse_sum += torch.sqrt(cri(pred, y)).item()

        mae = mae_sum / len(test_loader)
        rmse = rmse_sum / len(test_loader)

        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            torch.save(model.state_dict(), "best_pbfl_model.pth")

        print(f"Epoch {ep+1:2d} | TrainLoss {t_loss:.4f} | MAE {mae:.4f} | RMSE {rmse:.4f}")

    print("\n" + "="*60)
    print("          实验最终结果")
    print("="*60)
    print(f"  最佳 MAE = {best_mae:.4f}")
    print(f"  最佳 RMSE = {best_rmse:.4f}")
    print("  最优模型已保存：best_pbfl_model.pth")
    print("="*60)
