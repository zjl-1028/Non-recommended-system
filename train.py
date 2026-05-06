import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PBFLDataset(Dataset):
    def __init__(self, df, word2id, max_len=150):
        self.df = df
        self.word2id = word2id
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = int(row['user_id'])
        iid = int(row['item_id'])
        txt = str(row['review_text'])
        rt = float(row['rating'])

        ws = txt.split()[:self.max_len]
        ids = [self.word2id.get(w, 1) for w in ws]
        ids += [0]*(self.max_len-len(ids))
        return torch.tensor(uid), torch.tensor(iid), torch.tensor(ids), torch.tensor(rt)

def train_epoch(model, loader, opt, cri, dev):
    model.train()
    loss = 0
    for u,i,r,y in loader:
        u,i,r,y = u.to(dev),i.to(dev),r.to(dev),y.to(dev)
        opt.zero_grad()
        pred = model(u,i,r)
        l = cri(pred,y)
        l.backward()
        opt.step()
        loss += l.item()
    return loss/len(loader)

def eval(model, loader, cri, dev):
    model.eval()
    loss, mae = 0,0
    with torch.no_grad():
        for u,i,r,y in loader:
            u,i,r,y = u.to(dev),i.to(dev),r.to(dev),y.to(dev)
            pred = model(u,i,r)
            loss += cri(pred,y).item()
            mae += torch.mean(torch.abs(pred-y)).item()
    return loss/len(loader), mae/len(loader)

if __name__ == "__main__":
    set_seed()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    USER_NUM = 10000
    ITEM_NUM = 5000
    BATCH = 256
    EPOCHS = 20

    # 示例数据，替换为你的PBFL数据集路径
    df = pd.read_csv("pbfl_data.csv")
    all_words = []
    for t in df['review_text']: 
        all_words += str(t).split()
    vocab = list(set(all_words))
    w2i = {w:i+2 for i,w in enumerate(vocab)}
    w2i['<pad>']=0
    w2i['<unk>']=1

    split = int(0.8*len(df))
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]

    train_loader = DataLoader(PBFLDataset(train_df,w2i), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(PBFLDataset(val_df,w2i), batch_size=BATCH)

    from model import FinalModel
    model = FinalModel(USER_NUM, ITEM_NUM, len(w2i)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cri = nn.MSELoss()

    best = 1e9
    for ep in range(EPOCHS):
        tl = train_epoch(model,train_loader,opt,cri,dev)
        vl, vm = eval(model,val_loader,cri,dev)
        print(f"Ep {ep+1} | Train {tl:.4f} | Val {vl:.4f} | MAE {vm:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), "best_model.pth")
