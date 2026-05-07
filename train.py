import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append([d['user_id'], d['item_id'], d['review'], d['sarcasm_label'], d['rating']])
    return data

class ReviewDataset(Dataset):
    def __init__(self, data, word2id, max_len=150):
        self.data = data
        self.word2id = word2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, rev, sl, r = self.data[idx]
        rev = [self.word2id.get(w, 1) for w in rev.split()][:self.max_len]
        rev += [0]*(self.max_len-len(rev))
        return u, i, torch.tensor(rev), torch.tensor(rev), sl, torch.tensor(r, dtype=torch.float32)

def build_word2id(data):
    words = set()
    for u,i,r,sl,rt in data:
        words.update(r.split())
    word2id = {w:i+2 for i,w in enumerate(words)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    return word2id

if __name__ == "__main__":
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_data("/mnt/ZQL/PBFL/data.json")
    word2id = build_word2id(data)

    user_num = max([d[0] for d in data]) + 1
    item_num = max([d[1] for d in data]) + 1

    loader = DataLoader(ReviewDataset(data, word2id), batch_size=256, shuffle=True)

    from model import FinalModel
    model = FinalModel(user_num, item_num, len(word2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    for ep in range(10):
        model.train()
        for u,i,ur,ir,sl,y in loader:
            u,i,ur,ir,sl,y = u.to(device),i.to(device),ur.to(device),ir.to(device),sl.to(device),y.to(device)
            opt.zero_grad()
            pred, sarc_logits = model(u,i,ur,ir)

            loss_r = mse(pred, y)
            loss_s = ce(sarc_logits, sl)
            loss = loss_r + loss_s

            loss.backward()
            opt.step()

        model.eval()
        mae = 0
        acc = 0
        cnt = 0
        with torch.no_grad():
            for u,i,ur,ir,sl,y in loader:
                u,i,ur,ir,sl,y = u.to(device),i.to(device),ur.to(device),ir.to(device),sl.to(device),y.to(device)
                pred, sarc_logits = model(u,i,ur,ir)
                mae += torch.abs(pred-y).sum().item()
                acc += (torch.argmax(sarc_logits,-1)==sl).sum().item()
                cnt += len(y)
        print(f"EP{ep} | MAE:{mae/cnt:.4f} | ACC:{acc/cnt:.4f}")
