import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import random
from model import FinalModel

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
            data.append([
                d['user_id'], d['item_id'],
                d['user_review'], d['item_review'],
                d['rating']
            ])
    return data

class ReviewDataset(Dataset):
    def __init__(self, data, word2id, max_len=150):
        self.data = data
        self.word2id = word2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, urv, irv, r = self.data[idx]
        urv = [self.word2id.get(w, 1) for w in urv.split()][:self.max_len]
        irv = [self.word2id.get(w, 1) for w in irv.split()][:self.max_len]
        urv += [0]*(self.max_len-len(urv))
        irv += [0]*(self.max_len-len(irv))
        return u, i, torch.tensor(urv), torch.tensor(irv), torch.tensor(r, dtype=torch.float32)

def build_word2id(data):
    words = set()
    for u,i,ur,ir,_ in data:
        words.update(ur.split())
        words.update(ir.split())
    word2id = {w:i+2 for i,w in enumerate(words)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    return word2id

if __name__ == "__main__":
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "/mnt/ZQL/PBFL/Musical_Instruments_5.json"
    data = load_data(data_path)
    word2id = build_word2id(data)

    user_num = max([d[0] for d in data]) + 1
    item_num = max([d[1] for d in data]) + 1

    dataset = ReviewDataset(data, word2id)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = FinalModel(user_num, item_num, len(word2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_mae = 999
    best_rmse = 999

    for ep in range(10):
        model.train()
        total_loss = 0
        for u,i,ur,ir,y in loader:
            u,i,ur,ir,y = u.to(device),i.to(device),ur.to(device),ir.to(device),y.to(device)
            opt.zero_grad()
            pred = model(u,i,ur,ir)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        mae = 0
        rmse = 0
        cnt = 0
        with torch.no_grad():
            for u,i,ur,ir,y in loader:
                u,i,ur,ir,y = u.to(device),i.to(device),ur.to(device),ir.to(device),y.to(device)
                pred = model(u,i,ur,ir)
                mae += torch.abs(pred-y).sum().item()
                rmse += (pred-y)**2
                cnt += len(y)
        mae /= cnt
        rmse = (rmse/cnt).sqrt().item()

        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {ep+1} | Loss: {total_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    print("="*50)
    print("Best Result:")
    print(f"MAE: {best_mae:.4f}")
    print(f"RMSE: {best_rmse:.4f}")
    print("="*50)
