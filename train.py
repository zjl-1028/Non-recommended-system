import torch
import torch.nn as nn
import json
import re
from torch.utils.data import Dataset, DataLoader
from model import FinalModel

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def load_amazon_data():
    data = []
    path = "/mnt/ZQL/PBFL/Musical_Instruments_5.json"
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            uid = hash(d['reviewerID']) % 100000
            iid = hash(d['asin']) % 100000
            review = clean_text(d['reviewText'])
            rating = float(d['overall'])
            data.append([uid, iid, review, rating])
    return data

class ReviewDataset(Dataset):
    def __init__(self, data, word2id, max_len=150):
        self.data = data
        self.word2id = word2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, rev, rt = self.data[idx]
        tokens = [self.word2id.get(w, 1) for w in rev.split()][:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return u, i, torch.LongTensor(tokens), torch.FloatTensor([rt])

def build_vocab(data):
    vocab = set()
    for u, i, r, rt in data:
        vocab.update(r.split())
    word2id = {w: i+2 for i, w in enumerate(vocab)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    return word2id

if __name__ == "__main__":
    device = torch.device('cuda')
    amazon_data = load_amazon_data()
    word2id = build_vocab(amazon_data)
    dataset = ReviewDataset(amazon_data, word2id)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = FinalModel(100000, 100000, len(word2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for u, i, rev, rt in loader:
            u, i, rev = u.to(device), i.to(device), rev.to(device)
            rt = rt.squeeze().to(device)

            pred_score, _ = model(u, i, rev, rev)
            loss = mse_loss(pred_score, rt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:2d} | Loss = {total_loss:.4f}")
