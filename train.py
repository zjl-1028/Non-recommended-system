import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

DATA_PATH = "amazon_review.txt"

class AmazonDataset(Dataset):
    def __init__(self, max_seq_len=30):
        self.max_seq_len = max_seq_len
        self.user_num = 0
        self.item_num = 0
        self.vocab_size = 0
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                u = int(parts[0])
                i = int(parts[1])
                r = float(parts[2])
                seq = list(map(int, parts[3].split(',')))
                if len(seq) > self.max_seq_len:
                    seq = seq[:self.max_seq_len]
                else:
                    seq += [0] * (self.max_seq_len - len(seq))
                self.user_num = max(self.user_num, u + 1)
                self.item_num = max(self.item_num, i + 1)
                self.vocab_size = max(self.vocab_size, max(seq) + 1)
                data.append((u, i, r, seq))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, r, seq = self.data[idx]
        return torch.tensor(u), torch.tensor(i), torch.tensor(r), torch.tensor(seq)

def get_loaders(batch_size=32):
    dataset = AmazonDataset()
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, dataset.user_num, dataset.item_num, dataset.vocab_size

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, user_num, item_num, vocab_size = get_loaders()
    model = PBFLSarcModel(user_num, item_num, vocab_size).to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        train_loss = 0
        for u, i, r, seq in train_loader:
            u, i, r, seq = u.to(device), i.to(device), r.to(device), seq.to(device)
            optimizer.zero_grad()
            y_pred, _, _, _ = model(u, i, seq)
            loss = mse_loss(y_pred, r)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, r, seq in val_loader:
                u, i, r, seq = u.to(device), i.to(device), r.to(device), seq.to(device)
                y_pred, _, _, _ = model(u, i, seq)
                loss = mse_loss(y_pred, r)
                val_loss += loss.item()

        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    main()
