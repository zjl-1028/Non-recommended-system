import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from model import FinalModel

class ReviewDataset(Dataset):
    def __init__(self, user_num, item_num, vocab_size, max_len=20):
        self.user_num = user_num
        self.item_num = item_num
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.len = 10000

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        u = torch.randint(0, self.user_num, (1,)).item()
        i = torch.randint(0, self.item_num, (1,)).item()
        ur = torch.randint(0, self.vocab_size, (self.max_len,))
        ir = torch.randint(0, self.vocab_size, (self.max_len,))
        rating = torch.rand(1).item() * 4 + 1
        sarc = torch.randint(0, 2, (1,)).item()
        return u, i, ur, ir, rating, sarc

def get_data_loaders(user_num, item_num, vocab_size, batch_size=32):
    dataset = ReviewDataset(user_num, item_num, vocab_size)
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train():
    user_num = 1000
    item_num = 500
    vocab_size = 5000
    dim = 64
    lr = 1e-3
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = get_data_loaders(user_num, item_num, vocab_size)
    model = FinalModel(user_num, item_num, vocab_size, dim).to(device)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for u, i, ur, ir, r, sc in train_loader:
            u, i, ur, ir, r, sc = u.to(device), i.to(device), ur.to(device), ir.to(device), r.to(device), sc.to(device)
            optimizer.zero_grad()
            pred_r, sarc_logits = model(u, i, ur, ir)
            loss = mse_loss(pred_r, r) + ce_loss(sarc_logits, sc)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, ur, ir, r, sc in val_loader:
                u, i, ur, ir, r, sc = u.to(device), i.to(device), ur.to(device), ir.to(device), r.to(device), sc.to(device)
                pred_r, sarc_logits = model(u, i, ur, ir)
                loss = mse_loss(pred_r, r) + ce_loss(sarc_logits, sc)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == '__main__':
    train()
