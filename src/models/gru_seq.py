import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = Path("data/processed/interactions.csv")
MODEL_SAVE_PATH = Path("models/gru_model.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_item_mapping(df):
    items = df["item_id"].unique().tolist()
    item2idx = {item: idx + 1 for idx, item in enumerate(items)}  
    idx2item = {idx + 1: item for idx, item in enumerate(items)}
    return item2idx, idx2item


class SequenceDataset(Dataset):
    def __init__(self, df, item2idx, max_len=50):
        self.max_len = max_len
        self.item2idx = item2idx
        self.sequences = []

        # building sequences per user
        for uid, u_df in df.groupby("user_id"):
            items = u_df.sort_values("ts")["item_id"].tolist()
            if len(items) < 2:
                continue
            for i in range(1, len(items)):
                hist = items[max(0, i - max_len): i]
                target = items[i]
                self.sequences.append((hist, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        history, target = self.sequences[idx]
        hist_idx = [self.item2idx[it] for it in history]
        target_idx = self.item2idx[target]

        
        if len(hist_idx) < self.max_len:
            hist_idx = [0] * (self.max_len - len(hist_idx)) + hist_idx
        else:
            hist_idx = hist_idx[-self.max_len:]

        return torch.tensor(hist_idx, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)


class GRURec(nn.Module):
    def __init__(self, n_items, emb_dim=64, hidden_dim=128, max_len=50):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=n_items, embedding_dim=emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_items)

    def forward(self, x):
        x = self.emb(x)           # (batch, seq_len, emb_dim)
        _, h = self.gru(x)        # h: (1, batch, hidden_dim)
        h = h.squeeze(0)          # (batch, hidden_dim)
        logits = self.output(h)   # (batch, n_items)
        return logits

def train_gru():
    df = pd.read_csv(DATA_PATH)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

   
    item2idx, idx2item = build_item_mapping(train_df)
    n_items = len(item2idx) + 1  

   
    train_data = SequenceDataset(train_df, item2idx)
    val_data = SequenceDataset(val_df, item2idx)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)

    # building model
    model = GRURec(n_items=n_items).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training GRU model...")
    for epoch in range(3):  
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
        print(f"Epoch {epoch+1} | Val Loss: {val_loss / len(val_loader):.4f}")

    
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "item2idx": item2idx,
        "idx2item": idx2item,
    }, MODEL_SAVE_PATH)

    print(f"GRU model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_gru()

def load_model(model_path="models/gru_model.pt"):
    checkpoint = torch.load(model_path, map_location="cpu")

    item2idx = checkpoint["item2idx"]
    idx2item = checkpoint["idx2item"]
    n_items = len(item2idx) + 1

    class GRURec(nn.Module):
        def __init__(self, n_items, emb_dim=64, hidden_dim=128):
            super().__init__()
            self.emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
            self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
            self.output = nn.Linear(hidden_dim, n_items)

        def forward(self, x):
            x = self.emb(x)
            _, h = self.gru(x)
            h = h.squeeze(0)
            logits = self.output(h)
            return logits

    model = GRURec(n_items)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, item2idx, idx2item