import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from popular import MostPopularRecommender
from item_cf import ItemCFRecommender

import torch
import torch.nn as nn

DATA_PATH = Path("data/processed/interactions.csv")
GRU_MODEL_PATH = Path("models/gru_model.pt")


def precision_at_k(recommended, actual, k=10):
    recommended_k = recommended[:k]
    if not actual:
        return 0
    return len(set(recommended_k) & set(actual)) / k

def recall_at_k(recommended, actual, k=10):
    recommended_k = recommended[:k]
    if not actual:
        return 0
    return len(set(recommended_k) & set(actual)) / len(actual)

def ndcg_at_k(recommended, actual, k=10):
    recommended_k = recommended[:k]
    dcg = 0
    for i, item in enumerate(recommended_k):
        if item in actual:
            dcg += 1 / np.log2(i + 2)
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), k))])
    return dcg / idcg if idcg > 0 else 0

def mrr_at_k(recommended, actual, k=10):
    recommended_k = recommended[:k]
    for i, item in enumerate(recommended_k):
        if item in actual:
            return 1 / (i + 1)
    return 0


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


def load_gru_model():
    checkpoint = torch.load(GRU_MODEL_PATH, map_location="cpu")
    item2idx = checkpoint["item2idx"]
    idx2item = checkpoint["idx2item"]
    n_items = len(item2idx) + 1

    model = GRURec(n_items=n_items)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, item2idx, idx2item


def gru_recommend(model, item2idx, idx2item, history, k=10, max_len=50):
    if not history:
        return []

    seq = [item2idx[i] for i in history if i in item2idx]
    if len(seq) == 0:
        return []

    if len(seq) < max_len:
        seq = [0] * (max_len - len(seq)) + seq
    else:
        seq = seq[-max_len:]

    x = torch.tensor(seq, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits = model(x).numpy()[0]

    top_indices = np.argsort(logits)[::-1]
    recommended_items = [idx2item[i] for i in top_indices if i in idx2item]

    return recommended_items[:k]


def evaluate_all():
    df = pd.read_csv(DATA_PATH)
    test_df = df[df["split"] == "test"].copy()

    users = test_df["user_id"].unique().tolist()

    print("Loading models...")
    popular = MostPopularRecommender()
    popular.fit(df[df["split"] == "train"])

    cf = ItemCFRecommender()
    cf.fit(df[df["split"] == "train"])

    gru_model, item2idx, idx2item = load_gru_model()

    results = {
        "popular": {"precision": [], "recall": [], "ndcg": [], "mrr": []},
        "cf": {"precision": [], "recall": [], "ndcg": [], "mrr": []},
        "gru": {"precision": [], "recall": [], "ndcg": [], "mrr": []},
    }

    print("Evaluating models...")
    for user in tqdm(users):
        user_history = df[(df["user_id"] == user) & (df["split"] != "test")]["item_id"].tolist()
        actual_next = test_df[test_df["user_id"] == user]["item_id"].tolist()

        if len(actual_next) == 0 or len(user_history) == 0:
            continue

        # Popular
        pop_rec = popular.recommend(user_history, k=10)
        results["popular"]["precision"].append(precision_at_k(pop_rec, actual_next))
        results["popular"]["recall"].append(recall_at_k(pop_rec, actual_next))
        results["popular"]["ndcg"].append(ndcg_at_k(pop_rec, actual_next))
        results["popular"]["mrr"].append(mrr_at_k(pop_rec, actual_next))

        # CF
        cf_rec = cf.recommend(user_history, k=10)
        results["cf"]["precision"].append(precision_at_k(cf_rec, actual_next))
        results["cf"]["recall"].append(recall_at_k(cf_rec, actual_next))
        results["cf"]["ndcg"].append(ndcg_at_k(cf_rec, actual_next))
        results["cf"]["mrr"].append(mrr_at_k(cf_rec, actual_next))

        # GRU
        gru_rec = gru_recommend(gru_model, item2idx, idx2item, user_history, k=10)
        results["gru"]["precision"].append(precision_at_k(gru_rec, actual_next))
        results["gru"]["recall"].append(recall_at_k(gru_rec, actual_next))
        results["gru"]["ndcg"].append(ndcg_at_k(gru_rec, actual_next))
        results["gru"]["mrr"].append(mrr_at_k(gru_rec, actual_next))


    print("\n=== FINAL METRICS ===")

    for model in results:
        print(f"\n--- {model.upper()} ---")
        print("Precision@10:", np.mean(results[model]["precision"]))
        print("Recall@10:", np.mean(results[model]["recall"]))
        print("NDCG@10:", np.mean(results[model]["ndcg"]))
        print("MRR@10:", np.mean(results[model]["mrr"]))


if __name__ == "__main__":
    evaluate_all()
