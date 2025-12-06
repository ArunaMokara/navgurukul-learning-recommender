import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

DATA_PATH = Path("data/processed/interactions.csv")
df = pd.read_csv(DATA_PATH)
print(df.head(10))
print(df.columns)
class ItemCFRecommender:
    def __init__(self, max_items=1000):
        self.max_items = max_items
        self.item_sim = None
        self.item_to_idx = None
        self.idx_to_item = None

    def fit(self, df_train):
        item_counts = df_train.groupby("item_id").size().sort_values(ascending=False)
        top_items = item_counts.index.tolist()[:self.max_items]

        df = df_train[df_train["item_id"].isin(top_items)]

        # Maping items and users
        self.item_to_idx = {item: ix for ix, item in enumerate(top_items)}
        self.idx_to_item = {ix: item for item, ix in self.item_to_idx.items()}
        user_to_idx = {u: ix for ix, u in enumerate(df["user_id"].unique())}

        # Build sparse matrix (user x item)
        rows = df["user_id"].map(user_to_idx).values
        cols = df["item_id"].map(self.item_to_idx).values
        vals = df["correct"].fillna(1).values  

        matrix = csr_matrix((vals, (rows, cols)),
                            shape=(len(user_to_idx), len(self.item_to_idx)))

        print("Computing item-item cosine similarity...")
        self.item_sim = cosine_similarity(matrix.T)

    def recommend(self, user_history, k=10):
        indices = [self.item_to_idx[i] for i in user_history if i in self.item_to_idx]
        if not indices:
            sims = self.item_sim.sum(axis=0)
        else:
            sims = self.item_sim[indices].mean(axis=0)

        ranked_indices = np.argsort(sims)[::-1]

        seen = set(user_history)
        recs = []

        for idx in ranked_indices:
            item = self.idx_to_item[idx]
            if item not in seen:
                recs.append(item)
            if len(recs) >= k:
                break

        return recs

def load_train_data():
    df = pd.read_csv(DATA_PATH)
    return df[df["split"] == "train"]

if __name__ == "__main__":
    df_train = load_train_data()
    model = ItemCFRecommender()
    model.fit(df_train)
    print("Top recommendations for user with history [7014] →", model.recommend([7014], k=5))
