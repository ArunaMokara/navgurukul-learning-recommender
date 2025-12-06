import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/interactions.csv")

class MostPopularRecommender:
    def __init__(self, top_n=100):
        self.top_n = top_n
        self.rank_list = None

    def fit(self, df_train):
        item_counts = df_train.groupby("item_id").size().sort_values(ascending=False)
        self.rank_list = item_counts.index.tolist()[:self.top_n]

    def recommend(self, user_history, k=10):
        seen = set(user_history)
        recs = [item for item in self.rank_list if item not in seen]
        return recs[:k]

def load_train_data():
    df = pd.read_csv(DATA_PATH)
    return df[df["split"] == "train"]

if __name__ == "__main__":
    df_train = load_train_data()
    model = MostPopularRecommender()
    model.fit(df_train)
    print("Top 10 Popular Items:", model.rank_list[:10])
