import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RAW_PATH = Path("data/raw/2015_100_skill_builders_main_problems.csv")
OUT_PATH = Path("data/processed/interactions.csv")
PLOT_PATH = Path("data/processed/")
PLOT_PATH.mkdir(parents=True, exist_ok=True)

def load_and_preprocess():
    print("Loading dataset...")
    df = pd.read_csv(RAW_PATH)

    # NULL value check
    print("\n--- NULL VALUE CHECK ---")
    print(df.isnull().sum())

    # Duplicate check
    print("\n--- DUPLICATE CHECK ---")
    print("Duplicate rows:", df.duplicated().sum())

    print("\nUnique values for correct:", df["correct"].unique())

    # Renaming sequence_id -> item_id (learning item)
    df = df.rename(columns={"sequence_id": "item_id"})

    # Using log_id as timestamp/order
    df["ts"] = df["log_id"]

    # Sorting by user + time
    df = df.sort_values(["user_id", "ts"]).reset_index(drop=True)

    print("Creating temporal train/val/test split...")

    def temporal_split(user_df):
        n = len(user_df)
        if n < 5:
            user_df["split"] = "train"
        else:
            t1 = int(0.7 * n)
            t2 = int(0.85 * n)
            user_df.loc[user_df.index[:t1], "split"] = "train"
            user_df.loc[user_df.index[t1:t2], "split"] = "val"
            user_df.loc[user_df.index[t2:], "split"] = "test"
        return user_df

    df = df.groupby("user_id").apply(temporal_split).reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved processed data → {OUT_PATH}")

    return df

def run_eda(df):
    print("\nRunning EDA...")
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_inter = len(df)
    density = n_inter / (n_users * n_items)

    print("\n--- DATA STATS ---")
    print(f"Users: {n_users}")
    print(f"Learning Items (item_id): {n_items}")
    print(f"Total Interactions: {n_inter}")
    print(f"Matrix Density: {density:.8f}")

    # Plotting Interactions per user
    user_counts = df.groupby("user_id").size()
    user_counts.hist(bins=30)
    plt.title("Interactions per User")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_PATH / "interactions_per_user.png")
    plt.close()

    # Plotting Interactions per item
    item_counts = df.groupby("item_id").size()
    item_counts.hist(bins=30)
    plt.title("Interactions per Learning Item")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_PATH / "interactions_per_item.png")
    plt.close()


    # Cold start analysis
    cold_users = (user_counts < 5).mean()
    cold_items = (item_counts < 5).mean()

    print("\n--- COLD START ---")
    print(f"Cold Users (<5 interactions): {cold_users * 100:.2f}%")
    print(f"Cold Items (<5 interactions): {cold_items * 100:.2f}%")

def main():
    df = load_and_preprocess()
    run_eda(df)
    print("\nEDA completed successfully.")

if __name__ == "__main__":
    main()
