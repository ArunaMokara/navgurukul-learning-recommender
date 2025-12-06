import streamlit as st
import pandas as pd
import numpy as np
import torch

from src.models.popular import MostPopularRecommender
from src.models.item_cf import ItemCFRecommender
from src.models.gru_seq import load_model


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/interactions.csv")
    return df

df = load_data()

users = df["user_id"].unique().tolist()


@st.cache_resource
def load_all_models():
    df_train = df[df["split"] == "train"]

    pop = MostPopularRecommender()
    pop.fit(df_train)

    cf = ItemCFRecommender()
    cf.fit(df_train)

    gru_model, item2idx, idx2item = load_model()
    return pop, cf, gru_model, item2idx, idx2item


pop_model, cf_model, gru_model, item2idx, idx2item = load_all_models()


def gru_recommend(history, k=10, max_len=50):
    seq = [item2idx[i] for i in history if i in item2idx]

    if len(seq) == 0:
        return []

    if len(seq) < max_len:
        seq = [0] * (max_len - len(seq)) + seq
    else:
        seq = seq[-max_len:]

    x = torch.tensor(seq, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        logits = gru_model(x).numpy()[0]

    top_idx = np.argsort(logits)[::-1]
    recs = [idx2item[i] for i in top_idx if i in idx2item]

    return recs[:k]



st.title("Learning Recommendation System Demo")
st.write("Choose a user and a model to generate learning recommendations.")

selected_user = st.selectbox("Select User ID", users)

user_history = df[(df["user_id"] == selected_user) & (df["split"] != "test")]["item_id"].tolist()
st.write("### User Learning History (Most Recent 10)")
st.write(user_history[-10:])

model_choice = st.radio(
    "Choose Recommendation Model",
    ("Most Popular", "Collaborative Filtering (CF)", "GRU Sequence Model")
)

if st.button("Get Recommendations"):
    if model_choice == "Most Popular":
        recs = pop_model.recommend(user_history, k=10)

    elif model_choice == "Collaborative Filtering (CF)":
        recs = cf_model.recommend(user_history, k=10)

    elif model_choice == "GRU Sequence Model":
        recs = gru_recommend(user_history, k=10)

    st.write("### Recommended Next Skills")
    st.write(recs)
