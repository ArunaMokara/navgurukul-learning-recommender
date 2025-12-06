### Learning Recommendation System
This project implements an end-to-end learning recommendation system based on student interaction data.
It includes preprocessing, EDA, baseline models, an advanced GRU sequence model, evaluation metrics, and a Streamlit demo app.

#### Full Documentation
Navgurukul\NAVGURUKUL-HACKATHON-DOCUMENTATION.pdf

### Features

Temporal train/val/test split
EDA: user/item sparsity, distributions
##### Baseline models:
Most Popular
Item-based Collaborative Filtering
##### Advanced model:
GRU Sequence Recommender

##### Evaluation using:
Precision@10
Recall@10
NDCG@10
MRR@10

Streamlit demo to interact with all models

### Running the Project
#### 1. Run the App Directly
python -m streamlit run app/app.py

The app will open automatically at:
http://localhost:8501

### 2. Reproduce the Entire Pipeline (Optional)
#### Run Data Preprocessing
python src/data_preprocess.py
#### Train GRU Sequence Model
python src/models/gru_seq.py
#### Run Evaluation of All Models
python src/models/evaluate.py



