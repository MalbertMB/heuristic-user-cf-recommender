# Heuristic Recommender System

A high-performance, vectorized implementation of a **User-Based Collaborative Filtering** recommender system. This project processes the MovieLens 1M dataset to generate personalized movie recommendations using Euclidean distance similarity metrics.

---

## Project Overview

This repository contains a Python-based recommendation engine that employs memory-based collaborative filtering. Unlike iterative approaches, this implementation utilizes NumPy broadcasting and vectorized matrix operations to compute similarity matrices efficiently. It includes:

- A complete data pipeline  
- An object-oriented recommendation engine  
- A comprehensive evaluation suite for performance metrics  

---

## Features

- **Vectorized Computation**: Optimized calculation of the User–User similarity matrix using NumPy, significantly reducing computation time compared to iterative loops.  
- **Heuristic Similarity**: Implements a weighted Euclidean distance metric that prioritizes users with a higher volume of shared movie ratings.  
- **Object-Oriented Design**: Modular architecture separating data loading, core logic, and metric evaluation.  
- **Robust Evaluation**: Includes a custom train–test split strategy that respects user history and calculates MAE, RMSE, Precision@K, and Recall@K.  
- **Automatic Data Management**: Automatically downloads and extracts the MovieLens 1M dataset if not present.  

---

## Directory Structure

```text
heuristic-recommender/
├── data/                  # Storage for the MovieLens dataset (auto-generated)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # ETL pipeline for MovieLens data
│   ├── engine.py          # Core HeuristicRecommender class
│   └── metrics.py         # Evaluation metrics and train/test splitting logic
├── main.py                # Application entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd heuristic-recommender
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

To run the full pipeline, including data downloading, model evaluation, and sample recommendation generation:

```bash
python main.py
```

---

## Execution Flow

### 1. Data Ingestion

- Downloads the MovieLens 1M dataset (if missing).
- Parses **Users**, **Movies**, and **Ratings**.

### 2. Model Evaluation

- Splits the data into training and testing sets by user.
- Computes the similarity matrix on the training set.
- Calculates error metrics (**MAE**, **RMSE**) and ranking metrics (**Precision@10**, **Recall@10**).

### 3. Full Recommendation

- Retrains the model on the complete dataset.
- Generates and displays top-N recommendations for a sample user.

---

## Methodology

### Similarity Metric

The system calculates similarity between two users $u$ and $v$ using a modified Euclidean distance formula that rewards higher overlap in rated items:

$$
\mathrm{Sim}(u, v)
=
\frac{1}{1 + \mathrm{Dist}(u, v)}
\cdot
\frac{\lvert I_u \cap I_v \rvert}{\lvert I_{\text{total}} \rvert}
$$

Where:

- $\mathrm{Dist}(u, v)$ is the Euclidean distance between ratings of mutually rated items.  
- $\lvert I_u \cap I_v \rvert$ is the count of items rated by both users.  
- $\lvert I_{\text{total}} \rvert$ is the total number of items in the system.  

### Prediction Strategy

Ratings are predicted using a weighted average of the ratings from the $k$ most similar users:

$$
\hat{r}_{u,i}
=
\frac{
\sum_{v \in N_k(u)} \mathrm{Sim}(u, v)\, r_{v,i}
}{
\sum_{v \in N_k(u)} \left| \mathrm{Sim}(u, v) \right|
}
$$

Where:

- $N_k(u)$ represents the set of the $k$ most similar users to user $u$.  
- $r_{v,i}$ is the rating given by user $v$ to item $i$.  

---

## Performance Metrics

The system evaluates performance using the following metrics:

- **MAE (Mean Absolute Error):** Measures the average magnitude of errors in predicted ratings.  
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.  
- **Precision@K:** The proportion of recommended items that are relevant to the user.  
- **Recall@K:** The proportion of relevant items that were successfully recommended.  

---

## License

This project is open-source and available under the **MIT License**.
