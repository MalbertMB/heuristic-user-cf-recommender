import numpy as np
import pandas as pd
from src.data_loader import build_utility_matrix

def train_test_split_by_user(data: pd.DataFrame, test_users_frac: float = 0.1, test_items_frac: float = 0.2) -> tuple:
    """
    Splits data into Train and Test.
    1. Selects 10% of users to be 'Test Users'.
    2. For those users, hides 20% of their ratings (Test Set) and keeps 80% in Train Set.
    3. All other users are fully in Train Set.
    """
    print("Splitting data into Train and Test sets...")
    
    unique_users = data['user_id'].unique()
    test_user_ids = np.random.choice(unique_users, size=int(len(unique_users) * test_users_frac), replace=False)
    
    mask_test_users = data['user_id'].isin(test_user_ids)
    train_data_base = data[~mask_test_users]
    test_pool = data[mask_test_users]
    
    # Sample test items
    test_set = test_pool.groupby('user_id', group_keys=False).apply(lambda x: x.sample(frac=test_items_frac, random_state=42))
    
    # Fix for pandas 2.2+ dropping grouping keys
    if 'user_id' not in test_set.columns:
        test_set['user_id'] = test_pool.loc[test_set.index, 'user_id']

    train_data_extra = test_pool.drop(test_set.index)
    train_set = pd.concat([train_data_base, train_data_extra])
    
    return train_set, test_set

def calculate_metrics(recommender, test_set: pd.DataFrame, top_k_users: int = 50, k_recs: int = 10, threshold: float = 4.0) -> dict:
    """
    Calculates a comprehensive report of metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Precision@K: % of recommendations that are relevant
    - Recall@K: % of relevant items that were recommended
    
    :param k_recs: Number of recommendations to consider for Precision/Recall (e.g., Top 10)
    :param threshold: Rating threshold to consider a movie "Relevant" (e.g., ratings >= 4.0)
    """
    print(f"Calculating Verification Metrics (K={k_recs}, Threshold={threshold})...")
    
    mae_errors = []
    rmse_errors = []
    
    precision_scores = []
    recall_scores = []
    
    test_users = test_set['user_id'].unique()
    
    for user_id in test_users:
        # 1. Get predictions for this user
        # We need ALL predictions to find the Top K, so we predict for the entire test set candidates
        preds = recommender.predict_ratings(user_id, top_k_users)
        
        # Get actual ratings from Test data
        user_test_data = test_set[test_set['user_id'] == user_id]
        actuals = user_test_data.set_index('movie_id')['rating']
        
        # --- ERROR METRICS (MAE / RMSE) ---
        # Compare prediction vs actual for items that exist in both
        common_movies = set(preds.keys()).intersection(set(actuals.index))
        
        for movie_id in common_movies:
            pred_val = preds[movie_id]
            real_val = actuals[movie_id]
            
            mae_errors.append(abs(pred_val - real_val))
            rmse_errors.append((pred_val - real_val) ** 2)

        # --- RANKING METRICS (Precision / Recall) ---
        # Identify "Relevant" items in the test set (What the user actually liked)
        relevant_items = set(actuals[actuals >= threshold].index)
        
        if len(relevant_items) == 0:
            continue  # Skip users with no "good" movies in test set
            
        # Get Top K recommendations from our predictions
        # We only care about movies that are in the test set for verification
        # (In a real scenario, we'd rank against all movies, but here we can only verify against what we know)
        
        # Sort predictions by score descending
        top_recs = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        
        # Filter to keep only items that appear in the test set (so we can verify them)
        verify_candidates = [m_id for m_id, score in top_recs if m_id in actuals.index]
        
        # Take top K of verifyable candidates
        top_k_verify = verify_candidates[:k_recs]
        
        if not top_k_verify:
            continue

        # Hits: Recommended items that are also Relevant
        hits = len(set(top_k_verify).intersection(relevant_items))
        
        # Precision: Hits / K
        precision = hits / len(top_k_verify)
        
        # Recall: Hits / Total Relevant
        recall = hits / len(relevant_items)
        
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Compile results
    metrics = {
        "MAE": np.mean(mae_errors) if mae_errors else 0.0,
        "RMSE": np.sqrt(np.mean(rmse_errors)) if rmse_errors else 0.0,
        "Precision@10": np.mean(precision_scores) if precision_scores else 0.0,
        "Recall@10": np.mean(recall_scores) if recall_scores else 0.0
    }
    
    return metrics