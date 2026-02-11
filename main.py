import pandas as pd
from src.data_loader import load_data, build_utility_matrix
from src.engine import HeuristicRecommender
from src.metrics import train_test_split_by_user, calculate_metrics

def main():
    # 1. Load Data
    full_data = load_data()
    print(f"Data Loaded: {full_data.shape[0]} ratings.")

    # 2. Evaluation Mode
    print("\n--- Starting Evaluation Run ---")
    train_df, test_df = train_test_split_by_user(full_data)
    
    # Build utility matrix from TRAIN data only
    train_matrix = build_utility_matrix(train_df)
    
    # Initialize engine
    rec_eval = HeuristicRecommender(train_matrix)
    rec_eval.compute_similarity_matrix()
    
    # Calculate Verification Metrics
    metrics = calculate_metrics(rec_eval, test_df, top_k_users=50, k_recs=10, threshold=4.0)
    
    print("\n" + "="*40)
    print("      VERIFICATION REPORT")
    print("="*40)
    print(f"MAE (Mean Absolute Error) : {metrics['MAE']:.4f}")
    print(f"RMSE (Root Mean Sq Error) : {metrics['RMSE']:.4f}")
    print("-" * 40)
    print(f"Precision@10              : {metrics['Precision@10']:.2%}")
    print(f"Recall@10                 : {metrics['Recall@10']:.2%}")
    print("="*40 + "\n")
    
    print("Interpretation:")
    print("MAE/RMSE: Lower is better (0.0 is perfect).")
    print("Precision: Of the recommended movies, this % was actually liked.")
    print("Recall: Of all the movies users liked, this % was recommended.")

    # 3. Production Mode (Full Data)
    print("\n--- Starting Full Recommendation ---")
    full_matrix = build_utility_matrix(full_data)
    recommender = HeuristicRecommender(full_matrix)
    recommender.compute_similarity_matrix()
    
    # Example Recommendation
    target_user = 10
    recommendations = recommender.recommend(target_user, n_recommendations=5)
    
    print(f"\nTop 5 Recommendations for User {target_user}:")
    print(recommendations)

if __name__ == "__main__":
    main()