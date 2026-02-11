import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class HeuristicRecommender:
    def __init__(self, utility_matrix: pd.DataFrame):
        """
        Initializes the recommender with a utility matrix (Users x Movies).
        """
        self.utility_matrix = utility_matrix
        self.similarity_matrix = None
        self.num_movies = utility_matrix.shape[1]

    def compute_similarity_matrix(self):
        """
        Computes the User-User similarity matrix using vectorized operations.
        Logic: d = 1 / (1 + EuclideanDist) * (CommonItems / TotalItems)
        """
        print("Computing similarity matrix (Vectorized)...")
        
        # Fill NaNs with 0 for matrix math
        X = self.utility_matrix.fillna(0).values
        
        # Boolean matrix (1 if rated, 0 if not)
        B = (X != 0).astype(float)
        
        # Calculate Euclidean Distance components using (a-b)^2 = a^2 + b^2 - 2ab
        # Dot product of users
        X_dot = np.dot(X, X.T)
        
        # Squares
        X_sq = X ** 2
        
        # Norm restricted to common items
        # This calculates sum(x^2) but only for indices where the OTHER user also rated
        sq_norm = np.dot(X_sq, B.T)
        
        # Squared Euclidean Distance
        dist_sq = sq_norm + sq_norm.T - 2 * X_dot
        dist_sq = np.maximum(dist_sq, 0) # Clip negative zeros
        euclidean_dist = np.sqrt(dist_sq)
        
        # Transform distance to similarity component 1
        sim_component_1 = 1 / (1 + euclidean_dist)
        
        # Calculate number of common items (component 2)
        common_items = np.dot(B, B.T)
        sim_component_2 = common_items / self.num_movies
        
        # Final heuristic similarity
        self.similarity_matrix = sim_component_1 * sim_component_2
        
        # Set diagonal to 0 (user is not similar to self for recommendation purposes)
        np.fill_diagonal(self.similarity_matrix, 0.0)
        
        print("Similarity matrix computed.")

    def find_similar_users(self, user_id: int, top_k: int) -> pd.Series:
        """Returns the top_k most similar users and their normalized scores."""
        if self.similarity_matrix is None:
            raise Exception("Run compute_similarity_matrix first.")
            
        sim_scores = pd.Series(self.similarity_matrix[user_id], index=self.utility_matrix.index)
        top_users = sim_scores.sort_values(ascending=False).head(top_k)
        
        # Normalize so weights sum to 1
        total_score = top_users.sum()
        if total_score > 0:
            return top_users / total_score
        return top_users

    def predict_ratings(self, user_id: int, top_k_users: int = 50) -> Dict[int, float]:
        """
        Predicts ratings for un-rated movies based on similar users.
        Returns a dictionary {movie_id: predicted_score}.
        """
        similar_users = self.find_similar_users(user_id, top_k_users)
        
        if similar_users.empty:
            return {}

        # Get ratings of the neighbors
        neighbor_ratings = self.utility_matrix.loc[similar_users.index]
        
        # Weighted Average Calculation
        # Pred = sum(sim * rating) / sum(|sim|)
        numerator = neighbor_ratings.mul(similar_users, axis=0).sum(axis=0)
        denominator = neighbor_ratings.notna().mul(similar_users, axis=0).sum(axis=0)
        
        predicted_scores = numerator / denominator
        
        # Drop NaNs (movies no neighbor has seen)
        return predicted_scores.dropna().to_dict()

    def recommend(self, user_id: int, n_recommendations: int = 10, top_k_users: int = 50) -> pd.DataFrame:
        """
        Returns top N movie recommendations for a specific user, filtering out 
        movies they have already seen.
        """
        predictions = self.predict_ratings(user_id, top_k_users)
        
        # Filter out seen movies
        user_seen = self.utility_matrix.loc[user_id].dropna().index
        
        # Create DataFrame
        recs_df = pd.DataFrame(list(predictions.items()), columns=['movie_id', 'predicted_score'])
        recs_df = recs_df[~recs_df['movie_id'].isin(user_seen)]
        
        return recs_df.sort_values(by='predicted_score', ascending=False).head(n_recommendations)