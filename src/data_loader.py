import os
import zipfile
import requests
import pandas as pd
import numpy as np

DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_FOLDER = "data"
ML_FOLDER = os.path.join(DATA_FOLDER, "ml-1m")

def download_and_extract_data():
    """Downloads the MovieLens 1M dataset if not present."""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    
    zip_path = os.path.join(DATA_FOLDER, "ml-1m.zip")
    
    if not os.path.exists(ML_FOLDER):
        print("Downloading dataset...")
        response = requests.get(DATA_URL)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_FOLDER)

def load_data() -> pd.DataFrame:
    """
    Loads Users, Movies, and Ratings, merges them, and re-indexes IDs 
    to be 0-based consecutive integers.
    """
    download_and_extract_data()
    
    print("Loading data into DataFrames...")
    users = pd.read_table(f'{ML_FOLDER}/users.dat', sep='::', header=None, 
                          names=['user_id', 'gender', 'age', 'occupation', 'zip'], 
                          engine='python')
    
    ratings = pd.read_table(f'{ML_FOLDER}/ratings.dat', sep='::', header=None, 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'], 
                            engine='python')
    
    movies = pd.read_table(f'{ML_FOLDER}/movies.dat', sep='::', header=None, 
                           names=['movie_id', 'title', 'genres'], 
                           engine='python', encoding='latin-1')

    # Merge data
    data = pd.merge(pd.merge(ratings, users), movies)

    # Re-index IDs to ensure they are continuous (0 to N-1)
    # This is crucial for matrix operations
    data['user_id'] = pd.Categorical(data['user_id']).codes
    data['movie_id'] = pd.Categorical(data['movie_id']).codes

    return data

def build_utility_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a User-Item matrix where rows=users, cols=movies, values=ratings.
    """
    return data.pivot_table(index='user_id', columns='movie_id', values='rating')