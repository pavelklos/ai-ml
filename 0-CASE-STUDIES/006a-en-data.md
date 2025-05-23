# Getting Started with the Movie Recommendation System Project

This guide explains how to obtain the necessary dataset and set up your environment for the Movie Recommendation System case study.

## Required Python Packages

First, install the required Python libraries:

```python
pip install pandas numpy matplotlib seaborn scikit-learn surprise flask scipy joblib
```

## Dataset Information

This project uses the MovieLens 100K dataset, which contains 100,000 ratings from 943 users on 1,682 movies.

### Option 1: Download the MovieLens 100K Dataset Directly

```python
import os
import urllib.request
import zipfile
from io import BytesIO

# Create directories if they don't exist
os.makedirs('ml-100k', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Download the MovieLens 100K dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
print(f"Downloading MovieLens 100K dataset from {url}...")

try:
    # Download and extract the dataset
    response = urllib.request.urlretrieve(url, "ml-100k.zip")
    
    # Extract the ZIP file
    with zipfile.ZipFile("ml-100k.zip", "r") as zip_ref:
        zip_ref.extractall("./")
    
    # Remove the ZIP file
    os.remove("ml-100k.zip")
    
    print("Dataset downloaded and extracted successfully!")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
```

### Option 2: Manual Download

1. Visit the GroupLens MovieLens datasets page: https://grouplens.org/datasets/movielens/100k/
2. Download the ml-100k.zip file
3. Extract the ZIP file to your project directory
4. Ensure the extracted files are in a folder named `ml-100k`

## Project Structure Setup

Create the necessary directory structure for the project:

```python
import os

# Create project directories
directories = [
    'ml-100k',  # Will be created by dataset extraction if not already present
    'models',
    'visualizations'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## About the MovieLens 100K Dataset

The dataset contains:

### Main Files
- `u.data`: The full rating dataset with 100,000 ratings (user_id, movie_id, rating, timestamp)
- `u.item`: Information about the movies (movie_id, title, release date, genre indicators)
- `u.user`: Demographic information about users (user_id, age, gender, occupation, zip code)

### Data Format
- Ratings are on a scale of 1-5
- The data includes 19 different genres
- User demographic information includes age, gender, and occupation
- Release dates and IMDb links are available for movies

## Verifying the Setup

Run this code to check if your dataset is properly downloaded and formatted:

```python
import pandas as pd

try:
    # Define column names
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + \
                 ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    
    # Try to load the data files
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols)
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=movies_cols)
    users = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols)
    
    # Print dataset information
    print("Dataset verification successful!")
    print(f"Ratings data shape: {ratings.shape}")
    print(f"Movies data shape: {movies.shape}")
    print(f"Users data shape: {users.shape}")
    
    # Display sample data
    print("\nRatings sample:")
    print(ratings.head(3))
    
    print("\nMovies sample:")
    print(movies[['movie_id', 'title', 'release_date']].head(3))
    
    print("\nUsers sample:")
    print(users.head(3))
    
except Exception as e:
    print(f"Setup verification failed: {e}")
    print("Please ensure the MovieLens 100K dataset is correctly downloaded and extracted.")
```

## Expected Output Files and Visualizations

When running the complete code, the following files will be generated:

### Models
- `models/svd_model.pkl`: Trained SVD model for collaborative filtering
- `models/cosine_sim_matrix.npy`: Cosine similarity matrix for content-based filtering
- `models/train_matrix.pkl`: User-item matrix used for collaborative filtering

### Visualizations
Several charts will be displayed during execution:
- Rating distribution
- Average ratings over time
- Histogram of ratings per user
- Histogram of ratings per movie
- Genre popularity bar chart
- User demographics charts
- A/B testing results

## Using the Recommendation API

The case study includes a Flask API for serving recommendations. After running the model training code, you can start the API:

```python
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Load models (these should be created by running the full case study code)
# svd_model = joblib.load('models/svd_model.pkl')
# cosine_sim = np.load('models/cosine_sim_matrix.npy')
# train_matrix = pd.read_pickle('models/train_matrix.pkl')

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    # API implementation is included in the case study code
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Additional Resources

If you want to explore larger MovieLens datasets for more robust models:

1. **MovieLens 1M Dataset**: Contains 1 million ratings from 6,000 users on 4,000 movies
   - https://grouplens.org/datasets/movielens/1m/

2. **MovieLens 25M Dataset**: Contains 25 million ratings from 162,000 users on 62,000 movies
   - https://grouplens.org/datasets/movielens/25m/

3. **The Netflix Prize Dataset**: A larger dataset used for the famous Netflix recommendation competition
   - Available through academic channels

4. **IMDb Datasets**: Complementary movie metadata that can enhance content-based filtering
   - https://www.imdb.com/interfaces/

## Key Components of the Recommendation System

The case study covers multiple recommendation approaches:

1. **Memory-based collaborative filtering**: Finds similar users to make recommendations
2. **Model-based collaborative filtering (SVD)**: Uses matrix factorization to find latent factors
3. **Content-based filtering**: Recommends movies with similar attributes
4. **Hybrid approach**: Combines collaborative and content-based methods
5. **Cold start handling**: Special recommendations for new users

By following this guide, you'll have everything set up to build a complete recommendation system with multiple approaches and evaluation methods.