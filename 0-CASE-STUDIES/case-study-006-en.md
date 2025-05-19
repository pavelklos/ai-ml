# Case Study: End-to-End ML Project - Movie Recommendation System

## 1. Problem Definition and Business Context

Recommendation systems are a cornerstone of modern online platforms, helping users discover content they might enjoy among vast catalogs of options. In this case study, we'll build a movie recommendation system from the ground up.

```python
"""
Project: MovieLens Recommendation System

Business Problem: A streaming platform needs to improve user engagement by providing
personalized movie recommendations. Users are overwhelmed by too many choices, 
leading to decreased engagement.

Key Business Questions:
1. What movies should we recommend to each user?
2. How can we effectively handle new users with limited viewing history?
3. How can we incorporate both user preferences and movie attributes?

Success Metrics:
- Recommendation relevance (measured by predicted vs. actual ratings)
- User engagement increase
- Diversity of recommendations
"""
```

## 2. Dataset and Exploration

For this project, we'll use the MovieLens dataset, a popular benchmark for recommendation systems.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the MovieLens dataset
# We'll use the 100K dataset which contains 100,000 ratings from 943 users on 1,682 movies
ratings_file = 'ml-100k/u.data'
item_file = 'ml-100k/u.item'
user_file = 'ml-100k/u.user'

# Define column names
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 
               'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
               'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
               'Thriller', 'War', 'Western']
users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

# Load data
ratings = pd.read_csv(ratings_file, sep='\t', names=ratings_cols)
movies = pd.read_csv(item_file, sep='|', encoding='latin-1', names=movies_cols)
users = pd.read_csv(user_file, sep='|', names=users_cols)

print(f"Ratings data shape: {ratings.shape}")
print(f"Movies data shape: {movies.shape}")
print(f"Users data shape: {users.shape}")

# Display first few rows of each dataset
print("\nRatings preview:")
print(ratings.head())

print("\nMovies preview:")
print(movies[['movie_id', 'title', 'release_date']].head())

print("\nUsers preview:")
print(users.head())
```

### Exploratory Data Analysis

```python
# Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# Basic statistics of ratings
print("\nRating distribution:")
print(ratings['rating'].describe())

# Plot rating distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=ratings)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Ratings over time
plt.figure(figsize=(12, 6))
ratings.set_index('timestamp')['rating'].resample('M').mean().plot()
plt.title('Average Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()

# Number of ratings per user
user_ratings_count = ratings.groupby('user_id').size()
plt.figure(figsize=(12, 6))
sns.histplot(user_ratings_count, bins=50)
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Users')
plt.show()

# Number of ratings per movie
movie_ratings_count = ratings.groupby('movie_id').size()
plt.figure(figsize=(12, 6))
sns.histplot(movie_ratings_count, bins=50)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Count of Movies')
plt.show()

# Get genre information
genre_columns = movies.columns[5:].tolist()

# Calculate genre popularity
genre_popularity = movies[genre_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 7))
genre_popularity.plot(kind='bar')
plt.title('Movie Count by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()

# Analyze user demographics
plt.figure(figsize=(10, 6))
sns.countplot(y='occupation', data=users, order=users['occupation'].value_counts().index)
plt.title('User Count by Occupation')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(users['age'], bins=20)
plt.title('Distribution of User Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```

## 3. Data Preprocessing for Recommendation Systems

```python
# Create a user-item matrix (sparse representation of user ratings)
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')
print("\nUser-Item Matrix Shape:", user_item_matrix.shape)
print("\nSample of User-Item Matrix (first 5 users, first 5 movies):")
print(user_item_matrix.iloc[:5, :5])

# Check matrix sparsity
total_elements = user_item_matrix.shape[0] * user_item_matrix.shape[1]
non_zero_elements = user_item_matrix.count().sum()
sparsity = (total_elements - non_zero_elements) / total_elements
print(f"\nMatrix Sparsity: {sparsity:.4f} ({sparsity*100:.2f}% of elements are missing)")

# Create movie features
# Extract year from title (usually in format "Movie Name (Year)")
def extract_year(title):
    year = title.strip()[-5:-1]
    try:
        year = int(year)
        if 1900 <= year <= 2023:
            return year
        else:
            return np.nan
    except:
        return np.nan

movies['year'] = movies['title'].apply(extract_year)

# Create a genre feature vector for each movie
movies['genre_vector'] = movies[genre_columns].values.tolist()

# Normalize user ratings (center around each user's mean)
user_mean = ratings.groupby('user_id')['rating'].mean()
ratings_normalized = ratings.copy()
for user in ratings['user_id'].unique():
    user_mask = ratings_normalized['user_id'] == user
    ratings_normalized.loc[user_mask, 'rating_normalized'] = \
        ratings_normalized.loc[user_mask, 'rating'] - user_mean[user]

print("\nSample of normalized ratings:")
print(ratings_normalized[['user_id', 'movie_id', 'rating', 'rating_normalized']].head(10))

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# For collaborative filtering, we split by ratings not by users
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

print(f"\nTraining data: {train_data.shape[0]} ratings")
print(f"Testing data: {test_data.shape[0]} ratings")
```

## 4. Building Recommendation Models

### Model 1: Memory-Based Collaborative Filtering

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def memory_based_cf(ratings_matrix, user_id, k=10):
    """
    Predicts ratings for a user using user-based collaborative filtering.
    
    Parameters:
    -----------
    ratings_matrix : DataFrame
        User-item ratings matrix
    user_id : int
        ID of the user to make predictions for
    k : int
        Number of similar users to consider
        
    Returns:
    --------
    DataFrame
        Predicted ratings for unrated items
    """
    # Get the user's ratings
    user_ratings = ratings_matrix.loc[user_id]
    
    # Find similar users based on common ratings
    similarity_scores = []
    
    for other_user in ratings_matrix.index:
        if other_user == user_id:
            continue
            
        # Get items rated by both users
        common_items = user_ratings.dropna().index.intersection(ratings_matrix.loc[other_user].dropna().index)
        
        if len(common_items) < 5:  # Need enough common ratings
            continue
            
        # Calculate similarity
        user_vector = ratings_matrix.loc[user_id, common_items].values
        other_vector = ratings_matrix.loc[other_user, common_items].values
        
        similarity = np.corrcoef(user_vector, other_vector)[0, 1]
        
        if not np.isnan(similarity):
            similarity_scores.append((other_user, similarity))
    
    # Get the k most similar users
    most_similar_users = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:k]
    
    # Predict ratings for unrated items
    unrated_items = user_ratings[user_ratings.isna()].index
    predicted_ratings = {}
    
    for item in unrated_items:
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user, similarity in most_similar_users:
            # Check if the similar user has rated this item
            if not np.isnan(ratings_matrix.loc[similar_user, item]):
                weighted_sum += ratings_matrix.loc[similar_user, item] * similarity
                similarity_sum += abs(similarity)
        
        if similarity_sum > 0:
            predicted_ratings[item] = weighted_sum / similarity_sum
    
    return predicted_ratings

# Create a ratings matrix for training
train_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')

# Test the collaborative filtering function on a sample user
sample_user = 1
predicted_ratings = memory_based_cf(train_matrix, sample_user, k=20)

# Get top 5 movie recommendations for this user
top_recommendations = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 recommended movies for User", sample_user)
for movie_id, predicted_rating in top_recommendations:
    movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print(f"Movie: {movie_title}, Predicted Rating: {predicted_rating:.2f}")
```

### Model 2: Model-Based Collaborative Filtering with Matrix Factorization

```python
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_split

# Set up Surprise dataset
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Split into training and testing
trainset, testset = surprise_split(surprise_data, test_size=0.2, random_state=42)

# Train an SVD model (Matrix Factorization)
svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)
svd_model.fit(trainset)

# Make predictions on the test set
test_predictions = svd_model.test(testset)

# Evaluate the model
rmse = accuracy.rmse(test_predictions)
mae = accuracy.mae(test_predictions)

print(f"\nSVD Model Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Generate top recommendations for a sample user
sample_user = 1
user_seen_movies = set(ratings[ratings['user_id'] == sample_user]['movie_id'])
all_movies = set(movies['movie_id'])
unseen_movies = all_movies - user_seen_movies

# Predict ratings for unseen movies
predictions = []
for movie_id in unseen_movies:
    predicted_rating = svd_model.predict(sample_user, movie_id).est
    predictions.append((movie_id, predicted_rating))

# Sort by predicted rating
top_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 SVD model recommendations for User", sample_user)
for movie_id, predicted_rating in top_predictions:
    movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print(f"Movie: {movie_title}, Predicted Rating: {predicted_rating:.2f}")
```

### Model 3: Content-Based Filtering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re

# Create a feature for content-based filtering
# We'll combine genres into a single string
genre_columns = movies.columns[5:24].tolist()  # Get genre column names

# Create a string representation of genres for each movie
def create_genre_string(row):
    genres = []
    for genre in genre_columns:
        if row[genre] == 1:
            genres.append(genre)
    return ' '.join(genres)

# Clean movie titles (remove year and special characters)
def clean_title(title):
    # Remove year
    title = re.sub(r'\(\d{4}\)$', '', title).strip()
    # Remove special characters
    title = re.sub(r'[^\w\s]', '', title)
    return title

movies['genre_string'] = movies.apply(create_genre_string, axis=1)
movies['clean_title'] = movies['title'].apply(clean_title)

# Create combined features
movies['content_features'] = movies['genre_string'] + ' ' + movies['clean_title']

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content_features'])

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Calculate cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a function to get movie recommendations based on content similarity
def get_content_recommendations(movie_id, cosine_sim=cosine_sim):
    # Get the index of the movie
    idx = movies[movies['movie_id'] == movie_id].index[0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 10 similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:11]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return top 10 most similar movies
    return movies.iloc[movie_indices][['movie_id', 'title', 'genre_string']]

# Get content-based recommendations for a popular movie
popular_movie_id = 50  # Star Wars (1977)
movie_title = movies[movies['movie_id'] == popular_movie_id]['title'].values[0]

print(f"\nContent-based recommendations for '{movie_title}':")
content_recommendations = get_content_recommendations(popular_movie_id)
print(content_recommendations)
```

## 5. Building a Hybrid Recommendation System

```python
def hybrid_recommendations(user_id, movie_id_to_compare=None, k_cf=10, content_weight=0.3):
    """
    Generate hybrid recommendations using both collaborative filtering and content-based filtering
    
    Parameters:
    -----------
    user_id : int
        User ID to generate recommendations for
    movie_id_to_compare : int, optional
        Movie ID to use for content-based recommendations
    k_cf : int
        Number of similar users for collaborative filtering
    content_weight : float
        Weight for content-based recommendations (1 - content_weight for collaborative)
    
    Returns:
    --------
    list
        Top 10 movie recommendations
    """
    # Get collaborative filtering recommendations
    cf_predictions = memory_based_cf(train_matrix, user_id, k=k_cf)
    
    # If no movie_id provided, use the user's highest rated movie
    if movie_id_to_compare is None:
        user_ratings = ratings[ratings['user_id'] == user_id]
        if len(user_ratings) > 0:
            movie_id_to_compare = user_ratings.sort_values('rating', ascending=False)['movie_id'].values[0]
        else:
            # If user has no ratings, use a popular movie
            movie_id_to_compare = 50  # Star Wars
    
    # Get content-based recommendations
    content_recs = get_content_recommendations(movie_id_to_compare)
    content_scores = {row['movie_id']: 5 - i/2 for i, (_, row) in enumerate(content_recs.iterrows())}
    
    # Combine recommendations
    hybrid_scores = {}
    
    # Include all movies from collaborative filtering
    for movie_id, score in cf_predictions.items():
        hybrid_scores[movie_id] = (1 - content_weight) * score
    
    # Include content recommendations
    for movie_id, score in content_scores.items():
        if movie_id in hybrid_scores:
            hybrid_scores[movie_id] += content_weight * score
        else:
            hybrid_scores[movie_id] = content_weight * score
    
    # Remove movies the user has already seen
    user_seen_movies = set(ratings[ratings['user_id'] == user_id]['movie_id'])
    hybrid_scores = {k: v for k, v in hybrid_scores.items() if k not in user_seen_movies}
    
    # Return top 10 recommendations
    top_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return top_recs

# Get hybrid recommendations for our sample user
sample_user = 1
hybrid_recs = hybrid_recommendations(sample_user)

print(f"\nHybrid Recommendations for User {sample_user}:")
for i, (movie_id, score) in enumerate(hybrid_recs, 1):
    movie_title = movies[movies['movie_id'] == movie_id]['title'].values[0]
    print(f"{i}. {movie_title} (Score: {score:.2f})")
```

## 6. Model Evaluation

Let's evaluate our recommendation systems using relevant metrics for this task.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def evaluate_predictions(predictions, actual):
    """
    Evaluate prediction accuracy using RMSE and MAE
    """
    # Filter predictions to only include items in actual
    common_items = set(predictions.keys()) & set(actual.keys())
    
    if len(common_items) == 0:
        return None, None
    
    # Get predictions and actual ratings for common items
    y_pred = [predictions[i] for i in common_items]
    y_true = [actual[i] for i in common_items]
    
    # Calculate metrics
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return rmse, mae

# Create test user-item dictionary
test_user_item = {}
for _, row in test_data.iterrows():
    user = row['user_id']
    movie = row['movie_id']
    rating = row['rating']
    
    if user not in test_user_item:
        test_user_item[user] = {}
    test_user_item[user][movie] = rating

# Evaluate collaborative filtering
cf_results = []
for user in test_user_item:
    # Skip users not in training set
    if user not in train_matrix.index:
        continue
        
    # Get predictions
    predicted = memory_based_cf(train_matrix, user)
    
    # Evaluate
    metrics = evaluate_predictions(predicted, test_user_item[user])
    if metrics[0] is not None:
        cf_results.append(metrics)

# Calculate average metrics
cf_rmse = np.mean([r[0] for r in cf_results])
cf_mae = np.mean([r[1] for r in cf_results])

print("\nCollaborative Filtering Performance:")
print(f"Average RMSE: {cf_rmse:.4f}")
print(f"Average MAE: {cf_mae:.4f}")

# Also evaluate recommendation diversity and novelty
def calculate_diversity(recommendations_list, genre_matrix):
    """
    Calculate diversity of recommendations based on genre coverage
    """
    # Extract movie IDs from recommendations
    movie_ids = [rec[0] for rec in recommendations_list]
    
    # Get genre vectors for these movies
    genre_vectors = [movies[movies['movie_id'] == mid]['genre_vector'].values[0] for mid in movie_ids]
    
    # Calculate unique genres covered
    unique_genres = set()
    for vector in genre_vectors:
        for i, has_genre in enumerate(vector):
            if has_genre == 1:
                unique_genres.add(genre_columns[i])
    
    # Calculate diversity score (percentage of all genres covered)
    diversity_score = len(unique_genres) / len(genre_columns)
    
    return diversity_score

# Calculate diversity for hybrid recommendations
hybrid_diversity = calculate_diversity(hybrid_recs, movies[genre_columns])

print(f"\nHybrid Recommendation Diversity Score: {hybrid_diversity:.2f}")
print(f"(Coverage of {int(hybrid_diversity * len(genre_columns))} out of {len(genre_columns)} genres)")
```

## 7. Implementing the Recommendation API

```python
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Save models for later use
joblib.dump(svd_model, 'models/svd_model.pkl')
np.save('models/cosine_sim_matrix.npy', cosine_sim)
train_matrix.to_pickle('models/train_matrix.pkl')

# Create a Flask application
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    """API endpoint to get movie recommendations for a user"""
    try:
        # Get parameters from request
        user_id = int(request.args.get('user_id', 1))
        num_recs = int(request.args.get('num_recs', 5))
        rec_type = request.args.get('type', 'hybrid')  # 'cf', 'content', 'svd', or 'hybrid'
        
        # Load necessary data
        svd_model = joblib.load('models/svd_model.pkl')
        cosine_sim = np.load('models/cosine_sim_matrix.npy')
        train_matrix = pd.read_pickle('models/train_matrix.pkl')
        
        # Generate recommendations based on type
        if rec_type == 'cf':
            # Collaborative filtering
            predictions = memory_based_cf(train_matrix, user_id, k=20)
            recs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:num_recs]
            
        elif rec_type == 'content':
            # Content-based: first get user's highest rated movie
            user_ratings = ratings[ratings['user_id'] == user_id]
            if len(user_ratings) > 0:
                movie_id = user_ratings.sort_values('rating', ascending=False)['movie_id'].values[0]
                content_recs = get_content_recommendations(movie_id, cosine_sim)
                recs = [(row['movie_id'], 5 - i/2) for i, (_, row) in enumerate(content_recs.iterrows())][:num_recs]
            else:
                return jsonify({'error': 'User has no ratings for content-based recommendations'})
        
        elif rec_type == 'svd':
            # SVD model
            user_seen_movies = set(ratings[ratings['user_id'] == user_id]['movie_id'])
            all_movies = set(movies['movie_id'])
            unseen_movies = all_movies - user_seen_movies
            
            predictions = []
            for movie_id in unseen_movies:
                predicted_rating = svd_model.predict(user_id, movie_id).est
                predictions.append((movie_id, predicted_rating))
            
            recs = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recs]
            
        else:  # hybrid
            recs = hybrid_recommendations(user_id, k_cf=20)[:num_recs]
        
        # Format recommendations
        recommendations = []
        for movie_id, score in recs:
            movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
            recommendations.append({
                'movie_id': int(movie_id),
                'title': movie_info['title'],
                'genres': [genre for genre, has_genre in zip(genre_columns, movie_info[genre_columns]) if has_genre == 1],
                'score': float(score)
            })
        
        return jsonify({
            'user_id': user_id,
            'recommendation_type': rec_type,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Example of running the API (in production you'd use a proper WSGI server)
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
```

## 8. Handling Cold Start Problems

A common challenge in recommendation systems is dealing with new users or items.

```python
def recommend_for_new_user(age, gender, occupation, favorite_genres=None, k=10):
    """
    Generate recommendations for a new user with no rating history
    
    Parameters:
    -----------
    age : int
        User's age
    gender : str
        User's gender ('M' or 'F')
    occupation : str
        User's occupation
    favorite_genres : list, optional
        List of user's favorite genres
    k : int
        Number of recommendations to return
    
    Returns:
    --------
    list
        Top k movie recommendations
    """
    # Find similar users based on demographics
    similar_users = users[
        (users['age'].between(age - 5, age + 5)) &
        (users['gender'] == gender) &
        (users['occupation'] == occupation)
    ]['user_id'].tolist()
    
    if len(similar_users) == 0:
        # If no similar users found, relax criteria
        similar_users = users[
            (users['gender'] == gender) &
            (users['occupation'] == occupation)
        ]['user_id'].tolist()
    
    # Get movies highly rated by similar users
    similar_users_ratings = ratings[ratings['user_id'].isin(similar_users)]
    movie_avg_ratings = similar_users_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    
    # Filter movies with enough ratings
    popular_movies = movie_avg_ratings[movie_avg_ratings['count'] >= 5].sort_values('mean', ascending=False)
    
    # If user provided favorite genres, filter recommendations
    if favorite_genres:
        # Get movies in favorite genres
        genre_movies = set()
        for genre in favorite_genres:
            if genre in genre_columns:
                genre_movies.update(movies[movies[genre] == 1]['movie_id'])
        
        # Filter popular movies by genre
        genre_filtered = popular_movies[popular_movies.index.isin(genre_movies)]
        
        # If we have enough genre-filtered movies, use those
        if len(genre_filtered) >= k:
            top_movies = genre_filtered.head(k)
        else:
            # Otherwise supplement with generally popular movies
            top_genre = genre_filtered
            remaining = popular_movies[~popular_movies.index.isin(genre_filtered.index)].head(k - len(genre_filtered))
            top_movies = pd.concat([top_genre, remaining])
    else:
        # Without genre preferences, just return popular movies
        top_movies = popular_movies.head(k)
    
    # Get movie details for recommendations
    recommendations = []
    for movie_id, row in top_movies.iterrows():
        movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
        recommendations.append({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'average_rating': row['mean'],
            'num_ratings': row['count']
        })
    
    return recommendations

# Example usage for a new user
new_user_recs = recommend_for_new_user(
    age=30, 
    gender='M', 
    occupation='programmer', 
    favorite_genres=['Sci-Fi', 'Action', 'Adventure']
)

print("\nRecommendations for new user:")
for i, rec in enumerate(new_user_recs, 1):
    print(f"{i}. {rec['title']} - Avg Rating: {rec['average_rating']:.2f} ({rec['num_ratings']} ratings)")
```

## 9. Recommendation System Deployment Architecture

```python
# Pseudocode for a production recommendation system architecture

"""
1. Data Ingestion Pipeline
   - Collect user interaction data (views, ratings, clicks)
   - Store in appropriate database (e.g., PostgreSQL for structured data, MongoDB for events)
   - ETL process to transform raw data into features

2. Offline Model Training
   - Schedule regular retraining (e.g., daily or weekly)
   - Train models on historical data
   - Evaluate models with various metrics
   - Save models to model registry
   
3. Feature Store
   - User features (demographics, behavior patterns)
   - Item features (content metadata, popularity metrics)
   - Interaction features (historical ratings, clicks)
   
4. Model Serving
   - Real-time API for user recommendations
   - Caching layer for popular items and users
   - A/B testing infrastructure for model variants
   
5. Feedback Loop
   - Collect user interactions with recommendations
   - Update user profiles based on new interactions
   - Analyze recommendation performance
"""

# Example architecture diagram (ASCII art)
print("""
Recommendation System Architecture:

                                   +---------------+
                                   |   User Data   |
                                   +-------+-------+
                                           |
                                           v
+----------------+    +-------------+    +--------+    +----------------+
| User Interface | <- | API Gateway | <- | Models | <- | Feature Store  |
+----------------+    +-------------+    +--------+    +----------------+
                                           ^              ^
                                           |              |
                                     +-----+------+       |
                                     | Model      |       |
                                     | Registry   |       |
                                     +-----+------+       |
                                           ^              |
                                           |              |
                                     +-----+------+    +--+-------------+
                                     | Training   | <- | Data Pipeline  |
                                     | Pipeline   |    | (ETL)          |
                                     +-----------+     +----------------+
                                           ^                   ^
                                           |                   |
                                           |                   |
                                    +------+-----------------+-+
                                    |      Data Storage      |
                                    +-----------------------+
""")
```

## 10. A/B Testing for Recommendations

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Simulate A/B testing for recommendation algorithms
def simulate_ab_test(num_users=1000, days=14):
    """
    Simulate an A/B test comparing two recommendation algorithms
    
    Parameters:
    -----------
    num_users : int
        Number of users in each group (A and B)
    days : int
        Number of days to run the test
        
    Returns:
    --------
    DataFrame
        Results of the A/B test
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create user groups
    users = pd.DataFrame({
        'user_id': range(1, num_users * 2 + 1),
        'group': ['A'] * num_users + ['B'] * num_users
    })
    
    # Baseline metrics for both groups
    # Group A: Collaborative filtering
    # Group B: Hybrid recommendations
    
    # Define engagement metrics with some randomness
    # Group B (hybrid) has slightly better metrics
    metrics = {
        'A': {
            'click_rate': 0.12,  # 12% click-through rate
            'rating': 3.8,       # Average rating
            'watch_time': 25,    # Average watch time (minutes)
            'conversion': 0.08   # 8% conversion rate
        },
        'B': {
            'click_rate': 0.15,  # 15% click-through rate
            'rating': 4.0,       # Average rating
            'watch_time': 30,    # Average watch time (minutes)
            'conversion': 0.10   # 10% conversion rate
        }
    }
    
    # Generate daily results
    results = []
    
    for day in range(1, days + 1):
        for group in ['A', 'B']:
            # Add some daily variation
            daily_factor = np.random.normal(1.0, 0.05)  # 5% daily variance
            
            # Number of users active that day (70-90% of users are active each day)
            active_users = int(np.random.uniform(0.7, 0.9) * num_users)
            
            # Calculate metrics with randomness
            clicks = np.random.binomial(active_users, metrics[group]['click_rate'] * daily_factor)
            avg_rating = np.random.normal(metrics[group]['rating'], 0.2)
            avg_watch_time = np.random.normal(metrics[group]['watch_time'], 3)
            conversions = np.random.binomial(clicks, metrics[group]['conversion'] * daily_factor)
            
            results.append({
                'day': day,
                'group': group,
                'active_users': active_users,
                'clicks': clicks,
                'click_rate': clicks / active_users,
                'avg_rating': avg_rating,
                'avg_watch_time': avg_watch_time,
                'conversions': conversions,
                'conversion_rate': conversions / active_users if active_users > 0 else 0
            })
    
    return pd.DataFrame(results)

# Run the A/B test simulation
ab_results = simulate_ab_test()

# Analyze results
print("\nA/B Testing Results Summary:")
summary = ab_results.groupby('group').agg({
    'active_users': 'mean',
    'click_rate': 'mean',
    'avg_rating': 'mean',
    'avg_watch_time': 'mean',
    'conversion_rate': 'mean'
}).reset_index()

print(summary)

# Visualize results
metrics_to_plot = ['click_rate', 'avg_rating', 'avg_watch_time', 'conversion_rate']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    # Extract data for each group
    group_a = ab_results[ab_results['group'] == 'A'][metric]
    group_b = ab_results[ab_results['group'] == 'B'][metric]
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    
    # Plot
    axes[i].bar(['Group A (Collaborative)', 'Group B (Hybrid)'], 
                [group_a.mean(), group_b.mean()],
                yerr=[group_a.std(), group_b.std()],
                capsize=10)
    axes[i].set_title(f'{metric.replace("_", " ").title()}\np-value: {p_value:.4f}')
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate the overall business impact
improvement = (summary[summary['group'] == 'B']['conversion_rate'].values[0] / 
              summary[summary['group'] == 'A']['conversion_rate'].values[0] - 1) * 100

print(f"\nOverall conversion rate improvement: {improvement:.2f}%")
print("Based on this A/B test, we would recommend implementing the hybrid recommendation system.")
```

## 11. Ethical Considerations and Bias in Recommendation Systems

```python
# Analyze potential biases in our recommendation system

def analyze_recommendation_diversity():
    """
    Analyze recommendation diversity across different user demographics
    """
    # Sample users from different demographic groups
    age_groups = {
        'young': users[users['age'] < 25]['user_id'].tolist(),
        'middle': users[(users['age'] >= 25) & (users['age'] < 45)]['user_id'].tolist(),
        'older': users[users['age'] >= 45]['user_id'].tolist()
    }
    
    gender_groups = {
        'male': users[users['gender'] == 'M']['user_id'].tolist(),
        'female': users[users['gender'] == 'F']['user_id'].tolist()
    }
    
    # Sample 20 users from each group
    np.random.seed(42)
    sampled_users = {
        'age': {k: np.random.choice(v, min(20, len(v)), replace=False) for k, v in age_groups.items()},
        'gender': {k: np.random.choice(v, min(20, len(v)), replace=False) for k, v in gender_groups.items()}
    }
    
    # Generate recommendations for sampled users
    results = {
        'age': {k: [] for k in age_groups},
        'gender': {k: [] for k in gender_groups}
    }
    
    for demo_type, groups in sampled_users.items():
        for group_name, user_ids in groups.items():
            group_genres = []
            group_years = []
            
            for user_id in user_ids:
                try:
                    # Get hybrid recommendations for user
                    recs = hybrid_recommendations(user_id, k_cf=10)
                    
                    # Extract genre and year information
                    for movie_id, _ in recs:
                        movie_info = movies[movies['movie_id'] == movie_id].iloc[0]
                        
                        # Add genres
                        user_genres = [g for g, has_g in zip(genre_columns, movie_info[genre_columns]) if has_g == 1]
                        group_genres.extend(user_genres)
                        
                        # Add year if available
                        if not np.isnan(movie_info.get('year', np.nan)):
                            group_years.append(movie_info['year'])
                except:
                    # Skip users with errors
                    continue
            
            # Calculate genre distribution
            genre_dist = pd.Series(group_genres).value_counts() / len(group_genres)
            
            # Calculate year distribution
            year_mean = np.mean(group_years) if group_years else np.nan
            year_std = np.std(group_years) if len(group_years) > 1 else np.nan
            
            results[demo_type][group_name] = {
                'genre_distribution': genre_dist,
                'top_3_genres': genre_dist.nlargest(3).index.tolist(),
                'year_mean': year_mean,
                'year_std': year_std
            }
    
    return results

# Run the analysis
diversity_results = analyze_recommendation_diversity()

# Display results
print("\nRecommendation Diversity Analysis:")

print("\nBy Age Group:")
for age_group, data in diversity_results['age'].items():
    print(f"  {age_group.capitalize()} users:")
    print(f"    Top genres: {', '.join(data['top_3_genres'])}")
    print(f"    Avg. movie year: {data['year_mean']:.1f} ± {data['year_std']:.1f}")

print("\nBy Gender:")
for gender, data in diversity_results['gender'].items():
    print(f"  {gender.capitalize()} users:")
    print(f"    Top genres: {', '.join(data['top_3_genres'])}")
    print(f"    Avg. movie year: {data['year_mean']:.1f} ± {data['year_std']:.1f}")

# Discuss potential ethical issues
print("""
\nEthical Considerations in Recommendation Systems:

1. Filter Bubbles
   - Recommendation systems can create "echo chambers" where users only see content similar 
     to what they've already consumed
   - To mitigate: Introduce diversity metrics and occasionally recommend items outside the user's typical preferences

2. Popularity Bias
   - Popular items tend to get recommended more often, creating a rich-get-richer effect
   - To mitigate: Include novelty and serendipity metrics in the recommendation algorithm

3. Data Privacy
   - Recommendation systems require collecting user behavior data
   - To mitigate: Use differential privacy techniques, be transparent about data collection, allow opt-out

4. Fairness Across Groups
   - Recommendation quality may vary across demographic groups
   - To mitigate: Regularly analyze recommendation performance across different user segments

5. Transparency
   - Users may not understand why certain items are recommended to them
   - To mitigate: Provide explanations for recommendations, allow users to provide feedback
""")
```

## 12. Key Learnings and Best Practices

This case study on building a movie recommendation system has demonstrated several key aspects of the machine learning lifecycle:

### 1. Problem Definition and Data Understanding
- Recommendation systems solve the problem of information overload by helping users discover relevant content
- The MovieLens dataset provided user-item interactions and metadata for modeling
- Understanding data sparsity is crucial in recommendation systems (most users rate only a small fraction of items)

### 2. Feature Engineering for Recommendations
- Creating user-item matrices for collaborative filtering
- Extracting metadata features (genres, year) for content-based filtering
- Normalizing ratings to account for individual user biases

### 3. Model Development
- Memory-based collaborative filtering uses rating patterns to find similar users or items
- Model-based approaches like SVD (matrix factorization) capture latent factors
- Content-based filtering recommends items with similar attributes
- Hybrid approaches combine multiple recommendation strategies for better results

### 4. Evaluation Methods
- RMSE and MAE measure prediction accuracy
- Diversity metrics assess recommendation variety
- A/B testing determines real-world performance

### 5. Handling Common Challenges
- Cold start problems for new users or items
- Scalability issues with large user-item matrices
- Balancing recommendation accuracy with diversity

### 6. Deployment Considerations
- Creating an API for serving recommendations
- Designing a system architecture that can scale
- Implementing feedback loops to continuously improve recommendations

### 7. Ethical Considerations
- Monitoring and addressing potential biases
- Ensuring recommendation diversity
- Protecting user privacy while collecting interaction data

### Best Practices:
1. Combine multiple recommendation approaches in a hybrid system
2. Regularly retrain models to incorporate new user interactions
3. Include diversity metrics alongside accuracy metrics
4. Design for the cold-start problem from the beginning
5. Implement A/B testing to validate recommendation quality
6. Monitor recommendation biases across different user segments
7. Create a feedback loop to continuously improve the system

This case study provides a foundation for understanding and implementing recommendation systems, one of the most widely used applications of machine learning in consumer-facing products.