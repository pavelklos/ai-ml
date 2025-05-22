# Getting Started with the Movie Review Sentiment Analysis Project

This guide explains how to set up the necessary environment and datasets for the Movie Review Sentiment Analysis case study.

## Required Python Packages

The project relies on several Python libraries. You can install them using pip:

```python
pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow joblib flask
```

## Dataset Information

This project uses the Movie Reviews dataset from the NLTK (Natural Language Toolkit) library. Unlike many ML projects, you don't need to download a separate CSV file - the dataset is built into NLTK.

### Option 1: Downloading the NLTK Data Directly

You can download all the required NLTK data components using Python:

```python
import nltk

# Download required NLTK resources
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

print("Dataset and required NLTK resources successfully downloaded.")
```

### Option 2: Manual Download through NLTK Downloader

You can also use NLTK's built-in downloader interface:

```python
import nltk
nltk.download()  # This will open the NLTK downloader GUI
```

In the downloader window, navigate to the "Corpora" tab and select:
- movie_reviews
- punkt
- stopwords
- wordnet
- vader_lexicon

Then click "Download" to install these resources.

## Verifying the Setup

To verify that the dataset is correctly installed, run this simple check:

```python
import nltk
from nltk.corpus import movie_reviews

# Check if we can access the movie reviews
positive_fileids = movie_reviews.fileids('pos')
negative_fileids = movie_reviews.fileids('neg')

print(f"Number of positive reviews: {len(positive_fileids)}")
print(f"Number of negative reviews: {len(negative_fileids)}")

# Display a sample review
if positive_fileids:
    sample_review = movie_reviews.raw(positive_fileids[0])
    print(f"\nSample positive review (first 300 characters):")
    print(sample_review[:300] + "...")
```

## About the Movie Reviews Dataset

The NLTK Movie Reviews dataset contains:

- 2,000 movie reviews total
- 1,000 positive reviews
- 1,000 negative reviews
- Reviews taken from the Internet Movie Database (IMDb)
- Pre-categorized by sentiment (positive/negative)
- Plain text format

This dataset was originally collected by Bo Pang and Lillian Lee and has been widely used in sentiment analysis research.

## Model Output Files

When running the complete code, the following files will be generated:

- `sentiment_analysis_model.pkl`: The trained and tuned sentiment classification model
- `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer used for text feature extraction

These files are required for the deployment section of the project, where a Flask API is created to serve predictions.

## Optional: Alternative Movie Review Datasets

If you want to explore larger or more diverse datasets, consider:

1. **Large Movie Review Dataset (IMDB)**: 50,000 reviews with a more challenging binary sentiment classification task
   - Available at: https://ai.stanford.edu/~amaas/data/sentiment/

2. **Rotten Tomatoes Dataset**: Contains critic and audience reviews from Rotten Tomatoes
   - Available via Kaggle: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

To use these alternative datasets, you'll need to modify the data loading code accordingly.

## Run the Project

With the environment set up and data downloaded, you can now run the complete case study code to:
- Process the movie reviews
- Extract features
- Train multiple sentiment analysis models
- Evaluate performance
- Deploy a prediction API

The code is organized in a logical sequence through the notebook sections, allowing you to understand the complete machine learning workflow.