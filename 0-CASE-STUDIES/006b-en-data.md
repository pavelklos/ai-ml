# Getting the MovieLens 100K Dataset

The MovieLens 100K dataset is a publicly available dataset that you can download from the GroupLens Research website. Here's how to get it:

## Option 1: Direct Download

1. Visit the GroupLens datasets page: https://grouplens.org/datasets/movielens/

2. Look for the "MovieLens 100K Dataset" and download it (direct link: https://files.grouplens.org/datasets/movielens/ml-100k.zip)

3. Extract the ZIP file to your working directory. This will create a folder called `ml-100k` containing all the necessary files.

## Option 2: Download with Python

You can also download and extract the dataset using Python:

```python
import os
import requests
import zipfile
from io import BytesIO

# Create directory if it doesn't exist
os.makedirs('ml-100k', exist_ok=True)

# Download the MovieLens 100K dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
print("Downloading MovieLens 100K dataset...")
response = requests.get(url)
zipfile_bytes = BytesIO(response.content)

# Extract the ZIP file
with zipfile.ZipFile(zipfile_bytes) as zip_ref:
    zip_ref.extractall("./")

print("Dataset downloaded and extracted successfully!")
```

## Verifying the Dataset

Once downloaded, check if the following files exist in the `ml-100k` directory:
- `u.data` (ratings file)
- `u.item` (movie information file)
- `u.user` (user information file)

These are the key files used in the code example I provided.

## About the MovieLens 100K Dataset

This dataset contains:
- 100,000 ratings (1-5) from 943 users on 1,682 movies
- Demographic information about the users (age, gender, occupation, zip code)
- Information about the movies (title, release date, genres)

With these files in place, the code example should work as expected. Let me know if you encounter any issues or need further assistance!