# Getting Started with the Three ML Case Studies

This guide explains how to set up your environment and obtain the necessary datasets for the three machine learning case studies: Housing Price Prediction, Customer Churn Prediction, and Customer Segmentation.

## Required Python Packages

First, install the required Python libraries for all three case studies:

```python
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset Information

These case studies use three different datasets, each appropriate for the machine learning task being demonstrated.

### Case Study 1: Housing Price Prediction (Regression)

The first case study uses the California Housing dataset, which is built into scikit-learn.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Save to CSV if you need a local copy
df = X.copy()
df['PRICE'] = y
df.to_csv('california_housing.csv', index=False)
print("Dataset saved to 'california_housing.csv'")
```

### Case Study 2: Customer Churn Prediction (Classification)

The second case study uses the Telco Customer Churn dataset from Kaggle.

#### Option 1: Download Directly from Kaggle

1. Visit [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file
3. Place it in your project directory

#### Option 2: Download Using Python (requires Kaggle API)

```python
import os
import kaggle

# Ensure you have set up your Kaggle API credentials
# Create ~/.kaggle/kaggle.json with your API key from kaggle.com/account

# Create directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the Telco Customer Churn dataset
kaggle.api.dataset_download_files('blastchar/telco-customer-churn', path='data', unzip=True)

print("Telco Customer Churn dataset downloaded to data directory")
```

### Case Study 3: Customer Segmentation (Clustering)

The third case study uses the Mall Customer Segmentation dataset from Kaggle.

#### Option 1: Download Directly from Kaggle

1. Visit [Mall Customer Segmentation dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
2. Download the `Mall_Customers.csv` file
3. Place it in your project directory

#### Option 2: Download Using Python (requires Kaggle API)

```python
import os
import kaggle

# Create directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download the Mall Customer Segmentation dataset
kaggle.api.dataset_download_files('vjchoudhary7/customer-segmentation-tutorial-in-python', path='data', unzip=True)

print("Mall Customer Segmentation dataset downloaded to data directory")
```

## Project Structure Setup

Create a directory structure for the case studies:

```python
import os

# Create project directories
directories = [
    'data',
    'models',
    'visualizations',
    'results'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## About the Datasets

### California Housing Dataset

The California Housing dataset contains information about housing in California from the 1990 census:

- **Features:**
  - MedInc: Median income in block group
  - HouseAge: Median house age in block group
  - AveRooms: Average number of rooms per household
  - AveBedrms: Average number of bedrooms per household
  - Population: Block group population
  - AveOccup: Average number of household members
  - Latitude: Block group latitude
  - Longitude: Block group longitude

- **Target:**
  - PRICE: Median house value in $100,000s

### Telco Customer Churn Dataset

This dataset contains information about telecom customers and whether they churned:

- **Features:**
  - Demographics (gender, senior citizen status, etc.)
  - Account information (tenure, contract type, payment method)
  - Services subscribed (phone, internet, tech support, etc.)
  - Billing information (monthly charges, total charges)

- **Target:**
  - Churn: Whether the customer left the company (Yes/No)

### Mall Customer Segmentation Dataset

This dataset contains basic information about mall customers for segmentation:

- **Features:**
  - CustomerID: Unique identifier for each customer
  - Gender: Gender of the customer
  - Age: Age of the customer
  - Annual Income (k$): Annual income of the customer
  - Spending Score (1-100): Score assigned by the mall based on customer behavior and spending nature

## Verifying Your Setup

Run these checks to verify each dataset is properly loaded:

### Case Study 1: Housing Price Prediction

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
try:
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    print("California Housing dataset loaded successfully!")
    print(f"Shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Target range: {y.min()} to {y.max()}")
    print(f"Sample data:\n{X.head()}")
    
except Exception as e:
    print(f"Error loading California Housing dataset: {e}")
```

### Case Study 2: Customer Churn Prediction

```python
import pandas as pd

# Try to load the dataset
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Alternative path if downloaded using the Kaggle API
    if 'Churn' not in df.columns and os.path.exists('data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
        df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print("Telco Customer Churn dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    print(f"Sample data:\n{df.head()}")
    
except Exception as e:
    print(f"Error loading Telco Customer Churn dataset: {e}")
    print("Please make sure you've downloaded the dataset from Kaggle.")
```

### Case Study 3: Customer Segmentation

```python
import pandas as pd

# Try to load the dataset
try:
    df = pd.read_csv('Mall_Customers.csv')
    # Alternative path if downloaded using the Kaggle API
    if 'Annual Income (k$)' not in df.columns and os.path.exists('data/Mall_Customers.csv'):
        df = pd.read_csv('data/Mall_Customers.csv')
    
    print("Mall Customer Segmentation dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Gender distribution:\n{df['Gender'].value_counts()}")
    print(f"Sample data:\n{df.head()}")
    
except Exception as e:
    print(f"Error loading Mall Customer Segmentation dataset: {e}")
    print("Please make sure you've downloaded the dataset from Kaggle.")
```

## Expected Outputs from the Case Studies

When running the complete code, the following outputs will be generated:

### Case Study 1: Housing Price Prediction
- Feature correlation heatmap
- House price distribution plot
- Linear Regression and Random Forest models
- Feature importance visualization
- Actual vs. predicted prices plot

### Case Study 2: Customer Churn Prediction
- Churn distribution pie chart
- Contract type vs. churn bar chart
- Logistic Regression and Random Forest models
- Confusion matrices
- ROC curves
- Feature importance visualization

### Case Study 3: Customer Segmentation
- Age and gender distribution visualizations
- Income vs. spending score scatter plot
- Elbow method and silhouette score plots
- K-Means cluster visualization
- Hierarchical clustering dendrogram
- 3D visualization of customer segments

These case studies provide hands-on practice with the three main types of machine learning problems: regression, classification, and clustering. Each follows the complete data science workflow from exploration to model evaluation and business insights.