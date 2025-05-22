# Getting Started with the Diabetes Progression Prediction Project

This guide explains how to set up your environment and obtain the necessary dataset for the Diabetes Progression Prediction case study.

## Required Python Packages

First, install the required Python libraries:

```python
pip install numpy pandas matplotlib seaborn scikit-learn shap joblib flask
```

## Dataset Information

This project uses the diabetes dataset built into scikit-learn, which contains data for diabetes patients. Unlike many other ML projects, you don't need to download a separate file - the dataset is directly accessible through the scikit-learn package.

### Accessing the Diabetes Dataset

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Create a DataFrame with feature names
feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['disease_progression'] = y

print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of data:")
print(df.head())

# Save to CSV if you need a local copy
df.to_csv('data/diabetes_dataset.csv', index=False)
print("Dataset saved to 'data/diabetes_dataset.csv'")
```

## About the Diabetes Dataset

The scikit-learn diabetes dataset contains:

- **442 patients**
- **10 feature variables** (age, sex, bmi, bp, and 6 blood serum measurements)
- **Target variable**: Disease progression measure one year after baseline

All features have been standardized to have zero mean and unit variance. The dataset is commonly used for regression tasks in healthcare applications.

The feature descriptions are:
- **age**: Age of patient (standardized)
- **sex**: Gender of patient (standardized)
- **bmi**: Body Mass Index (standardized)
- **bp**: Average blood pressure (standardized)
- **s1-s6**: Six blood serum measurements (standardized)

## Project Structure Setup

Create the necessary directory structure for the project:

```python
import os

# Create project directories
directories = [
    'data',
    'models',
    'visualizations',
    'notebooks',
    'src'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## Expected Output Files

When running the complete code, the following files will be generated:

### Models
- `diabetes_progression_model.pkl`: The trained and optimized ML model
- `diabetes_preprocessing_pipeline.pkl`: The data preprocessing pipeline

### Visualizations
Several visualization files will be created:
- Disease progression distribution
- Correlation matrix
- Feature relationship plots
- Model comparison charts
- Residual plots
- Feature importance visualizations
- SHAP plots (for explainable AI)
- Model monitoring charts
- Clinical impact assessment charts

## Verifying Your Setup

Run this quick check to confirm your environment is set up correctly:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Create a DataFrame 
    feature_names = diabetes.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['disease_progression'] = y
    
    print("Diabetes dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
    # Quick visualization of target variable
    plt.figure(figsize=(8, 5))
    sns.histplot(df['disease_progression'], bins=30, kde=True)
    plt.title('Distribution of Diabetes Progression Measure')
    plt.xlabel('Disease Progression')
    plt.ylabel('Frequency')
    plt.show()
    
    # Check correlation with BMI (a key predictor)
    correlation = df['bmi'].corr(df['disease_progression'])
    print(f"Correlation between BMI and disease progression: {correlation:.3f}")
    
    print("\nEnvironment and dataset verified successfully!")
    
except Exception as e:
    print(f"Setup verification failed: {e}")
```

## Feature Descriptions Dictionary

You'll need this dictionary for the model interpretation section:

```python
# Feature descriptions based on the diabetes dataset documentation
feature_descriptions = {
    'age': 'Age of patient (standardized)',
    'sex': 'Gender of patient (standardized)',
    'bmi': 'Body Mass Index (standardized)',
    'bp': 'Average blood pressure (standardized)',
    's1': 'Total serum cholesterol (standardized)',
    's2': 'Low-density lipoproteins (standardized)',
    's3': 'High-density lipoproteins (standardized)',
    's4': 'Total cholesterol / HDL (standardized)',
    's5': 'Log of serum triglycerides level (standardized)',
    's6': 'Blood sugar level (standardized)',
    'disease_progression': 'Measure of disease progression one year after baseline'
}
```

## Web Application Setup

The case study includes a Flask web application for healthcare providers. To run the app:

```python
from flask import Flask, request, jsonify, render_template_string
import joblib

# Load the model, pipeline, and other required components
# model = joblib.load('diabetes_progression_model.pkl')
# pipeline = joblib.load('diabetes_preprocessing_pipeline.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    # Web app implementation included in the case study code
    pass

@app.route('/predict', methods=['POST'])
def predict():
    # Prediction endpoint implementation included in the case study code
    pass

if __name__ == '__main__':
    app.run(debug=True)
```

## Additional Resources

If you want to explore other diabetes-related datasets for expanded analysis:

1. **UCI Machine Learning Repository - Diabetes Dataset**:  
   https://archive.ics.uci.edu/ml/datasets/diabetes

2. **Pima Indians Diabetes Database**:  
   https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

3. **NHANES (National Health and Nutrition Examination Survey)**:  
   https://www.cdc.gov/nchs/nhanes/index.htm

With this setup, you'll have everything needed to work through the Diabetes Progression Prediction case study, from data analysis to model development, evaluation, and deployment.