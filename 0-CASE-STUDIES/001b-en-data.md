# Getting Started with the Customer Churn Prediction Project

This guide explains how to obtain the necessary dataset and set up your environment for the Customer Churn Prediction case study.

## Required Python Packages

First, install the required Python libraries:

```python
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib flask
```

## Dataset Information

This project uses telecom customer data to predict customer churn. There are two options to obtain this dataset:

### Option 1: Download the IBM Telco Customer Churn Dataset

```python
import pandas as pd
import os
import requests
from io import StringIO

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)

# Define the URL for the Telco Customer Churn dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

try:
    # Download the dataset
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Read the CSV
    df = pd.read_csv(StringIO(response.text))
    
    # Save the dataset locally
    df.to_csv('data/raw/telecom_customers.csv', index=False)
    
    print(f"Successfully downloaded and saved the dataset to data/raw/telecom_customers.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print("\nSample data:")
    print(df.head())
    
except Exception as e:
    print(f"Error downloading the dataset: {e}")
    print("Please run the synthetic data generation code instead.")
```

### Option 2: Generate Synthetic Telecom Customer Data

If you can't access the IBM dataset, you can generate synthetic data using this code:

```python
import pandas as pd
import numpy as np
import os

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers to generate
n_customers = 2000

# Generate customer IDs
customer_ids = [f'CUST-{i:04d}' for i in range(1, n_customers + 1)]

# Generate categorical features
def generate_categorical(options, probabilities=None, n=n_customers):
    return np.random.choice(options, size=n, p=probabilities)

# Generate numeric features with correlation to churn
def generate_numeric_with_churn_corr(mean, std, churn_effect, churn, n=n_customers):
    base = np.random.normal(mean, std, n)
    # Adjust values based on churn status to create correlation
    base[churn == 'Yes'] += churn_effect
    return base

# Generate churn status (target variable) - 27% churn rate
churn = generate_categorical(['No', 'Yes'], [0.73, 0.27])

# Generate other categorical features
gender = generate_categorical(['Female', 'Male'], [0.5, 0.5])
senior_citizen = generate_categorical(['0', '1'], [0.8, 0.2])
partner = generate_categorical(['No', 'Yes'], [0.55, 0.45]) 
dependents = generate_categorical(['No', 'Yes'], [0.7, 0.3])

phone_service = generate_categorical(['No', 'Yes'], [0.1, 0.9])
multiple_lines = np.array(['No phone service'] * n_customers)
multiple_lines[phone_service == 'Yes'] = generate_categorical(
    ['No', 'Yes'], [0.6, 0.4], sum(phone_service == 'Yes'))

internet_service = generate_categorical(['No', 'DSL', 'Fiber optic'], [0.2, 0.4, 0.4])

# Internet-dependent services
def generate_internet_service(internet_service):
    result = np.array(['No internet service'] * n_customers)
    has_internet = internet_service != 'No'
    result[has_internet] = generate_categorical(['No', 'Yes'], [0.6, 0.4], sum(has_internet))
    return result

online_security = generate_internet_service(internet_service)
online_backup = generate_internet_service(internet_service)
device_protection = generate_internet_service(internet_service)
tech_support = generate_internet_service(internet_service)
streaming_tv = generate_internet_service(internet_service)
streaming_movies = generate_internet_service(internet_service)

contract = generate_categorical(
    ['Month-to-month', 'One year', 'Two year'], 
    [0.55, 0.25, 0.2]
)
paperless_billing = generate_categorical(['No', 'Yes'], [0.4, 0.6])
payment_method = generate_categorical(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    [0.35, 0.2, 0.25, 0.2]
)

# Generate numeric features
tenure = np.clip(np.random.exponential(scale=24, size=n_customers), 1, 72).astype(int)
# Customers who will churn tend to have lower tenure
tenure[churn == 'Yes'] = np.clip(tenure[churn == 'Yes'] * 0.5, 1, 72).astype(int)

monthly_charges = np.random.normal(65, 30, n_customers)
monthly_charges[internet_service == 'Fiber optic'] += 30
monthly_charges[contract == 'Month-to-month'] += 10
monthly_charges[phone_service == 'Yes'] += 15
monthly_charges = np.clip(monthly_charges, 15, 120)

total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, n_customers)
total_charges = [f"{charge:.2f}" for charge in total_charges]

# Create the DataFrame
df = pd.DataFrame({
    'customerID': customer_ids,
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn
})

# Save the DataFrame to CSV
df.to_csv('data/raw/telecom_customers.csv', index=False)

print(f"Created synthetic dataset with {n_customers} customers at data/raw/telecom_customers.csv")
print(f"Churn rate: {sum(churn == 'Yes')/len(churn):.2%}")
print("\nSample data:")
print(df.head())
```

## Project Structure Setup

Before running the case study code, set up the proper project structure:

```python
import os

# Create project directories
directories = [
    'data/raw', 
    'data/processed', 
    'models',
    'notebooks', 
    'reports/figures',
    'src/features', 
    'src/models', 
    'src/visualization'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## About the Telecom Customer Churn Dataset

The dataset contains information about telecom customers, including:

- **Demographics**: Gender, senior citizen status, partners, dependents
- **Account Information**: Tenure, contract type, payment method
- **Services**: Phone, internet, online security, tech support, streaming services
- **Billing Details**: Monthly charges, total charges, paperless billing
- **Target Variable**: Churn (whether the customer left the company)

This information is used to predict which customers are likely to leave the service, enabling proactive retention strategies.

## Verifying the Dataset

After downloading or generating the dataset, verify it's properly loaded:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the dataset
    df = pd.read_csv('data/raw/telecom_customers.csv')
    
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Churn distribution: {df['Churn'].value_counts(normalize=True)}")
    
    # Quick visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='Contract', hue='Churn', data=df)
    plt.title('Churn by Contract Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error verifying dataset: {e}")
    print("Please ensure the dataset is correctly downloaded or generated.")
```

## Expected Output Files

When running the complete case study code, several files will be generated:

- `models/churn_prediction_model.pkl`: The trained machine learning model
- `models/model_metadata.json`: Information about the model and its performance
- `reports/figures/churn_prediction_summary.png`: Visualization summary for presentation
- `reports/executive_summary.md`: Executive summary of findings and recommendations

## Additional Resources

If you want to explore other customer churn datasets:

1. **Kaggle Telco Customer Churn**: https://www.kaggle.com/blastchar/telco-customer-churn
2. **E-Commerce Customer Churn**: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
3. **Bank Customer Churn**: https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers

These datasets have similar structures but come from different industries, allowing you to test your models in different business contexts.

With this setup, you'll have everything you need to complete the Customer Churn Prediction case study, from data preparation to model training, evaluation, and business recommendation generation.