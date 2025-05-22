# Getting the Telecom Customer Churn Dataset

This guide explains how to obtain the necessary data for the Customer Churn Prediction case study. The main dataset needed is the telecom customers dataset, which contains information about customers and whether they churned.

## Option 1: Download the IBM Telco Customer Churn Dataset

1. You can download the IBM Telco Customer Churn dataset directly from GitHub:
   
   Direct link: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

2. Save the file as `telecom_customers_1.csv` in your `data/raw/` directory.

3. Alternatively, use the Python code below to download it automatically:

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
    df.to_csv('data/raw/telecom_customers_1.csv', index=False)
    
    print(f"Successfully downloaded and saved the dataset to data/raw/telecom_customers_1.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print("\nSample data:")
    print(df.head())
    
except Exception as e:
    print(f"Error downloading the dataset: {e}")
```

## Option 2: Generate Synthetic Test Data

If you prefer using synthetic data or can't access the IBM dataset, you can generate synthetic data using the code below:

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
df.to_csv('data/raw/telecom_customers_2.csv', index=False)

print(f"Created synthetic dataset with {n_customers} customers at data/raw/telecom_customers_2.csv")
```

## Setting Up the Project Structure

Before running the case study code, you'll need to set up the proper project structure:

```python
# Set up project directory structure
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

## Verifying the Dataset

After downloading or generating the dataset, verify that it contains the expected information:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/raw/telecom_customers_1.csv')  # or telecom_customers_2.csv

# Check basic information
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nSample data:")
print(df.head())
```

## About the Telecom Customer Churn Dataset

The dataset contains information about telecom customers, including:

- Demographics (gender, senior citizen status, partner, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, security, backup, etc.)
- Billing information (monthly charges, total charges)
- Churn status (the target variable)

This dataset is commonly used for developing predictive models to identify customers at risk of churning, allowing companies to take proactive retention measures.

Once you have the dataset and project structure set up, you can proceed with the rest of the case study code to build and evaluate the customer churn prediction model.