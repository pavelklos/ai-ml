# Getting Started with Three Advanced ML Case Studies

This guide explains how to obtain the necessary datasets and set up your environment for the three advanced machine learning case studies: Energy Consumption Forecasting, Credit Risk Assessment, and Customer Segmentation.

## Required Python Packages

These advanced case studies require several specialized libraries. Install all required packages:

```python
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm prophet statsmodels 
pip install imbalanced-learn holidays openpyxl shap tensorflow flask
```

## Project Structure Setup

Create a directory structure for all three case studies:

```python
import os

# Create project directories
directories = [
    'data/energy_consumption',
    'data/credit_risk',
    'data/customer_segmentation',
    'models',
    'visualizations',
    'reports'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## Case Study 1: Energy Consumption Forecasting

### Dataset Information

This case study requires the Building Energy Consumption dataset, which contains hourly energy usage data along with weather conditions.

#### Option 1: Download from Building Data Genome Project

```python
import os
import pandas as pd
import urllib.request
import zipfile

# Create data directory if it doesn't exist
os.makedirs('data/energy_consumption', exist_ok=True)

# URL for the Building Data Genome Project 2 (a subset of it)
url = "https://github.com/buds-lab/building-data-genome-project-2/archive/refs/heads/master.zip"
zip_path = "data/energy_consumption/bdg_data.zip"

try:
    print(f"Downloading Building Energy dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/energy_consumption")
    
    # Process the data to match the format needed for the case study
    base_path = "data/energy_consumption/building-data-genome-project-2-master/data/meters/cleaned"
    electricity_file = os.path.join(base_path, "electricity_cleaned.csv")
    site_file = os.path.join(base_path, "../metadata/metadata.csv")
    weather_file = os.path.join(base_path, "../weather/weather.csv")
    
    # Create a simplified dataset with energy and weather data
    if os.path.exists(electricity_file) and os.path.exists(weather_file) and os.path.exists(site_file):
        # Load electricity data
        elec_df = pd.read_csv(electricity_file, parse_dates=['timestamp'])
        
        # Load site metadata
        sites_df = pd.read_csv(site_file)
        
        # Load weather data
        weather_df = pd.read_csv(weather_file, parse_dates=['timestamp'])
        
        # Select one building for the case study
        building_id = sites_df[sites_df['primary_use'] == 'Office']['building_id'].iloc[0]
        site_id = sites_df[sites_df['building_id'] == building_id]['site_id'].iloc[0]
        
        # Filter data
        building_elec = elec_df[elec_df['building_id'] == building_id].copy()
        site_weather = weather_df[weather_df['site_id'] == site_id].copy()
        
        # Merge electricity and weather data
        merged_df = pd.merge(
            building_elec, 
            site_weather, 
            on='timestamp',
            how='inner'
        )
        
        # Rename columns to match case study
        merged_df = merged_df.rename(columns={
            'timestamp': 'timestamp',
            'value': 'energy_consumption',
            'air_temperature': 'outdoor_temperature',
            'dew_temperature': 'dew_point',
            'cloud_coverage': 'cloud_coverage',
        })
        
        # Select relevant columns
        final_df = merged_df[['timestamp', 'energy_consumption', 'outdoor_temperature', 'dew_point', 'cloud_coverage']]
        
        # Save the processed dataset
        final_df.to_csv('data/energy_consumption/building_energy_data.csv', index=False)
        
        print("Successfully created building_energy_data.csv")
        print(f"Dataset shape: {final_df.shape}")
        print(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    else:
        print("Required files not found in the downloaded archive.")
        
    # Clean up the ZIP file
    os.remove(zip_path)
    
except Exception as e:
    print(f"Error: {e}")
    print("Please use Option 2 to generate synthetic data instead.")
```

#### Option 2: Generate Synthetic Energy Consumption Data

If you encounter issues with the real dataset, you can generate synthetic data:

```python
import pandas as pd
import numpy as np
import datetime as dt

# Create data directory
os.makedirs('data/energy_consumption', exist_ok=True)

# Generate synthetic energy consumption data
np.random.seed(42)

# Generate timestamps for 1 year of hourly data
start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime(2022, 12, 31, 23)
timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

# Create dataframe
df = pd.DataFrame({'timestamp': timestamps})

# Generate outdoor temperature with seasonal variation
day_of_year = df['timestamp'].dt.dayofyear
hour_of_day = df['timestamp'].dt.hour

# Temperature: seasonal component + daily component + noise
seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 30) / 365)  # Peak in summer
daily_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)  # Peak at 2PM
temp_noise = np.random.normal(0, 2, len(df))
df['outdoor_temperature'] = seasonal_temp + daily_temp + temp_noise

# Dew point (correlated with temperature but lower)
df['dew_point'] = df['outdoor_temperature'] * 0.7 + np.random.normal(0, 1, len(df))

# Cloud coverage (0-10)
df['cloud_coverage'] = np.random.beta(2, 5, len(df)) * 10

# Base energy consumption: higher in extreme temperatures
base_load = 30 + 0.1 * np.abs(df['outdoor_temperature'] - 21)**2  # Optimal temp = 21°C

# Add time patterns
# Weekly: higher on weekdays
weekday_factor = np.where(df['timestamp'].dt.dayofweek < 5, 1.2, 0.7)

# Daily: higher during working hours (9am-5pm)
hour = df['timestamp'].dt.hour
daily_factor = np.where((hour >= 9) & (hour <= 17), 1.5, 0.8)
night_factor = np.where((hour >= 1) & (hour <= 5), 0.5, 1.0)

# Random variation
random_factor = np.random.normal(1, 0.1, len(df))

# Holidays (simplified)
is_holiday = ((df['timestamp'].dt.month == 1) & (df['timestamp'].dt.day == 1)) | \
             ((df['timestamp'].dt.month == 12) & (df['timestamp'].dt.day == 25))
holiday_factor = np.where(is_holiday, 0.6, 1.0)

# Calculate energy consumption
df['energy_consumption'] = base_load * weekday_factor * daily_factor * night_factor * random_factor * holiday_factor

# Add some anomalies
anomaly_indices = np.random.choice(len(df), size=48, replace=False)
df.loc[anomaly_indices, 'energy_consumption'] *= np.random.uniform(1.5, 2.5, size=len(anomaly_indices))

# Save to CSV
df.to_csv('data/energy_consumption/building_energy_data.csv', index=False)

print("Generated synthetic building energy data:")
print(f"Rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Average energy consumption: {df['energy_consumption'].mean():.2f}")
print("Sample data:")
print(df.head())
```

### Verifying the Energy Consumption Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the data
    energy_df = pd.read_csv('data/energy_consumption/building_energy_data.csv', parse_dates=['timestamp'])
    
    # Set timestamp as index
    energy_df.set_index('timestamp', inplace=True)
    
    print("Energy consumption dataset loaded successfully!")
    print(f"Shape: {energy_df.shape}")
    print(f"Date range: {energy_df.index.min()} to {energy_df.index.max()}")
    print(f"Columns: {energy_df.columns.tolist()}")
    
    # Quick visualization of energy consumption patterns
    plt.figure(figsize=(12, 6))
    daily = energy_df['energy_consumption'].resample('D').mean()
    daily.plot()
    plt.title('Daily Average Energy Consumption')
    plt.ylabel('Energy Consumption')
    plt.tight_layout()
    plt.show()
    
    # Verify temperature vs energy consumption relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(energy_df['outdoor_temperature'], energy_df['energy_consumption'], alpha=0.2)
    plt.title('Energy Consumption vs. Temperature')
    plt.xlabel('Outdoor Temperature')
    plt.ylabel('Energy Consumption')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error during verification: {e}")
    print("Please ensure the dataset is correctly downloaded or generated.")
```

## Case Study 2: Credit Risk Assessment

### Dataset Information

This case study uses a credit risk dataset containing loan data, borrower information, and historical payment patterns.

#### Option 1: Download from Kaggle LendingClub Dataset

```python
import os
import pandas as pd
import kaggle

# Create data directory
os.makedirs('data/credit_risk', exist_ok=True)

try:
    # You need to configure your Kaggle API credentials first
    # See: https://github.com/Kaggle/kaggle-api
    
    # Download the dataset
    kaggle.api.dataset_download_files('wordsforthewise/lending-club', 
                                     path='data/credit_risk', 
                                     unzip=True)
    
    # The dataset is large, so let's create a manageable subset
    loan_file = os.path.join('data/credit_risk', 'accepted_2007_to_2018Q4.csv')
    if os.path.exists(loan_file):
        print(f"Reading {loan_file} (this may take a while)...")
        # Read just a sample to reduce memory usage
        df = pd.read_csv(loan_file, nrows=100000)
        
        # Select relevant columns
        cols_to_keep = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
            'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
            'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
            'revol_bal', 'revol_util', 'total_acc', 'zip_code', 'addr_state',
            'loan_status'
        ]
        
        df_subset = df[cols_to_keep].copy()
        
        # Create the loan_default target variable (1 if charged off or defaulted)
        df_subset['loan_default'] = df_subset['loan_status'].isin(
            ['Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']
        ).astype(int)
        
        # Drop original loan_status
        df_subset = df_subset.drop('loan_status', axis=1)
        
        # Rename some columns to match the case study
        df_subset = df_subset.rename(columns={
            'loan_amnt': 'loan_amount',
            'annual_inc': 'annual_income',
            'revol_bal': 'credit_used',
            'total_acc': 'credit_limit',
            'inq_last_6mths': 'recent_inquiries',
            'delinq_2yrs': 'credit_history_length',
            'dti': 'debt_to_income',
            'installment': 'monthly_payment'
        })
        
        # Add some binary features needed for the case study
        df_subset['has_mortgage'] = df_subset['home_ownership'].map({'MORTGAGE': 'Yes', 'OWN': 'No', 'RENT': 'No', 'OTHER': 'No', 'ANY': 'No'})
        df_subset['has_dependents'] = np.random.choice(['Yes', 'No'], size=len(df_subset), p=[0.4, 0.6])
        df_subset['customer_id'] = ['CUST-' + str(i).zfill(7) for i in range(len(df_subset))]
        
        # Save the processed dataset
        df_subset.to_csv('data/credit_risk/credit_risk_data.csv', index=False)
        
        print("Successfully created credit_risk_data.csv")
        print(f"Dataset shape: {df_subset.shape}")
        print(f"Default rate: {df_subset['loan_default'].mean():.2%}")
    else:
        print(f"File {loan_file} not found. Using Option 2 instead.")
        
except Exception as e:
    print(f"Error with Kaggle download: {e}")
    print("Using Option 2 to generate synthetic data instead.")
```

#### Option 2: Generate Synthetic Credit Risk Data

```python
import pandas as pd
import numpy as np
import os

# Create data directory
os.makedirs('data/credit_risk', exist_ok=True)

# Set random seed
np.random.seed(42)

# Number of records
n_records = 10000

# Generate synthetic credit risk data
data = {
    'customer_id': ['CUST-' + str(i).zfill(7) for i in range(n_records)],
    'loan_amount': np.random.normal(15000, 8000, n_records),
    'annual_income': np.random.lognormal(10.5, 0.7, n_records),
    'age': np.random.randint(18, 75, n_records),
    'debt_to_income': np.random.beta(2, 5, n_records) * 0.5,
    'credit_score': np.random.normal(700, 80, n_records),
    'credit_used': np.random.lognormal(8.5, 1, n_records),
    'credit_limit': np.random.lognormal(10, 0.6, n_records),
    'monthly_payment': np.random.lognormal(6, 0.5, n_records),
    'monthly_debt': np.random.lognormal(7, 0.6, n_records),
    'employment_length': np.random.choice(range(0, 30), n_records),
    'credit_history_length': np.random.randint(1, 30, n_records),
    'recent_inquiries': np.random.poisson(1.5, n_records),
    'ProductVariety': np.random.randint(1, 10, n_records),
    'OrderVariability': np.random.uniform(0, 100, n_records)
}

# Add categorical features
data['loan_purpose'] = np.random.choice(
    ['home_improvement', 'debt_consolidation', 'credit_card_refinancing', 
     'major_purchase', 'small_business', 'medical_expenses', 'vacation', 
     'moving', 'wedding', 'car_financing', 'other'],
    size=n_records,
    p=[0.15, 0.25, 0.15, 0.1, 0.05, 0.1, 0.05, 0.05, 0.03, 0.05, 0.02]
)

data['home_ownership'] = np.random.choice(
    ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
    size=n_records,
    p=[0.4, 0.4, 0.15, 0.05]
)

data['zipcode'] = [str(np.random.randint(10000, 99999)) for _ in range(n_records)]
data['has_mortgage'] = data['home_ownership'].map({'MORTGAGE': 'Yes', 'OWN': 'No', 'RENT': 'No', 'OTHER': 'No'})
data['has_dependents'] = np.random.choice(['Yes', 'No'], size=n_records, p=[0.4, 0.6])

# Create target variable with realistic relationship to features
# Higher probability of default for:
# - Higher debt-to-income ratio
# - Lower income
# - Higher loan amount relative to income
# - More recent inquiries
# - Lower credit score

# Calculate default probability based on features
income_factor = np.exp(-data['annual_income'] / 100000)
dti_factor = data['debt_to_income'] * 3
loan_income_ratio = data['loan_amount'] / data['annual_income']
inquiry_factor = data['recent_inquiries'] * 0.2
credit_factor = np.exp(-(data['credit_score'] - 500) / 100)

default_prob = (income_factor + dti_factor + loan_income_ratio + inquiry_factor + credit_factor) / 5
default_prob = default_prob / max(default_prob) * 0.5  # Scale to reasonable default rate

# Create target variable
data['loan_default'] = np.random.binomial(1, default_prob)

# Create DataFrame
df = pd.DataFrame(data)

# Ensure realistic relationships 
# Cap loan amount
df['loan_amount'] = df['loan_amount'].clip(1000, 50000)
# Ensure credit_limit > credit_used for most customers
df.loc[df['credit_limit'] < df['credit_used'], 'credit_limit'] = df.loc[df['credit_limit'] < df['credit_used'], 'credit_used'] * 1.5
# Cap ratios
df['debt_to_income'] = df['debt_to_income'].clip(0, 0.6)
# Fix credit score range
df['credit_score'] = df['credit_score'].clip(350, 850)

# Save to CSV
df.to_csv('data/credit_risk/credit_risk_data.csv', index=False)
print("Successfully created synthetic credit risk data:")
print(f"Records: {len(df)}")
print(f"Default rate: {df['loan_default'].mean():.2%}")
print("Sample data:")
print(df[['loan_amount', 'annual_income', 'debt_to_income', 'loan_default']].head())
```

### Verifying the Credit Risk Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the data
    credit_df = pd.read_csv('data/credit_risk/credit_risk_data.csv')
    
    print("Credit risk dataset loaded successfully!")
    print(f"Shape: {credit_df.shape}")
    print(f"Default rate: {credit_df['loan_default'].mean():.2%}")
    print(f"Columns: {credit_df.columns.tolist()}")
    
    # Check data types
    print("\nData types:")
    print(credit_df.dtypes)
    
    # View relationship between debt-to-income and default
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='loan_default', y='debt_to_income', data=credit_df)
    plt.title('Debt-to-Income Ratio vs. Loan Default')
    plt.tight_layout()
    plt.show()
    
    # View relationship between loan amount and default
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='loan_default', y='loan_amount', data=credit_df)
    plt.title('Loan Amount vs. Loan Default')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error during verification: {e}")
    print("Please ensure the dataset is correctly downloaded or generated.")
```

## Case Study 3: Customer Segmentation for E-commerce

### Dataset Information

This case study uses the Online Retail II dataset, which contains transactional data from a UK-based online retailer.

#### Option 1: Download from UCI Machine Learning Repository

```python
import os
import pandas as pd
import urllib.request
from zipfile import ZipFile
import io

# Create data directory
os.makedirs('data/customer_segmentation', exist_ok=True)

try:
    # Download the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    print(f"Downloading the Online Retail II dataset from {url}...")
    print("This is a large file (~22MB) and may take a few minutes.")
    
    # Download the file
    filename, _ = urllib.request.urlretrieve(url, 'data/customer_segmentation/online_retail_II.xlsx')
    
    # Verify the download
    try:
        # Load a small part to verify
        df = pd.read_excel(filename, sheet_name="Year 2010-2011", nrows=5, engine='openpyxl')
        print("Download successful!")
        print("Sample data:")
        print(df.head())
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
    
except Exception as e:
    print(f"Error downloading the dataset: {e}")
    print("Using Option 2 to generate synthetic data instead.")
```

#### Option 2: Generate Synthetic Online Retail Data

If you encounter issues downloading the real dataset, you can create synthetic data:

```python
import pandas as pd
import numpy as np
import datetime as dt
import random
import os

# Create data directory
os.makedirs('data/customer_segmentation', exist_ok=True)

# Set random seed
np.random.seed(42)
random.seed(42)

# Number of records
n_records = 50000

# Customer IDs (1000 unique customers)
customer_ids = [str(random.randint(10000, 99999)) for _ in range(1000)]

# Generate dates between 2010-01-01 and 2011-12-31
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2011, 12, 31)
days_between = (end_date - start_date).days

# Generate product data
def generate_product_id():
    return str(random.randint(10000, 99999))

def generate_product_description():
    adjectives = ["VINTAGE", "RETRO", "MODERN", "CLASSIC", "ANTIQUE", "METAL", "WOODEN", "GLASS", "CERAMIC", "PAPER"]
    objects = ["LAMP", "FRAME", "SIGN", "BOX", "HANGER", "HOOK", "CANDLE", "HOLDER", "PLATE", "CUP", "BOWL"]
    items = ["SET", "COLLECTION", "BUNDLE", "PACK", "ASSORTMENT"]
    
    if random.random() < 0.7:
        return f"{random.choice(adjectives)} {random.choice(objects)}"
    else:
        return f"{random.choice(adjectives)} {random.choice(objects)} {random.choice(items)} OF {random.randint(2, 12)}"

# Create a list of 1000 unique products
products = []
for _ in range(1000):
    stock_code = generate_product_id()
    description = generate_product_description()
    unit_price = round(random.uniform(0.5, 100), 2)
    products.append((stock_code, description, unit_price))

# Generate transactions
data = []
for _ in range(n_records):
    # Select a random date
    days = random.randint(0, days_between)
    transaction_date = start_date + dt.timedelta(days=days)
    
    # Add time
    hour = random.randint(9, 17)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    transaction_date = transaction_date.replace(hour=hour, minute=minute, second=second)
    
    # Select a customer
    customer_id = random.choice(customer_ids)
    
    # Select a product
    stock_code, description, unit_price = random.choice(products)
    
    # Generate quantity
    quantity = random.randint(1, 15)
    if random.random() < 0.05:  # 5% chance of bulk order
        quantity = random.randint(16, 100)
    
    # Calculate total price
    total = quantity * unit_price
    
    # Generate invoice number (format: INVYYYYMMDDXXXX)
    invoice_no = f"INV{transaction_date.strftime('%Y%m%d')}{random.randint(1000, 9999)}"
    
    # Country (mostly UK)
    countries = ['United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 'Netherlands', 
                'Belgium', 'Switzerland', 'Portugal', 'Australia', 'USA']
    weights = [0.7, 0.05, 0.05, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02]
    country = random.choices(countries, weights=weights)[0]
    
    # Create record
    data.append({
        'Invoice': invoice_no,
        'StockCode': stock_code,
        'Description': description,
        'Quantity': quantity,
        'InvoiceDate': transaction_date,
        'Price': unit_price,
        'Customer ID': customer_id,
        'Country': country
    })

# Create dataframe
df = pd.DataFrame(data)

# Save as Excel
df.to_excel('data/customer_segmentation/online_retail_II.xlsx', sheet_name="Year 2010-2011", index=False, engine='openpyxl')

print("Successfully created synthetic online retail data:")
print(f"Records: {len(df)}")
print(f"Unique customers: {df['Customer ID'].nunique()}")
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print("Sample data:")
print(df.head())
```

### Verifying the Customer Segmentation Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the data (reading just the first sheet)
    retail_df = pd.read_excel('data/customer_segmentation/online_retail_II.xlsx', 
                             sheet_name="Year 2010-2011", 
                             engine='openpyxl')
    
    print("Online Retail dataset loaded successfully!")
    print(f"Shape: {retail_df.shape}")
    print(f"Unique customers: {retail_df['Customer ID'].nunique()}")
    print(f"Date range: {retail_df['InvoiceDate'].min()} to {retail_df['InvoiceDate'].max()}")
    
    # Basic data checks
    print("\nMissing values:")
    print(retail_df.isnull().sum())
    
    # View transaction distribution
    plt.figure(figsize=(12, 6))
    retail_df['InvoiceDate'].dt.month.value_counts().sort_index().plot(kind='bar')
    plt.title('Number of Transactions by Month')
    plt.xlabel('Month')
    plt.ylabel('Transaction Count')
    plt.tight_layout()
    plt.show()
    
    # View country distribution
    plt.figure(figsize=(12, 6))
    country_counts = retail_df['Country'].value_counts().head(10)
    sns.barplot(x=country_counts.index, y=country_counts.values)
    plt.title('Top 10 Countries by Number of Transactions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error during verification: {e}")
    print("Please ensure the dataset is correctly downloaded or generated.")
```

## Expected Outputs from the Case Studies

When running the complete code for each case study, you'll generate the following outputs:

### Case Study 1: Energy Consumption Forecasting
- Time series plots at different intervals (daily, weekly, monthly)
- Seasonal patterns (hourly, daily)
- Feature importance visualizations
- Comparative model performance charts
- Energy consumption forecasts for the next 7 days
- Cost savings analysis based on time-of-use pricing

### Case Study 2: Credit Risk Assessment
- Feature distribution charts by default status
- Correlation matrices
- ROC curves for model performance comparison
- Confusion matrices for each model
- SHAP plots for model interpretation
- Optimal threshold analysis
- Business impact and ROI calculations

### Case Study 3: Customer Segmentation
- RFM distribution plots
- PCA visualizations of customer clusters
- Clustering evaluation metrics
- Dendrograms for hierarchical clustering
- Radar charts for segment profiles
- Marketing budget allocation charts
- ROI analysis by customer segment

## Computational Requirements

These advanced case studies are more computationally intensive than the basic examples:

- **Energy Consumption Forecasting**: Requires at least 8GB RAM for time series forecasting
- **Credit Risk Assessment**: Requires at least 8GB RAM, preferably 16GB for larger datasets
- **Customer Segmentation**: May require 8-16GB RAM for clustering algorithms with large datasets

If you're working with limited resources, consider:
1. Using the synthetic data options (smaller datasets)
2. Reducing the number of cross-validation folds
3. Simplifying model hyperparameters
4. Using a subset of the full datasets

## Additional Resources

For deeper exploration of these topics:

- **Energy Forecasting**: 
  - [Building Data Genome Project](https://github.com/buds-lab/building-data-genome-project-2)
  - [Prophet Documentation](https://facebook.github.io/prophet/)

- **Credit Risk Assessment**:
  - [LendingClub Dataset on Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)
  - [SHAP Documentation](https://shap.readthedocs.io/)

- **Customer Segmentation**:
  - [UCI Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
  - [RFM Analysis Guide](https://clevertap.com/blog/rfm-analysis/)

These advanced case studies demonstrate sophisticated ML techniques that are directly applicable to real-world business problems in energy management, financial services, and e-commerce.