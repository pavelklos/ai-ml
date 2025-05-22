# Getting Started with the Retail Sales Prediction Project

This guide explains how to obtain the necessary datasets and set up your environment for the Retail Sales Prediction case study.

## Required Python Packages

First, install the required Python libraries:

```python
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost flask joblib
```

## Dataset Information

This project requires two main datasets:
1. `store_sales.csv` - Historical sales data for retail stores
2. `store_info.csv` - Information about store characteristics

### Option 1: Using the Rossmann Store Sales Dataset

The Rossmann Store Sales dataset from Kaggle is an excellent match for this case study:

1. Visit [Kaggle's Rossmann Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales/data)
2. Download the following files:
   - `train.csv` (rename to `store_sales.csv`)
   - `store.csv` (rename to `store_info.csv`) 

```python
import os
import pandas as pd

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)

# If you've downloaded the Kaggle files, load and process them
train_df = pd.read_csv('train.csv')
train_df.to_csv('data/raw/store_sales.csv', index=False)

store_df = pd.read_csv('store.csv')
store_df.to_csv('data/raw/store_info.csv', index=False)

print("Datasets prepared successfully")
```

### Option 2: Generate Synthetic Retail Data

If you can't access the Kaggle dataset, you can generate synthetic data that matches the expected format:

```python
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Create directories
os.makedirs('data/raw', exist_ok=True)

# Generate store_info.csv
np.random.seed(42)
n_stores = 100

# Generate store information
store_types = ['a', 'b', 'c', 'd']
assortments = ['a', 'b', 'c']

store_data = {
    'Store': list(range(1, n_stores + 1)),
    'StoreType': np.random.choice(store_types, size=n_stores),
    'Assortment': np.random.choice(assortments, size=n_stores),
    'CompetitionDistance': np.random.randint(500, 20000, size=n_stores)
}

# Add some missing values to CompetitionDistance
missing_indices = np.random.choice(n_stores, size=10, replace=False)
for idx in missing_indices:
    store_data['CompetitionDistance'][idx] = np.nan

store_df = pd.DataFrame(store_data)
store_df.to_csv('data/raw/store_info.csv', index=False)
print("Generated store_info.csv")

# Generate store_sales.csv
start_date = datetime(2020, 1, 1)
end_date = datetime(2022, 12, 31)
date_range = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') 
              for x in range((end_date - start_date).days + 1)]

sales_data = []
for store_id in range(1, n_stores + 1):
    # Generate more data for certain stores
    sample_size = min(len(date_range), np.random.randint(500, len(date_range)))
    dates = np.random.choice(date_range, size=sample_size, replace=False)
    
    for date in dates:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Base sales with seasonal pattern
        base_sales = 5000 + 2000 * np.sin(date_obj.month / 12 * 2 * np.pi)
        
        # Add day of week effect
        day_effect = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 0.7]  # Mon-Sun
        day_multiplier = day_effect[date_obj.weekday()]
        
        # Store type effect
        store_type = store_df.loc[store_df['Store'] == store_id, 'StoreType'].iloc[0]
        type_multiplier = {'a': 1.2, 'b': 1.0, 'c': 0.8, 'd': 0.9}[store_type]
        
        # Randomness
        noise = np.random.normal(1, 0.2)
        
        # Promotion effect (20% of days have promotion)
        promo = np.random.choice([0, 1], p=[0.8, 0.2])
        promo_effect = 1.3 if promo else 1.0
        
        # Holiday effect (3% of days are holidays)
        holiday_types = ['0', 'a', 'b', 'c']
        holiday_probs = [0.97, 0.01, 0.01, 0.01]
        state_holiday = np.random.choice(holiday_types, p=holiday_probs)
        holiday_effect = 0.5 if state_holiday != '0' else 1.0
        
        # Calculate final sales
        sales = base_sales * day_multiplier * type_multiplier * noise * promo_effect * holiday_effect
        
        # Add some randomness to customer numbers (correlated with sales)
        customers = int(sales / 50 * np.random.normal(1, 0.1))
        
        sales_data.append({
            'Store': store_id,
            'Date': date,
            'Sales': max(0, int(sales)),
            'Customers': max(1, customers),
            'Open': 1 if state_holiday != 'a' else 0,
            'Promo': promo,
            'StateHoliday': state_holiday
        })

sales_df = pd.DataFrame(sales_data)
sales_df.to_csv('data/raw/store_sales.csv', index=False)
print(f"Generated store_sales.csv with {len(sales_df)} records")

# Display sample data
print("\nStore Info Sample:")
print(store_df.head())
print("\nSales Data Sample:")
print(sales_df.head())
```

## Project Structure Setup

Create the necessary directory structure for the project:

```python
import os

# Create project directories
directories = [
    'data/raw',
    'data/processed',
    'models',
    'visualizations',
    'reports'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")
```

## About the Dataset

### Sales Data (`store_sales.csv`)
Contains historical sales data with these key columns:
- `Store`: Store number
- `Date`: Date of the sales record
- `Sales`: Sales amount for that day
- `Customers`: Number of customers on that day
- `Open`: Whether the store was open (0/1)
- `Promo`: Whether there was a promotion running (0/1)
- `StateHoliday`: Type of holiday (0=None, a=Public holiday, b=Easter, c=Christmas)
- `DayOfWeek`: Day of week (1=Monday, 7=Sunday)

### Store Information (`store_info.csv`)
Contains details about each store:
- `Store`: Store number
- `StoreType`: Type of store (a, b, c, d)
- `Assortment`: Level of product assortment (a=basic, b=extra, c=extended)
- `CompetitionDistance`: Distance to nearest competitor store (meters)

## Verifying the Setup

Run this code to check if your datasets are properly formatted and ready for the case study:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
try:
    sales_df = pd.read_csv('data/raw/store_sales.csv')
    store_df = pd.read_csv('data/raw/store_info.csv')
    
    # Convert date to datetime
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    
    print("Datasets loaded successfully!")
    print(f"Sales data shape: {sales_df.shape}")
    print(f"Store info shape: {store_df.shape}")
    
    # Check required columns in sales data
    required_sales_cols = ['Store', 'Date', 'Sales', 'Customers', 'Promo']
    missing_cols = [col for col in required_sales_cols if col not in sales_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in sales data: {missing_cols}")
    else:
        print("Sales data has all required columns.")
    
    # Check required columns in store data
    required_store_cols = ['Store', 'StoreType', 'CompetitionDistance']
    missing_cols = [col for col in required_store_cols if col not in store_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in store data: {missing_cols}")
    else:
        print("Store data has all required columns.")
    
    # Quick visualization
    plt.figure(figsize=(10, 5))
    daily_sales = sales_df.groupby('Date')['Sales'].sum()
    plt.plot(daily_sales.index, daily_sales.values)
    plt.title("Total Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error during setup verification: {e}")
```

## Additional Resources

If you need larger or more diverse retail datasets, consider:

1. **Walmart Store Sales Forecasting**: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
   - Contains data for 45 Walmart stores including holiday details

2. **Retail Data Analytics**: https://www.kaggle.com/datasets/manjeetsingh/retaildataset
   - Contains 3 years of data across multiple product categories

3. **Superstore Sales Dataset**: https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls
   - A widely used sample dataset with detailed retail sales information

## Expected Output Files

When running the complete code, the following files will be generated:

- `models/sales_prediction_model.pkl`: The trained ML model
- `models/feature_names.pkl`: List of feature names used by the model
- `model_monitoring.png`: Visualizations of model performance metrics over time
- `business_impact.png`: Visualizations of business impact and ROI analysis

With these datasets and the code from the case study, you'll be able to build a complete retail sales prediction system and evaluate its business impact.