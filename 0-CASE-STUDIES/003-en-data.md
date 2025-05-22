# Getting Started with the Energy Consumption Forecasting Project

This guide explains how to obtain the necessary dataset and set up your environment for the Energy Consumption Forecasting case study.

## Required Python Packages

The project requires several Python libraries. You can install them using pip:

```python
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm flask joblib
```

## Dataset Information

This project uses household power consumption data from the UCI Machine Learning Repository, prepared by Jason Brownlee for time series analysis.

### Downloading the Dataset

You can download the required dataset using Python:

```python
import os
import urllib.request

# Create directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# URL for hourly load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption_days.csv"
data_path = "data/raw/energy_consumption.csv"

# Download the dataset
urllib.request.urlretrieve(url, data_path)
print(f"Downloaded data to {data_path}")

# Verify the download
import pandas as pd
df = pd.read_csv(data_path, parse_dates=True)
print(f"Dataset shape: {df.shape}")
print(df.head())
```

### Alternative: Direct Download

You can also download the dataset directly by visiting:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption_days.csv

Save the file as `energy_consumption.csv` in the `data/raw` directory.

## Project Structure Setup

Before running the code, you'll need to set up the proper project structure:

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

## About the Energy Consumption Dataset

This dataset contains measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. It includes several electrical quantities and sub-metering values:

- **Global_active_power**: The household global active power (in kilowatts)
- **Global_reactive_power**: The household global reactive power (in kilowatts)
- **Voltage**: Voltage (in volts)
- **Global_intensity**: Household global current intensity (in amperes)
- **Sub_metering_1**: Energy sub-metering for the kitchen (in watt-hours of active energy)
- **Sub_metering_2**: Energy sub-metering for laundry room (in watt-hours of active energy)
- **Sub_metering_3**: Energy sub-metering for water heater & air-conditioner (in watt-hours of active energy)

The daily aggregated version used in this project makes it easier to work with for time series forecasting.

## Expected Output Files

When running the complete code, the following files will be generated:

### Models
- `models/energy_forecast_model.pkl`: The trained ML model for energy forecasting
- `models/scaler.pkl`: The fitted scaler for data normalization

### Visualizations
- `visualizations/power_over_time.png`: Time series plot of energy consumption
- `visualizations/daily_seasonality.png`: Analysis of daily patterns
- `visualizations/weekly_seasonality.png`: Analysis of weekly patterns
- `visualizations/monthly_seasonality.png`: Analysis of monthly patterns
- `visualizations/feature_correlations.png`: Correlation matrix of features
- `visualizations/outliers.png`: Visualization of identified outliers
- `visualizations/model_comparison.png`: Performance comparison of different models
- `visualizations/actual_vs_predicted.png`: Comparison of actual vs predicted values
- `visualizations/residuals.png`: Plot of prediction errors over time
- `visualizations/feature_importance.png`: Importance of different features
- `visualizations/forecast_48h.png`: 48-hour consumption forecast
- `visualizations/model_monitoring.png`: Model performance monitoring charts
- `visualizations/business_impact.png`: Visualizations of business impact metrics

### Reports
- `reports/executive_summary.md`: Summary of project results for stakeholders
- `reports/technical_documentation.md`: Technical details about the model and implementation

## Verifying the Setup

After downloading the dataset, you can verify it's working correctly:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/raw/energy_consumption.csv', parse_dates=True)

# Convert date strings to datetime
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
df.set_index('datetime', inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Quick visualization
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Global_active_power'])
plt.title('Global Active Power Over Time')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.tight_layout()
plt.show()
```

## Running the Case Study

With the environment set up and data downloaded, you can now run the complete case study code to:
- Explore and preprocess the energy consumption data
- Create time-based features and train various forecasting models
- Evaluate model performance
- Generate forecasts
- Calculate business impact
- Create an API for serving predictions

The code is structured to follow the complete machine learning lifecycle from data exploration to deployment and monitoring.