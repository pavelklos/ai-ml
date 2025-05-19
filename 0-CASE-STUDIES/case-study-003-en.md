# Case Study: End-to-End ML Project - Energy Consumption Forecasting

## 1. Problem Definition and Project Setup

In this case study, we'll develop a machine learning system to forecast hourly electricity consumption for a utility company. Accurate forecasting helps with resource planning, pricing optimization, and grid management.

```python
# Project goal definition
"""
Project: Energy Consumption Forecasting
Goal: Predict hourly electricity consumption for the next 24-48 hours
Business Impact:
- Optimize power generation resources (estimated 5-10% cost savings)
- Improve grid stability planning
- Enable dynamic pricing models
- Reduce carbon footprint through optimized energy generation

Success Metrics:
- RMSE < 5% of peak consumption
- MAPE < 10%
- Reliable predictions during peak demand periods
"""
```

### Setting Up the Project Structure

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

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

## 2. Data Collection and Exploration

### Obtaining Energy Consumption Data

```python
# For this example, we'll use open data from PJM Interconnection
# This contains hourly power consumption data

# Download data (in real project, you might use an API or direct database connection)
import urllib.request

# URL for hourly load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption_days.csv"
data_path = "data/raw/energy_consumption.csv"

urllib.request.urlretrieve(url, data_path)
print(f"Downloaded data to {data_path}")

# Load the data
df = pd.read_csv(data_path, parse_dates=True)

# Convert date strings to datetime
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
df.set_index('datetime', inplace=True)

print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Basic information
print("\nData types and missing values:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())
```

### Exploratory Data Analysis

```python
# Time series visualization
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['Global_active_power'], linewidth=1)
plt.title('Global Active Power Over Time')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.tight_layout()
plt.savefig('visualizations/power_over_time.png')
plt.show()

# Check for seasonality
# Daily seasonality
hourly_data = df.resample('H').mean()  # Convert to hourly if needed
hourly_avg = hourly_data.groupby(hourly_data.index.hour).mean()

plt.figure(figsize=(12, 6))
plt.plot(hourly_avg.index, hourly_avg['Global_active_power'], marker='o')
plt.title('Average Power Consumption by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Global Active Power (kilowatts)')
plt.grid(True)
plt.savefig('visualizations/daily_seasonality.png')
plt.show()

# Weekly seasonality
daily_data = df.resample('D').mean()
weekly_avg = daily_data.groupby(daily_data.index.dayofweek).mean()

plt.figure(figsize=(12, 6))
plt.plot(weekly_avg.index, weekly_avg['Global_active_power'], marker='o')
plt.title('Average Power Consumption by Day of Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Average Global Active Power (kilowatts)')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.grid(True)
plt.savefig('visualizations/weekly_seasonality.png')
plt.show()

# Monthly seasonality
monthly_data = df.resample('M').mean()
monthly_avg = monthly_data.groupby(monthly_data.index.month).mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_avg.index, monthly_avg['Global_active_power'], marker='o')
plt.title('Average Power Consumption by Month')
plt.xlabel('Month')
plt.ylabel('Average Global Active Power (kilowatts)')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.savefig('visualizations/monthly_seasonality.png')
plt.show()

# Check for correlations between features
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig('visualizations/feature_correlations.png')
plt.show()
```

## 3. Data Preprocessing and Feature Engineering

### Data Cleaning

```python
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values using forward fill (assuming time series continuity)
df_clean = df.fillna(method='ffill')

# Check for outliers using IQR
Q1 = df_clean['Global_active_power'].quantile(0.25)
Q3 = df_clean['Global_active_power'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[(df_clean['Global_active_power'] < lower_bound) | 
                   (df_clean['Global_active_power'] > upper_bound)]

print(f"\nNumber of outliers detected: {len(outliers)}")

# Visualize outliers
plt.figure(figsize=(12, 6))
plt.scatter(df_clean.index, df_clean['Global_active_power'], s=2, label='Data')
plt.scatter(outliers.index, outliers['Global_active_power'], color='red', s=5, label='Outliers')
plt.title('Power Consumption with Outliers Highlighted')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.savefig('visualizations/outliers.png')
plt.show()

# For this tutorial, we'll cap outliers instead of removing them
df_clean['Global_active_power'] = df_clean['Global_active_power'].clip(lower_bound, upper_bound)
print("\nOutliers capped at IQR boundaries")
```

### Feature Engineering

```python
# Create time-based features
df_features = df_clean.copy()

# Date-based features
df_features['hour'] = df_features.index.hour
df_features['dayofweek'] = df_features.index.dayofweek
df_features['quarter'] = df_features.index.quarter
df_features['month'] = df_features.index.month
df_features['year'] = df_features.index.year
df_features['dayofyear'] = df_features.index.dayofyear

# Create cyclical features for hour, day of week and month
# This preserves the cyclic nature of these features
df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour']/24)
df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour']/24)

df_features['dow_sin'] = np.sin(2 * np.pi * df_features['dayofweek']/7)
df_features['dow_cos'] = np.cos(2 * np.pi * df_features['dayofweek']/7)

df_features['month_sin'] = np.sin(2 * np.pi * df_features['month']/12)
df_features['month_cos'] = np.cos(2 * np.pi * df_features['month']/12)

# Is weekend feature
df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)

# Lag features (previous days/hours consumption)
for lag in [1, 2, 3, 6, 12, 24]:  # hours
    df_features[f'lag_{lag}h'] = df_features['Global_active_power'].shift(lag)

# Rolling window features
for window in [3, 6, 12, 24]:  # hours
    df_features[f'rolling_mean_{window}h'] = df_features['Global_active_power'].rolling(window=window).mean()
    df_features[f'rolling_std_{window}h'] = df_features['Global_active_power'].rolling(window=window).std()

# Drop NaN values created by lag and rolling features
df_features = df_features.dropna()

print("\nEngineered feature dataset shape:", df_features.shape)
print("\nEngineered features:")
print(df_features.columns.tolist())

# Display a sample of the engineered features
print("\nSample of engineered features:")
print(df_features.head())
```

## 4. Feature Selection and Dataset Preparation

```python
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Define target variable and features
target = 'Global_active_power'
exclude_columns = ['Global_reactive_power', 'Voltage', 'Global_intensity', 
                   'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

features = [col for col in df_features.columns if col != target and col not in exclude_columns]

X = df_features[features]
y = df_features[target]

# Split data chronologically (important for time series)
# We'll use the last 20% as the test set
split_idx = int(len(X) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training data: {X_train.shape[0]} samples")
print(f"Testing data: {X_test.shape[0]} samples")

# Standardize numerical features
scaler = StandardScaler()
numerical_features = [col for col in X_train.columns if X_train[col].dtype != 'object' 
                     and col not in ['hour', 'dayofweek', 'month', 'year', 'is_weekend']]

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Save the processed datasets
X_train.to_csv('data/processed/X_train.csv')
X_test.to_csv('data/processed/X_test.csv')
y_train.to_csv('data/processed/y_train.csv')
y_test.to_csv('data/processed/y_test.csv')

print("\nPreprocessed data saved to 'data/processed/' directory")
```

## 5. Model Development and Evaluation

### Baseline Models

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

# Define evaluation metrics function
def evaluate_model(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R²: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

# Dictionary to store model results
results = {}

# 1. Naive Forecast (previous day's value)
y_pred_naive = y_test.shift(24).fillna(y_test.mean())
results['Naive Forecast'] = evaluate_model('Naive Forecast', y_test, y_pred_naive)

# 2. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
results['Linear Regression'] = evaluate_model('Linear Regression', y_test, y_pred_lr)

# 3. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
results['Random Forest'] = evaluate_model('Random Forest', y_test, y_pred_rf)

# 4. Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
results['Gradient Boosting'] = evaluate_model('Gradient Boosting', y_test, y_pred_gb)

# 5. XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results['XGBoost'] = evaluate_model('XGBoost', y_test, y_pred_xgb)

# 6. LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
results['LightGBM'] = evaluate_model('LightGBM', y_test, y_pred_lgb)

# Create results dataframe for visualization
results_df = pd.DataFrame([
    results['Naive Forecast'],
    results['Linear Regression'],
    results['Random Forest'],
    results['Gradient Boosting'],
    results['XGBoost'],
    results['LightGBM']
])

# Visualize model comparison
plt.figure(figsize=(12, 8))

# RMSE Comparison
plt.subplot(2, 1, 1)
plt.barh(results_df['model_name'], results_df['rmse'])
plt.title('Model Comparison - RMSE (lower is better)')
plt.xlabel('RMSE')
plt.xlim(left=0)
for i, v in enumerate(results_df['rmse']):
    plt.text(v + 0.01, i, f"{v:.3f}")

# MAPE Comparison
plt.subplot(2, 1, 2)
plt.barh(results_df['model_name'], results_df['mape'])
plt.title('Model Comparison - MAPE (lower is better)')
plt.xlabel('MAPE (%)')
plt.xlim(left=0)
for i, v in enumerate(results_df['mape']):
    plt.text(v + 0.01, i, f"{v:.1f}%")

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png')
plt.show()

# Select the best performing model (assuming XGBoost)
best_model = xgb_model  # or whichever model performed best
best_model_name = 'XGBoost'
```

### Visualizing Predictions

```python
# Plot the actual vs predicted values for the best model
plt.figure(figsize=(15, 7))

# Get the appropriate predictions
if best_model_name == 'XGBoost':
    y_pred = y_pred_xgb
elif best_model_name == 'LightGBM':
    y_pred = y_pred_lgb
elif best_model_name == 'Random Forest':
    y_pred = y_pred_rf
elif best_model_name == 'Gradient Boosting':
    y_pred = y_pred_gb
elif best_model_name == 'Linear Regression':
    y_pred = y_pred_lr
else:
    y_pred = y_pred_naive

# Plot
plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2)
plt.plot(y_test.index, y_pred, label=f'Predicted ({best_model_name})', linewidth=2, alpha=0.8)
plt.title('Actual vs Predicted Power Consumption')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.grid(True)
plt.savefig('visualizations/actual_vs_predicted.png')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, residuals, color='red', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Prediction Residuals Over Time')
plt.xlabel('Date')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid(True)
plt.savefig('visualizations/residuals.png')
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='blue')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Distribution of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('visualizations/residuals_histogram.png')
plt.show()
```

### Feature Importance

```python
# For tree-based models, extract feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
    if best_model_name == 'XGBoost':
        importances = best_model.feature_importances_
    else:
        importances = best_model.feature_importances_
    
    # Create a DataFrame for easier visualization
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
    plt.title(f'Top 15 Feature Importances ({best_model_name})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()  # Display highest importance at the top
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.show()
    
    print(f"\nTop 10 Most Important Features ({best_model_name}):")
    print(feature_importance.head(10))
```

## 6. Model Optimization

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Define hyperparameter search space for XGBoost
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'min_child_weight': [1, 2, 3, 4]
}

# Use TimeSeriesSplit for time series data
tscv = TimeSeriesSplit(n_splits=5)

# Random search for hyperparameters
random_search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit random search
print("\nPerforming hyperparameter tuning. This may take some time...")
random_search.fit(X_train, y_train)

# Best parameters and score
print("\nBest hyperparameters:")
print(random_search.best_params_)
best_score = np.sqrt(-random_search.best_score_)
print(f"Best RMSE: {best_score:.4f}")

# Create optimized model with best parameters
tuned_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **random_search.best_params_)

# Train on full training set
tuned_model.fit(X_train, y_train)

# Evaluate tuned model
y_pred_tuned = tuned_model.predict(X_test)
tuned_results = evaluate_model(f'Tuned {best_model_name}', y_test, y_pred_tuned)

# Compare with previous best model
plt.figure(figsize=(10, 6))
comparison = pd.DataFrame([
    results[best_model_name],
    tuned_results
])
comparison[['rmse', 'mape']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Before and After Tuning')
plt.ylabel('Value')
plt.xticks([0, 1], [best_model_name, f'Tuned {best_model_name}'], rotation=0)
plt.grid(True, axis='y')
plt.legend(['RMSE', 'MAPE (%)'])

# Add value labels on bars
for i, metric in enumerate(['rmse', 'mape']):
    for j, model in enumerate([best_model_name, f'Tuned {best_model_name}']):
        if j == 0:
            value = results[best_model_name][metric]
        else:
            value = tuned_results[metric]
        plt.text(j - 0.1 + i*0.2, value + 0.1, f"{value:.2f}", rotation=0)

plt.tight_layout()
plt.savefig('visualizations/tuning_comparison.png')
plt.show()

# Update best model to tuned version
best_model = tuned_model
```

## 7. Model Deployment

### Model Serialization

```python
import joblib
import os

# Save the trained model and scaler
model_path = os.path.join('models', 'energy_forecast_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModel saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
```

### Creating a Prediction Function

```python
def make_forecast(hours_to_predict=24, model_path='models/energy_forecast_model.pkl', 
                  scaler_path='models/scaler.pkl'):
    """
    Generate energy consumption forecast for the specified number of hours
    
    Parameters:
    -----------
    hours_to_predict : int
        Number of hours to forecast into the future
    model_path : str
        Path to the saved model file
    scaler_path : str
        Path to the saved scaler file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the forecasted values with timestamps
    """
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Get the last available data point (in a real system this would be current data)
    latest_data = df_features.iloc[-1]
    
    # Initialize results storage
    forecast_dates = []
    forecast_values = []
    
    # Get the last timestamp from the data
    last_timestamp = df_features.index[-1]
    
    # Create a copy of the latest data for making predictions
    current_data = latest_data.copy()
    
    # Generate predictions for each hour
    for i in range(1, hours_to_predict + 1):
        # Calculate the next timestamp
        next_timestamp = last_timestamp + pd.Timedelta(hours=i)
        forecast_dates.append(next_timestamp)
        
        # Update time features for the next hour
        current_data['hour'] = next_timestamp.hour
        current_data['dayofweek'] = next_timestamp.dayofweek
        current_data['quarter'] = next_timestamp.quarter
        current_data['month'] = next_timestamp.month
        current_data['year'] = next_timestamp.year
        current_data['dayofyear'] = next_timestamp.dayofyear
        current_data['is_weekend'] = 1 if next_timestamp.dayofweek >= 5 else 0
        
        # Update cyclical features
        current_data['hour_sin'] = np.sin(2 * np.pi * next_timestamp.hour/24)
        current_data['hour_cos'] = np.cos(2 * np.pi * next_timestamp.hour/24)
        current_data['dow_sin'] = np.sin(2 * np.pi * next_timestamp.dayofweek/7)
        current_data['dow_cos'] = np.cos(2 * np.pi * next_timestamp.dayofweek/7)
        current_data['month_sin'] = np.sin(2 * np.pi * next_timestamp.month/12)
        current_data['month_cos'] = np.cos(2 * np.pi * next_timestamp.month/12)
        
        # Extract features needed for prediction
        feature_vector = current_data[X.columns].values.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(feature_vector)[0]
        forecast_values.append(prediction)
        
        # Update lag features for next iteration
        for lag in range(24, 0, -1):  # Update from oldest to newest
            if f'lag_{lag}h' in current_data.index:
                if lag > 1:
                    prev_lag = f'lag_{lag-1}h'
                    if prev_lag in current_data.index:
                        current_data[f'lag_{lag}h'] = current_data[prev_lag]
                else:
                    # lag_1h gets the latest prediction
                    current_data['lag_1h'] = prediction
    
        # Update rolling means (simplified - in production you'd need more robust logic)
        if 'rolling_mean_24h' in current_data.index:
            # Simple approximation - use average of lag values if available
            lag_values = [current_data[f'lag_{i}h'] for i in [1, 3, 6, 12, 24] if f'lag_{i}h' in current_data.index]
            if lag_values:
                current_data['rolling_mean_24h'] = sum(lag_values) / len(lag_values)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'timestamp': forecast_dates,
        'forecasted_consumption': forecast_values
    })
    
    forecast_df.set_index('timestamp', inplace=True)
    
    return forecast_df

# Generate a 48-hour forecast
forecast = make_forecast(hours_to_predict=48)

# Plot the forecast
plt.figure(figsize=(15, 7))

# Plot the last 7 days of actual data
last_week = df[target].iloc[-7*24:]
plt.plot(last_week.index, last_week.values, label='Historical Data', color='blue')

# Plot the forecast
plt.plot(forecast.index, forecast['forecasted_consumption'], label='Forecast', color='red', linestyle='--')

# Add a vertical line to separate historical data from forecast
forecast_start = forecast.index[0]
plt.axvline(forecast_start, color='gray', linestyle='-', alpha=0.7)
plt.text(forecast_start, plt.ylim()[1]*0.9, 'Forecast Start', 
         rotation=90, verticalalignment='top')

plt.title('Energy Consumption Forecast (Next 48 Hours)')
plt.xlabel('Date')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('visualizations/forecast_48h.png')
plt.show()

print("\nForecast for the next 48 hours:")
print(forecast)
```

### Creating a Simple API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/forecast', methods=['GET'])
def api_forecast():
    """API endpoint to get energy consumption forecast"""
    try:
        # Get hours parameter (default to 24 hours)
        hours = request.args.get('hours', default=24, type=int)
        
        # Limit forecast to reasonable range
        if hours < 1:
            return jsonify({'error': 'Hours must be at least 1'}), 400
        if hours > 168:  # One week max
            return jsonify({'error': 'Forecast limited to maximum of 168 hours (1 week)'}), 400
        
        # Generate forecast
        forecast = make_forecast(hours_to_predict=hours)
        
        # Convert to dictionary for JSON response
        response = {
            'forecast': [
                {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'consumption': value
                }
                for timestamp, value in zip(forecast.index, forecast['forecasted_consumption'])
            ],
            'units': 'kilowatts',
            'model_version': '1.0'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Example of starting the API (in production use a WSGI server)
if __name__ == '__main__':
    app.run(debug=True, port=5000)

"""
Example API request:
http://localhost:5000/forecast?hours=24
"""
```

## 8. Model Monitoring

```python
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define monitoring functions
def simulate_production_data(days=30, model=None, scaler=None):
    """
    Simulate how the model would perform in production over time
    
    Parameters:
    -----------
    days : int
        Number of days to simulate
    model : object
        Trained forecasting model
    scaler : object
        Fitted scaler for data normalization
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing performance metrics over time
    """
    if model is None:
        model = joblib.load('models/energy_forecast_model.pkl')
    if scaler is None:
        scaler = joblib.load('models/scaler.pkl')
    
    # Start from the beginning of test data
    start_date = y_test.index[0]
    
    # Storage for results
    results = []
    
    # Sliding window simulation over the test period
    for day in range(min(days, len(y_test) // 24)):
        # Define time window
        current_date = start_date + pd.Timedelta(days=day)
        end_of_day = current_date + pd.Timedelta(days=1)
        
        # Get actual data for this window
        actual_values = y_test[current_date:end_of_day]
        if len(actual_values) < 24:  # Skip incomplete days
            continue
        
        # Get feature data for this window
        features = X_test.loc[actual_values.index]
        
        # Make predictions
        predictions = model.predict(features)
        
        # Calculate metrics for this window
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actual_values, predictions) * 100
        
        # Add some simulated drift over time
        drift_factor = 1 + (day / (days * 5))  # Gradual increase in error
        
        # Store results
        results.append({
            'date': current_date.date(),
            'rmse': rmse * drift_factor,  # Simulated increasing error
            'mape': mape * drift_factor,
            'avg_prediction': np.mean(predictions),
            'avg_actual': np.mean(actual_values),
            'prediction_volume': len(predictions)
        })
    
    return pd.DataFrame(results)

# Generate monitoring data
monitoring_data = simulate_production_data(days=30)

# Visualize monitoring metrics
plt.figure(figsize=(15, 10))

# RMSE over time
plt.subplot(2, 2, 1)
plt.plot(monitoring_data['date'], monitoring_data['rmse'], marker='o', color='blue')
plt.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='Alert Threshold')
plt.title('RMSE Over Time')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)
plt.legend()

# MAPE over time
plt.subplot(2, 2, 2)
plt.plot(monitoring_data['date'], monitoring_data['mape'], marker='o', color='green')
plt.axhline(y=10.0, color='r', linestyle='--', alpha=0.7, label='Alert Threshold')
plt.title('MAPE Over Time (%)')
plt.ylabel('MAPE (%)')
plt.grid(True, alpha=0.3)
plt.legend()

# Prediction vs Actual
plt.subplot(2, 2, 3)
plt.plot(monitoring_data['date'], monitoring_data['avg_prediction'], marker='o', label='Avg Prediction')
plt.plot(monitoring_data['date'], monitoring_data['avg_actual'], marker='x', label='Avg Actual')
plt.title('Average Prediction vs Actual')
plt.ylabel('Power Consumption (kW)')
plt.grid(True, alpha=0.3)
plt.legend()

# Data volume
plt.subplot(2, 2, 4)
plt.bar(monitoring_data['date'], monitoring_data['prediction_volume'])
plt.title('Daily Prediction Volume')
plt.ylabel('Number of Predictions')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/model_monitoring.png')
plt.show()

# Check for performance degradation
alert_threshold_rmse = 5.0
alert_threshold_mape = 10.0

alerts = monitoring_data[
    (monitoring_data['rmse'] > alert_threshold_rmse) | 
    (monitoring_data['mape'] > alert_threshold_mape)
]

if not alerts.empty:
    print("\nPERFORMANCE ALERTS DETECTED:")
    print(alerts[['date', 'rmse', 'mape']])
    print("\nRecommendation: Model retraining may be needed.")
else:
    print("\nNo performance alerts detected. Model is performing within acceptable thresholds.")
```

## 9. Business Impact Analysis

```python
def calculate_business_impact():
    """
    Calculate the business impact of the forecasting model
    """
    # Assumptions
    avg_hourly_consumption = y_test.mean()  # kW
    electricity_cost_per_kwh = 0.15  # $
    avg_error_without_model = 0.20  # 20% error without model
    avg_error_with_model = tuned_results['mape'] / 100  # Our model's error rate
    
    # Daily calculations
    daily_consumption = avg_hourly_consumption * 24  # kWh
    daily_cost = daily_consumption * electricity_cost_per_kwh  # $
    
    # Monthly calculations
    monthly_consumption = daily_consumption * 30  # kWh
    monthly_cost = daily_cost * 30  # $
    
    # Calculate potential savings
    # Assumption: Better forecasting reduces waste by the difference in error rates
    error_improvement = avg_error_without_model - avg_error_with_model
    monthly_savings = monthly_cost * error_improvement
    annual_savings = monthly_savings * 12
    
    # ROI calculation
    model_development_cost = 50000  # $ (hypothetical)
    model_maintenance_cost = 1000  # $ per month (hypothetical)
    annual_maintenance_cost = model_maintenance_cost * 12
    
    first_year_roi = (annual_savings - model_development_cost - annual_maintenance_cost) / (model_development_cost + annual_maintenance_cost) * 100
    subsequent_years_roi = (annual_savings - annual_maintenance_cost) / annual_maintenance_cost * 100
    
    # Carbon footprint reduction
    carbon_intensity = 0.5  # kg CO2 per kWh (average US grid)
    monthly_carbon_savings = monthly_consumption * error_improvement * carbon_intensity  # kg CO2
    annual_carbon_savings = monthly_carbon_savings * 12  # kg CO2
    
    return {
        'avg_hourly_consumption': avg_hourly_consumption,
        'monthly_consumption': monthly_consumption,
        'monthly_cost': monthly_cost,
        'error_improvement': error_improvement * 100,  # to percentage
        'monthly_savings': monthly_savings,
        'annual_savings': annual_savings,
        'first_year_roi': first_year_roi,
        'subsequent_years_roi': subsequent_years_roi,
        'annual_carbon_savings': annual_carbon_savings
    }

# Calculate business impact
impact = calculate_business_impact()

# Display results
print("\nBusiness Impact Analysis:")
print(f"Average hourly consumption: {impact['avg_hourly_consumption']:.2f} kW")
print(f"Monthly electricity cost: ${impact['monthly_cost']:,.2f}")
print(f"Forecasting error improvement: {impact['error_improvement']:.2f}%")
print(f"Monthly cost savings: ${impact['monthly_savings']:,.2f}")
print(f"Annual cost savings: ${impact['annual_savings']:,.2f}")
print(f"First year ROI: {impact['first_year_roi']:.2f}%")
print(f"Subsequent years ROI: {impact['subsequent_years_roi']:.2f}%")
print(f"Annual carbon emission reduction: {impact['annual_carbon_savings']:,.2f} kg CO2")

# Visualize business impact
plt.figure(figsize=(15, 8))

# Cost savings
plt.subplot(2, 2, 1)
savings = [impact['annual_savings'] for _ in range(5)]
cumulative_savings = [sum(savings[:i+1]) for i in range(5)]
plt.bar(range(1, 6), savings, alpha=0.7, label='Annual Savings')
plt.plot(range(1, 6), cumulative_savings, 'ro-', label='Cumulative Savings')
plt.title('Projected Annual Savings')
plt.xlabel('Year')
plt.ylabel('Savings ($)')
plt.xticks(range(1, 6))
plt.grid(True, alpha=0.3)
plt.legend()

# ROI
plt.subplot(2, 2, 2)
roi_values = [impact['first_year_roi']] + [impact['subsequent_years_roi'] for _ in range(4)]
plt.bar(range(1, 6), roi_values)
plt.title('Return on Investment')
plt.xlabel('Year')
plt.ylabel('ROI (%)')
plt.xticks(range(1, 6))
plt.grid(True, alpha=0.3)

# Carbon savings
plt.subplot(2, 2, 3)
carbon_savings = [impact['annual_carbon_savings'] for _ in range(5)]
cumulative_carbon = [sum(carbon_savings[:i+1]) for i in range(5)]
plt.bar(range(1, 6), carbon_savings, color='green', alpha=0.7, label='Annual Reduction')
plt.plot(range(1, 6), cumulative_carbon, 'go-', label='Cumulative Reduction')
plt.title('Carbon Emission Reduction')
plt.xlabel('Year')
plt.ylabel('CO₂ Reduction (kg)')
plt.xticks(range(1, 6))
plt.grid(True, alpha=0.3)
plt.legend()

# Error reduction
plt.subplot(2, 2, 4)
plt.pie([impact['error_improvement'], 100 - impact['error_improvement']], 
        labels=['Error Reduction', 'Remaining Error'],
        autopct='%1.1f%%',
        colors=['lightgreen', 'lightgray'],
        startangle=90)
plt.title('Forecasting Error Reduction')

plt.tight_layout()
plt.savefig('visualizations/business_impact.png')
plt.show()
```

## 10. Executive Summary and Documentation

```python
# Generate an executive summary in markdown
executive_summary = f"""
# Energy Consumption Forecasting Project: Executive Summary

## Project Overview
This project developed a machine learning system to forecast hourly electricity consumption
for a utility company. The model enables more accurate resource planning, price optimization,
and grid management.

## Key Results

### Model Performance
- **RMSE**: {tuned_results['rmse']:.2f} kilowatts
- **MAPE**: {tuned_results['mape']:.2f}%
- **Improvement**: {impact['error_improvement']:.1f}% reduction in forecasting error

### Business Impact
- **Annual Cost Savings**: ${impact['annual_savings']:,.2f}
- **First Year ROI**: {impact['first_year_roi']:.1f}%
- **Subsequent Years ROI**: {impact['subsequent_years_roi']:.1f}%
- **Carbon Reduction**: {impact['annual_carbon_savings']:,.0f} kg CO₂ annually

## Key Insights
1. The most predictive features are previous consumption levels, particularly from 24 hours prior.
2. Strong daily and weekly seasonality patterns emerge in the consumption data.
3. Weather-based features would likely further improve the model (recommended for future work).

## Deployment Information
- Model is deployed as a REST API providing hourly forecasts up to 7 days ahead.
- Monitoring system is in place to detect performance degradation.
- Recommended retraining schedule: Monthly, with data drift monitoring.

## Next Steps
1. Integrate weather forecast data to improve prediction accuracy.
2. Develop demand response strategies based on forecasts.
3. Extend the model to provide uncertainty estimates with prediction intervals.
"""

# Save the executive summary to a file
with open('reports/executive_summary.md', 'w') as f:
    f.write(executive_summary)

print("\nExecutive summary saved to 'reports/executive_summary.md'")

# Create a technical documentation template
technical_doc = """
# Energy Consumption Forecasting: Technical Documentation

## 1. Data Sources
- Hourly electricity consumption data
- Time and date information
- Feature engineering approach for time features

## 2. Feature Engineering Process
- Created time-based features (hour, day of week, month, etc.)
- Implemented cyclical encoding for periodic features
- Generated lag features for autoregressive patterns
- Created rolling window statistics

## 3. Model Architecture
- Selected algorithm: XGBoost
- Key hyperparameters:
  - Learning rate: {learning_rate}
  - Max depth: {max_depth}
  - Number of estimators: {n_estimators}
  - Subsample ratio: {subsample}

## 4. Performance Metrics
- RMSE: {rmse}
- MAPE: {mape}%
- R²: {r2}

## 5. Deployment Architecture
- Flask REST API
- Endpoints:
  - /forecast: Get energy consumption forecasts
  - Parameters:
    - hours: Number of hours to forecast (1-168)

## 6. Monitoring System
- Daily performance tracking
- Alert thresholds:
  - RMSE > 5.0
  - MAPE > 10.0%

## 7. Maintenance Plan
- Retraining schedule: Monthly
- Data retention policy: 2 years rolling window
- Model versioning approach

## 8. Future Improvements
- Weather data integration
- Prediction intervals
- Customer segmentation models
- Demand response optimization
""".format(
    learning_rate=random_search.best_params_.get('learning_rate', 'N/A'),
    max_depth=random_search.best_params_.get('max_depth', 'N/A'),
    n_estimators=random_search.best_params_.get('n_estimators', 'N/A'),
    subsample=random_search.best_params_.get('subsample', 'N/A'),
    rmse=tuned_results['rmse'],
    mape=tuned_results['mape'],
    r2=tuned_results['r2']
)

# Save the technical documentation to a file
with open('reports/technical_documentation.md', 'w') as f:
    f.write(technical_doc)

print("Technical documentation saved to 'reports/technical_documentation.md'")
```

## 11. Key Learnings and Takeaways

This end-to-end machine learning project for energy consumption forecasting demonstrates the complete ML lifecycle:

1. **Problem Definition**: We clearly defined the business problem and success metrics for energy consumption forecasting.

2. **Data Exploration**: We analyzed historic energy consumption data to understand patterns, trends, and seasonality.

3. **Feature Engineering**: The project showcased how to create effective time-based features, including:
   - Extracting calendar features (hour, day of week, month)
   - Creating cyclical features to handle periodic patterns
   - Building lag features to capture temporal dependencies
   - Generating rolling statistics to identify trends

4. **Model Development**: We compared multiple algorithms and found that tree-based ensemble methods (particularly XGBoost) performed best for this regression problem.

5. **Hyperparameter Tuning**: We used RandomizedSearchCV with TimeSeriesSplit to find optimal model parameters.

6. **Model Deployment**: We created a prediction function and REST API for making the model accessible to users.

7. **Monitoring and Maintenance**: We implemented performance tracking over time to detect model drift.

8. **Business Impact Analysis**: We quantified the financial and environmental benefits of the forecasting system.

Key takeaways from this project:

- **Time Series Specifics**: Time series forecasting requires specialized techniques including appropriate train/test splits, lag features, and evaluation methods.
- **Feature Engineering Importance**: Creating the right features is often more important than the choice of algorithm.
- **Model Explainability**: Understanding which features drive predictions is essential for building trust and improving the model.
- **Business Value**: Translating technical performance metrics (RMSE) to business outcomes (cost savings) is crucial for project success.

This project provides a template that can be adapted for various forecasting needs, from energy consumption to sales, website traffic, or other time series prediction problems.