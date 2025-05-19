# Case Study: End-to-End ML Project - Retail Sales Prediction

## 1. Problem Definition and Business Context

Predicting future sales is crucial for retail businesses to optimize inventory, staffing, and marketing strategies. This case study focuses on building a robust machine learning model to forecast store sales based on historical data.

```python
"""
Retail Sales Prediction Project

Business Problem: A retail chain with multiple stores needs to forecast daily sales
to improve inventory management and optimize staffing levels.

Key Business Questions:
1. What will be the sales volume for each store in the next 6 weeks?
2. Which factors most strongly influence sales performance?
3. How do promotions and holidays affect sales patterns?

Success Metrics:
- RMSE (Root Mean Square Error) < 15% of average daily sales
- Accurate prediction of sales trends during promotional periods
- Interpretable results to guide business decisions
"""
```

## 2. Dataset Exploration

For this project, we'll use a modified version of a popular retail sales dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the dataset
sales_df = pd.read_csv('store_sales.csv')
store_df = pd.read_csv('store_info.csv')

# Display basic information
print(f"Sales data shape: {sales_df.shape}")
print("\nSales data first 5 rows:")
print(sales_df.head())

print(f"\nStore information shape: {store_df.shape}")
print("\nStore information first 5 rows:")
print(store_df.head())
```

Let's explore the data structure and key characteristics:

```python
# Convert date to datetime
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

# Basic statistics
print("\nSales data summary statistics:")
print(sales_df.describe())

# Check for missing values
print("\nMissing values in sales data:")
print(sales_df.isnull().sum())

print("\nMissing values in store data:")
print(store_df.isnull().sum())

# Check data types
print("\nSales data types:")
print(sales_df.dtypes)

# Merge store information with sales data
df = pd.merge(sales_df, store_df, on='Store', how='left')
print("\nMerged data first 5 rows:")
print(df.head())

# Aggregate total sales by date
daily_sales = df.groupby('Date')['Sales'].sum().reset_index()

# Plot time series of sales
plt.figure(figsize=(15, 6))
plt.plot(daily_sales['Date'], daily_sales['Sales'])
plt.title('Total Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

# Sales distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Sales'], bins=50, kde=True)
plt.title('Distribution of Daily Sales')
plt.xlabel('Sales')
plt.tight_layout()
plt.show()
```

### Exploring Key Variables

```python
# Sales by store type
plt.figure(figsize=(12, 6))
sns.boxplot(x='StoreType', y='Sales', data=df)
plt.title('Sales Distribution by Store Type')
plt.tight_layout()
plt.show()

# Effect of promotions on sales
plt.figure(figsize=(10, 6))
sns.barplot(x='Promo', y='Sales', data=df)
plt.title('Impact of Promotions on Sales')
plt.xlabel('Promotion (0 = No, 1 = Yes)')
plt.ylabel('Average Sales')
plt.tight_layout()
plt.show()

# Effect of holidays on sales
plt.figure(figsize=(10, 6))
sns.barplot(x='StateHoliday', y='Sales', data=df)
plt.title('Impact of State Holidays on Sales')
plt.xlabel('Holiday Type (0 = None, a = Public Holiday, b = Easter, c = Christmas)')
plt.ylabel('Average Sales')
plt.tight_layout()
plt.show()

# Sales by day of week
df['DayOfWeek'] = df['Date'].dt.dayofweek
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(12, 6))
sns.barplot(x='DayOfWeek', y='Sales', data=df)
plt.title('Average Sales by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.xticks(range(7), day_names)
plt.tight_layout()
plt.show()

# Correlation analysis
correlation_vars = ['Sales', 'Customers', 'Promo', 'DayOfWeek', 'CompetitionDistance']
correlation = df[correlation_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

## 3. Data Preprocessing and Feature Engineering

### Handling Missing Values and Outliers

```python
# Check for missing Competition Distance values
print(f"Missing Competition Distance: {df['CompetitionDistance'].isnull().sum()}")

# Fill missing CompetitionDistance with median
median_distance = df['CompetitionDistance'].median()
df['CompetitionDistance'].fillna(median_distance, inplace=True)

# Handle outliers in sales data (using IQR method)
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)]
print(f"\nNumber of outliers in Sales: {len(outliers)}")

# We'll cap outliers instead of removing them
df['Sales'] = df['Sales'].clip(lower=lower_bound, upper=upper_bound)
```

### Feature Engineering

```python
# Extract datetime features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Create month-end and month-start flags
df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)

# Create weekend flag
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'StateHoliday'], drop_first=True)

# Create promotion duration features
df['PromoOpen'] = df['Promo'].astype(int)
df['PromoDays'] = df.groupby('Store')['Promo'].transform(lambda x: x.cumsum())

# Create rolling average sales features (for each store)
df_sorted = df.sort_values(['Store', 'Date'])
df_sorted['Sales_7d_avg'] = df_sorted.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df_sorted['Sales_14d_avg'] = df_sorted.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
df_sorted['Sales_30d_avg'] = df_sorted.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())

# Fill NaN values created by rolling and lag features
for col in ['Sales_7d_avg', 'Sales_14d_avg', 'Sales_30d_avg']:
    df_sorted[col].fillna(df_sorted.groupby('Store')['Sales'].transform('mean'), inplace=True)

# Update our main dataframe
df = df_sorted

# Display the new features
print("\nDataframe with engineered features:")
print(df.head())
```

### Feature Selection and Data Preparation

```python
from sklearn.model_selection import train_test_split

# Drop unnecessary columns for modeling
features_to_drop = ['Date', 'Customers']  # In a real scenario, we wouldn't have Customers in test data
df_model = df.drop(features_to_drop, axis=1)

# Define target variable and features
X = df_model.drop('Sales', axis=1)
y = df_model['Sales']

# Create training set (data before 2017) and test set (2017 data)
# In this example, we'll use a simple train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
```

## 4. Model Selection and Training

### Training Simple Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Define models to test
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2, y_pred

# Train and evaluate each model
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    rmse, mae, r2, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2,
        'Predictions': y_pred
    }
    
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

# Compare models visually
model_names = list(results.keys())
rmse_values = [results[model]['RMSE'] for model in model_names]
mae_values = [results[model]['MAE'] for model in model_names]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(model_names, rmse_values)
plt.title('RMSE by Model')
plt.xticks(rotation=45, ha='right')
plt.ylabel('RMSE (lower is better)')

plt.subplot(1, 2, 2)
plt.bar(model_names, mae_values)
plt.title('MAE by Model')
plt.xticks(rotation=45, ha='right')
plt.ylabel('MAE (lower is better)')

plt.tight_layout()
plt.show()

# Select the best performing model for further analysis (based on RMSE)
best_model_name = model_names[np.argmin(rmse_values)]
print(f"\nBest performing model: {best_model_name}")
```

### Visualizing Predictions and Residuals

```python
# Get actual and predicted values for the best model
y_pred = results[best_model_name]['Predictions']

plt.figure(figsize=(15, 6))

# Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'{best_model_name}: Actual vs Predicted')

# Residuals
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title(f'{best_model_name}: Residuals Plot')

plt.tight_layout()
plt.show()

# Distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30)
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True)
plt.show()
```

### Feature Importance Analysis

```python
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get the best model
    best_model = models[best_model_name]
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance[:15])
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
```

## 5. Model Improvement - Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV, KFold, TimeSeriesSplit
import numpy as np

# Assuming Gradient Boosting was the best model
if best_model_name == 'Gradient Boosting':
    # Define hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Create cross-validation strategy
    # For time series data, it's better to use TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create random search
    random_search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter combinations to try
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Perform random search
    print("Training hyperparameter tuning. This may take some time...")
    random_search.fit(X_train, y_train)
    
    # Best parameters and score
    print("\nBest hyperparameters:")
    print(random_search.best_params_)
    print(f"Best RMSE: {-random_search.best_score_:.2f}")
    
    # Train model with best parameters
    best_gb = random_search.best_estimator_
    
    # Evaluate
    y_pred_tuned = best_gb.predict(X_test)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    
    print(f"\nTuned Gradient Boosting - RMSE: {rmse_tuned:.2f}, MAE: {mae_tuned:.2f}, R²: {r2_tuned:.4f}")
    
    # Compare with base model
    print(f"RMSE Improvement: {results['Gradient Boosting']['RMSE'] - rmse_tuned:.2f}")
    print(f"Percent Improvement: {(results['Gradient Boosting']['RMSE'] - rmse_tuned) / results['Gradient Boosting']['RMSE'] * 100:.2f}%")
    
    # Update best model for deployment
    best_model = best_gb
elif best_model_name == 'Random Forest':
    # Define hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }
    
    # Create cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create random search
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_grid,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Perform random search
    print("Training hyperparameter tuning. This may take some time...")
    random_search.fit(X_train, y_train)
    
    # Best parameters and score
    print("\nBest hyperparameters:")
    print(random_search.best_params_)
    print(f"Best RMSE: {-random_search.best_score_:.2f}")
    
    # Train model with best parameters
    best_rf = random_search.best_estimator_
    
    # Evaluate
    y_pred_tuned = best_rf.predict(X_test)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    
    print(f"\nTuned Random Forest - RMSE: {rmse_tuned:.2f}, MAE: {mae_tuned:.2f}, R²: {r2_tuned:.4f}")
    
    # Compare with base model
    print(f"RMSE Improvement: {results['Random Forest']['RMSE'] - rmse_tuned:.2f}")
    print(f"Percent Improvement: {(results['Random Forest']['RMSE'] - rmse_tuned) / results['Random Forest']['RMSE'] * 100:.2f}%")
    
    # Update best model for deployment
    best_model = best_rf
```

## 6. Advanced Feature Engineering and Model Ensembling

```python
import lightgbm as lgb
import xgboost as xgb

# Create additional features based on insights
# Let's assume we found that promotional effects vary by store type
df['Promo_StoreTypeA'] = df['Promo'] * df['StoreType_a']
df['Promo_StoreTypeB'] = df['Promo'] * df['StoreType_b']
df['Promo_StoreTypeC'] = df['Promo'] * df['StoreType_c']

# Create holiday period flags (days before and after holidays)
for offset in [-1, 1, 2]:
    col_name = f'StateHoliday_shifted_{offset}'
    df[col_name] = df.groupby('Store')['StateHoliday_a'].shift(offset).fillna(0)

# Create month-based seasonal features using Fourier transforms
for n in [1, 2, 3]:
    df[f'sin_month_{n}'] = np.sin(2 * np.pi * n * df['Month'] / 12)
    df[f'cos_month_{n}'] = np.cos(2 * np.pi * n * df['Month'] / 12)

# Prepare the new dataset for modeling
df_model_enhanced = df.drop(['Date', 'Customers'], axis=1)

X_new = df_model_enhanced.drop('Sales', axis=1)
y_new = df_model_enhanced['Sales']

# Use the same train-test split approach
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Train additional models
print("Training LightGBM model...")
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=300,
    random_state=42
)
lgb_model.fit(X_train_new, y_train_new)

print("Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=7,
    learning_rate=0.05,
    n_estimators=300,
    random_state=42
)
xgb_model.fit(X_train_new, y_train_new)

# Make predictions with all models
y_pred_gb = best_model.predict(X_test_new)
y_pred_lgb = lgb_model.predict(X_test_new)
y_pred_xgb = xgb_model.predict(X_test_new)

# Create an ensemble prediction (simple averaging)
y_pred_ensemble = (y_pred_gb + y_pred_lgb + y_pred_xgb) / 3

# Evaluate all models and the ensemble
print("\nIndividual model performance:")
print(f"Tuned Gradient Boosting - RMSE: {np.sqrt(mean_squared_error(y_test_new, y_pred_gb)):.2f}")
print(f"LightGBM - RMSE: {np.sqrt(mean_squared_error(y_test_new, y_pred_lgb)):.2f}")
print(f"XGBoost - RMSE: {np.sqrt(mean_squared_error(y_test_new, y_pred_xgb)):.2f}")
print(f"Ensemble - RMSE: {np.sqrt(mean_squared_error(y_test_new, y_pred_ensemble)):.2f}")

# Visualize ensemble predictions
plt.figure(figsize=(12, 6))
plt.scatter(y_test_new, y_pred_ensemble, alpha=0.5)
plt.plot([y_test_new.min(), y_test_new.max()], [y_test_new.min(), y_test_new.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Ensemble Model: Actual vs Predicted')
plt.tight_layout()
plt.show()
```

## 7. Model Deployment

```python
import joblib
import os

# Create a directory for model artifacts
os.makedirs('models', exist_ok=True)

# Save the final model and preprocessing artifacts
joblib.dump(best_model, 'models/sales_prediction_model.pkl')
print("Model saved to models/sales_prediction_model.pkl")

# Save any preprocessing objects if needed (like scalers)
# For this example, we'll just save a list of feature names to ensure consistency
feature_names = X_train_new.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')

# Create a function for making predictions on new data
def predict_sales(store_data, model_path='models/sales_prediction_model.pkl', features_path='models/feature_names.pkl'):
    """
    Make sales predictions for new store data.
    
    Parameters:
    -----------
    store_data : pd.DataFrame
        Dataframe containing store features
    model_path : str
        Path to saved model file
    features_path : str
        Path to saved feature names file
        
    Returns:
    --------
    pd.Series
        Predicted sales values
    """
    # Load the model and feature names
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # Ensure input data has all required features
    missing_features = set(feature_names) - set(store_data.columns)
    if missing_features:
        raise ValueError(f"Input data is missing required features: {missing_features}")
    
    # Select and order features correctly
    X = store_data[feature_names]
    
    # Make predictions
    predictions = model.predict(X)
    
    return predictions

# Example of using the prediction function
# (In a real scenario, this would be new data)
print("\nExample prediction using the first 5 test samples:")
sample_data = X_test_new.iloc[:5]
predictions = predict_sales(sample_data)
actuals = y_test_new.iloc[:5].values

print("\nPredictions vs Actuals:")
for i, (pred, actual) in enumerate(zip(predictions, actuals)):
    print(f"Sample {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={((pred-actual)/actual)*100:.2f}%")
```

### Creating a Simple Flask API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data)
        
        # Make predictions
        predictions = predict_sales(input_df)
        
        # Return predictions
        return jsonify({
            'predictions': predictions.tolist(),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

# Example of how to run the API (in a production environment, use a proper WSGI server)
if __name__ == '__main__':
    app.run(debug=True, port=8000)

# Example API request:
"""
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "Store": [1, 2, 3],
           "DayOfWeek": [1, 2, 3],
           "Promo": [1, 0, 1],
           "StateHoliday_a": [0, 0, 0],
           ...
         }'
"""
```

### Batch Prediction Script

```python
def batch_predict_sales(input_file, output_file, model_path='models/sales_prediction_model.pkl', features_path='models/feature_names.pkl'):
    """
    Generate sales predictions for a batch of stores and save to a CSV file.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file with store features
    output_file : str
        Path to save predictions
    model_path : str
        Path to saved model file
    features_path : str
        Path to saved feature names file
    """
    # Load input data
    store_data = pd.read_csv(input_file)
    
    # Load model and feature names
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # Ensure all required features exist
    missing_features = set(feature_names) - set(store_data.columns)
    if missing_features:
        for feature in missing_features:
            print(f"Warning: Adding missing feature '{feature}' with zeros")
            store_data[feature] = 0
    
    # Select features in the correct order
    X = store_data[feature_names]
    
    # Make predictions
    predictions = model.predict(X)
    
    # Add predictions to the original data
    store_data['PredictedSales'] = predictions
    
    # Save results
    store_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return store_data

# Example usage:
# batch_predict_sales('new_store_data.csv', 'sales_predictions.csv')
```

## 8. Model Monitoring and Maintenance

```python
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to simulate model drift over time
def simulate_model_performance_over_time(model, X, y, n_periods=12):
    """
    Simulate how model performance might change over time due to data drift.
    
    Parameters:
    -----------
    model : trained model
        The model to evaluate
    X : DataFrame
        Feature data
    y : Series
        Target data
    n_periods : int
        Number of time periods to simulate
        
    Returns:
    --------
    DataFrame
        Performance metrics over time
    """
    # Initialize results storage
    results = []
    
    # Create base date
    base_date = dt.datetime(2022, 1, 1)
    
    # Simulate drift by gradually modifying the data
    for i in range(n_periods):
        # Calculate current date
        current_date = base_date + dt.timedelta(days=30*i)
        
        # Create a drift factor that increases over time
        drift_factor = 1 + (i / (n_periods * 2))
        
        # Apply drift to the features (for simplicity, we'll just scale some numeric features)
        X_drifted = X.copy()
        numeric_cols = X_drifted.select_dtypes(include=np.number).columns[:3]  # First 3 numeric columns
        for col in numeric_cols:
            X_drifted[col] = X_drifted[col] * drift_factor
        
        # Make predictions with the drifted data
        y_pred = model.predict(X_drifted)
        
        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Add simulated data volume (more data over time)
        data_volume = int(1000 + i * 100)
        
        # Store results
        results.append({
            'date': current_date,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'drift_factor': drift_factor,
            'data_volume': data_volume
        })
    
    return pd.DataFrame(results)

# Simulate model performance over time
monitoring_data = simulate_model_performance_over_time(best_model, X_test_new, y_test_new, n_periods=24)

# Visualize monitoring metrics
plt.figure(figsize=(15, 10))

# RMSE over time
plt.subplot(2, 2, 1)
plt.plot(monitoring_data['date'], monitoring_data['rmse'], marker='o', color='red')
plt.title('RMSE Over Time')
plt.xlabel('Date')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)

# R² over time
plt.subplot(2, 2, 2)
plt.plot(monitoring_data['date'], monitoring_data['r2'], marker='o', color='blue')
plt.title('R² Over Time')
plt.xlabel('Date')
plt.ylabel('R²')
plt.grid(True, alpha=0.3)

# Performance vs Drift
plt.subplot(2, 2, 3)
plt.scatter(monitoring_data['drift_factor'], monitoring_data['rmse'], alpha=0.7)
plt.plot(monitoring_data['drift_factor'], monitoring_data['rmse'], 'r--')
plt.title('Performance vs Data Drift')
plt.xlabel('Drift Factor')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)

# Data volume over time
plt.subplot(2, 2, 4)
plt.bar(monitoring_data['date'], monitoring_data['data_volume'])
plt.title('Data Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Records')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_monitoring.png')
plt.show()

# Detect when model needs retraining
rmse_threshold = monitoring_data['rmse'].iloc[0] * 1.2  # 20% increase in RMSE
retraining_dates = monitoring_data[monitoring_data['rmse'] > rmse_threshold]

if not retraining_dates.empty:
    retraining_date = retraining_dates['date'].iloc[0]
    print(f"\nModel retraining recommended after: {retraining_date.strftime('%Y-%m-%d')}")
    print(f"RMSE increased by {((retraining_dates['rmse'].iloc[0] / monitoring_data['rmse'].iloc[0]) - 1) * 100:.1f}% since baseline")
else:
    print("\nNo significant model degradation detected within the simulation period.")
```

## 9. Business Impact Analysis

```python
# Function to calculate business impact
def calculate_business_impact(baseline_accuracy, improved_accuracy, average_daily_sales, n_stores):
    """
    Calculate the business impact of improved sales forecasting
    
    Parameters:
    -----------
    baseline_accuracy : float
        Accuracy of the baseline method (expressed as error percentage)
    improved_accuracy : float
        Accuracy of the improved model (expressed as error percentage)
    average_daily_sales : float
        Average daily sales per store
    n_stores : int
        Number of stores in the chain
        
    Returns:
    --------
    dict
        Dictionary with business impact metrics
    """
    # Assumed business parameters
    inventory_carrying_cost = 0.25  # 25% annual cost of inventory
    stockout_cost = 0.5  # 50% of potential sales lost due to stockouts
    
    # Calculate daily inventory error in dollars
    baseline_error_dollars = average_daily_sales * baseline_accuracy
    improved_error_dollars = average_daily_sales * improved_accuracy
    error_reduction_dollars = baseline_error_dollars - improved_error_dollars
    
    # Calculate annual savings across all stores
    annual_carrying_cost_savings = error_reduction_dollars * inventory_carrying_cost * 365 * n_stores
    annual_stockout_reduction = error_reduction_dollars * stockout_cost * 365 * n_stores
    total_annual_savings = annual_carrying_cost_savings + annual_stockout_reduction
    
    # Calculate ROI (assuming model development and maintenance costs)
    model_development_cost = 100000  # Hypothetical cost
    annual_maintenance_cost = 25000  # Hypothetical cost
    
    first_year_roi = (total_annual_savings - model_development_cost - annual_maintenance_cost) / (model_development_cost + annual_maintenance_cost) * 100
    subsequent_years_roi = (total_annual_savings - annual_maintenance_cost) / annual_maintenance_cost * 100
    
    return {
        'error_reduction_percentage': (baseline_accuracy - improved_accuracy) / baseline_accuracy * 100,
        'annual_carrying_cost_savings': annual_carrying_cost_savings,
        'annual_stockout_reduction': annual_stockout_reduction,
        'total_annual_savings': total_annual_savings,
        'first_year_roi': first_year_roi,
        'subsequent_years_roi': subsequent_years_roi
    }

# Calculate impact using the model results
# Convert RMSE to percentage of average sales for easier interpretation
average_sales = y_test.mean()
baseline_error_pct = results['Linear Regression']['RMSE'] / average_sales
improved_error_pct = rmse_tuned / average_sales

impact = calculate_business_impact(
    baseline_accuracy=baseline_error_pct,
    improved_accuracy=improved_error_pct,
    average_daily_sales=average_sales,
    n_stores=100  # Assuming 100 stores in the chain
)

# Display results
print("\nBusiness Impact Analysis:")
print(f"Error reduction: {impact['error_reduction_percentage']:.1f}%")
print(f"Annual inventory carrying cost savings: ${impact['annual_carrying_cost_savings']:,.2f}")
print(f"Annual stockout reduction value: ${impact['annual_stockout_reduction']:,.2f}")
print(f"Total annual savings: ${impact['total_annual_savings']:,.2f}")
print(f"First year ROI: {impact['first_year_roi']:.1f}%")
print(f"ROI in subsequent years: {impact['subsequent_years_roi']:.1f}%")

# Visualize business impact
plt.figure(figsize=(15, 8))

# Error reduction
plt.subplot(2, 2, 1)
plt.bar(['Baseline Model', 'Improved Model'], 
        [baseline_error_pct * 100, improved_error_pct * 100])
plt.title('Forecast Error Reduction')
plt.ylabel('Error Percentage (%)')
plt.grid(True, alpha=0.3, axis='y')

# Cost savings breakdown
plt.subplot(2, 2, 2)
savings = [impact['annual_carrying_cost_savings'], impact['annual_stockout_reduction']]
plt.pie(savings, 
        labels=['Inventory Carrying Cost', 'Stockout Prevention'],
        autopct='%1.1f%%',
        startangle=90)
plt.title('Cost Savings Breakdown')

# ROI
plt.subplot(2, 2, 3)
plt.bar(['First Year', 'Subsequent Years'], 
        [impact['first_year_roi'], impact['subsequent_years_roi']])
plt.title('Return on Investment')
plt.ylabel('ROI (%)')
plt.grid(True, alpha=0.3, axis='y')

# Cumulative savings over 5 years
plt.subplot(2, 2, 4)
years = range(1, 6)
first_year_savings = impact['total_annual_savings'] - 100000  # Subtract initial development cost
yearly_savings = [first_year_savings] + [impact['total_annual_savings'] - 25000] * 4  # Subtract maintenance cost
cumulative_savings = np.cumsum(yearly_savings)

plt.bar(years, yearly_savings, label='Annual Net Savings')
plt.plot(years, cumulative_savings, 'ro-', label='Cumulative Savings')
plt.title('5-Year Financial Impact')
plt.xlabel('Year')
plt.ylabel('Savings ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('business_impact.png')
plt.show()
```

## 10. Key Learnings and Best Practices

This end-to-end retail sales prediction project has demonstrated the full machine learning lifecycle, from problem definition to deployment and monitoring. Let's summarize the key learnings:

### 1. Data Understanding and Preparation
- Time series data requires special handling, including proper train-test splitting and feature engineering
- Creating lag features and rolling averages significantly improves forecasting accuracy
- Transforming cyclical features (like month, day of week) using sine/cosine encoding preserves their circular nature
- Domain knowledge is crucial for effective feature engineering (e.g., understanding how promotions and holidays affect sales)

### 2. Model Development
- Tree-based algorithms (Random Forest, Gradient Boosting) typically outperform linear models for retail sales prediction
- Ensemble methods combining multiple algorithms can further improve performance
- Hyperparameter tuning provides significant accuracy improvements, especially for complex models
- Feature importance analysis helps identify key drivers of sales, informing both the model and business strategy

### 3. Deployment and Monitoring
- Creating a simple prediction API allows for integration with business systems
- Batch prediction capability supports regular forecasting processes
- Model monitoring is essential as performance degrades over time due to data drift
- Regular retraining is needed to maintain accuracy (typically every 3-6 months)

### 4. Business Value
- Improved forecast accuracy directly translates to inventory cost savings and reduced stockouts
- The ROI of an ML-based forecasting system is typically excellent, even accounting for development costs
- The greatest business value comes from sustained use and continuous improvement of the model

### Best Practices:
1. Always maintain a validation dataset that represents the future prediction scenario
2. Document data transformations to ensure they can be replicated in production
3. Test model performance across different store types and time periods to ensure robustness
4. Create explainable models and visualizations for business stakeholders
5. Design for automation and easy retraining when new data becomes available
6. Monitor not just model performance, but also input data quality and distribution
7. Build feedback loops to continuously improve the model based on actual vs. predicted results

This case study provides a foundation for tackling retail sales prediction problems using machine learning, a common and valuable application with direct business impact.