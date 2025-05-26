#!/bin/bash
# Setup script for energy consumption forecasting

echo "Setting up environment for energy consumption forecasting..."

# Create a directory for the project
mkdir -p energy_forecasting
cd energy_forecasting

# Create a virtual environment (optional)
echo "Creating Python virtual environment..."
python -m venv env

# Activate the virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source env/Scripts/activate
else
    source env/bin/activate
fi

# Install required packages
echo "Installing required packages..."
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

# Copy the Python scripts into the directory
echo "Creating Python scripts..."

# Create the data generator script
cat > generate_data.py << 'EOL'
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def generate_synthetic_energy_data(start_date='2020-01-01', end_date='2022-12-31', 
                                  random_seed=42):
    """
    Generate synthetic building energy consumption data
    
    Parameters:
    - start_date: Beginning date for the dataset (string)
    - end_date: End date for the dataset (string)
    - random_seed: Seed for reproducibility
    
    Returns:
    - DataFrame with hourly energy consumption data
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create date range with hourly frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    n_samples = len(date_range)
    
    # Initialize dataframe
    df = pd.DataFrame(index=date_range)
    df.index.name = 'timestamp'
    
    # Add temporal features first (we'll use these to generate other features)
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Generate realistic outdoor temperature with seasonal patterns
    # Base seasonal pattern
    seasonal_temp = 15 - 15 * np.cos(2 * np.pi * (df.index.dayofyear / 365))
    
    # Add daily variations (warmer during day, cooler at night)
    hour_temp = 5 * np.sin(np.pi * (df['hour'] - 4) / 24)
    
    # Add some random noise
    temp_noise = np.random.normal(0, 2, n_samples)
    
    # Combine components
    df['outdoor_temperature'] = seasonal_temp + hour_temp + temp_noise
    
    # Generate energy consumption with realistic patterns
    
    # Base load (always present)
    base_load = 20 + np.random.normal(0, 2, n_samples)
    
    # Business hours load (higher during working hours on weekdays)
    business_mask = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['day_of_week'] < 5))
    business_load = np.zeros(n_samples)
    business_load[business_mask] = 30 + np.random.normal(0, 5, sum(business_mask))
    
    # Temperature-dependent load (HVAC)
    # Heating (when cold)
    heating_load = np.maximum(0, (18.5 - df['outdoor_temperature']) * 2)
    
    # Cooling (when hot)
    cooling_load = np.maximum(0, (df['outdoor_temperature'] - 21) * 3)
    
    # Seasonal adjustments (more energy use in winter months)
    seasonal_factor = 1.2 + 0.3 * np.cos(2 * np.pi * (df.index.dayofyear / 365))
    
    # Time of day factor (varies throughout day)
    time_factor = 1 + 0.5 * np.sin(np.pi * (df['hour'] - 2) / 12)
    
    # Weekend reduction
    weekend_factor = 0.7 * df['is_weekend'] + 1.0 * (1 - df['is_weekend'])
    
    # Combine all factors for energy consumption
    df['energy_consumption'] = (base_load + business_load + heating_load + cooling_load) * \
                               seasonal_factor * time_factor * weekend_factor
    
    # Add random noise to make it more realistic
    df['energy_consumption'] = df['energy_consumption'] * (1 + np.random.normal(0, 0.05, n_samples))
    
    # Add humidity
    base_humidity = 50 + 20 * np.sin(2 * np.pi * (df.index.dayofyear / 365 + 0.5))
    humidity_noise = np.random.normal(0, 5, n_samples)
    df['humidity'] = base_humidity + humidity_noise
    df['humidity'] = df['humidity'].clip(20, 95)  # Realistic range
    
    # Add cloud cover (0-100%)
    df['cloud_cover'] = np.random.beta(2, 3, n_samples) * 100
    
    # Add wind speed (m/s)
    df['wind_speed'] = np.random.gamma(2, 2, n_samples)
    
    # Add occupancy
    df['occupancy'] = 0
    
    # Weekday occupancy pattern
    weekday_mask = (df['day_of_week'] < 5)
    morning_arrival = (df['hour'] >= 7) & (df['hour'] < 10) & weekday_mask
    working_hours = (df['hour'] >= 10) & (df['hour'] < 17) & weekday_mask
    evening_departure = (df['hour'] >= 17) & (df['hour'] <= 19) & weekday_mask
    
    df.loc[morning_arrival, 'occupancy'] = 0.7 * (1 + 0.3 * np.random.random(sum(morning_arrival)))
    df.loc[working_hours, 'occupancy'] = 0.9 * (1 + 0.1 * np.random.random(sum(working_hours)))
    df.loc[evening_departure, 'occupancy'] = 0.5 * (1 + 0.3 * np.random.random(sum(evening_departure)))
    
    # Weekend occupancy (much lower)
    weekend_hours = (df['hour'] >= 10) & (df['hour'] <= 15) & df['is_weekend'].astype(bool)
    df.loc[weekend_hours, 'occupancy'] = 0.2 * (1 + 0.5 * np.random.random(sum(weekend_hours)))
    
    # Ensure values are realistic and positive
    df['energy_consumption'] = df['energy_consumption'].clip(lower=10)
    
    # Add some missing values to simulate real-world data issues
    for col in ['outdoor_temperature', 'humidity', 'wind_speed']:
        missing_idx = random.sample(range(n_samples), int(n_samples * 0.01))  # 1% missing data
        df.loc[df.index[missing_idx], col] = np.nan
    
    # Round values to make them more realistic
    df['energy_consumption'] = df['energy_consumption'].round(2)
    df['outdoor_temperature'] = df['outdoor_temperature'].round(1)
    df['humidity'] = df['humidity'].round(1)
    df['wind_speed'] = df['wind_speed'].round(1)
    df['cloud_cover'] = df['cloud_cover'].round(1)
    
    # Keep only the columns needed for the regression task
    final_df = df[['energy_consumption', 'outdoor_temperature', 'humidity', 
                  'wind_speed', 'cloud_cover', 'occupancy']]
    
    # Reset index to make timestamp a column
    final_df = final_df.reset_index()
    
    return final_df

if __name__ == "__main__":
    print("Generating synthetic building energy data...")
    df = generate_synthetic_energy_data(start_date='2020-01-01', end_date='2022-12-31')
    
    # Save to CSV
    df.to_csv('building_energy_data.csv', index=False)
    
    print(f"Generated dataset with {len(df)} rows covering {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Data columns: {', '.join(df.columns)}")
    print("Saved to building_energy_data.csv")
EOL

# Generate data first
echo "Generating synthetic energy data..."
python generate_data.py

echo "Data generation complete. You're all set!"
echo "Run the forecasting code with:"
echo "python energy_consumption_forecasting.py"
