# Case Study: End-to-End ML Project - Diabetes Progression Prediction

## 1. Problem Definition and Medical Context

Diabetes is a chronic metabolic disorder affecting millions of people worldwide. Early prediction of disease progression can help healthcare providers tailor treatment plans and improve patient outcomes. In this case study, we'll develop a machine learning model to predict diabetes progression based on patient health metrics.

```python
"""
Project: Diabetes Progression Prediction

Clinical Goal: Develop a machine learning model to predict diabetes progression 
               in patients based on medical measurements.

Medical Significance:
- Early identification of high-risk patients who may need intervention
- Personalized treatment planning based on predicted disease trajectory
- Resource allocation optimization for healthcare systems

Success Metrics:
- Model R² > 0.4 (comparable to medical literature)
- Mean Absolute Error < 50 units for progression metric
- Clinically interpretable results
"""
```

## 2. Dataset and Medical Features

We'll use the diabetes dataset from scikit-learn, which contains data from diabetes patients. The target is a quantitative measure of disease progression one year after baseline.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Create a DataFrame with feature names
feature_names = diabetes.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['disease_progression'] = y

print(f"Dataset shape: {df.shape}")
print("\nFeature names:")
for i, name in enumerate(feature_names):
    print(f"  {i+1}. {name}")

# Display first few rows
print("\nFirst 5 rows of data:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())
```

### Understanding the Medical Features

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

# Print feature descriptions
print("Medical Feature Descriptions:")
for feature, description in feature_descriptions.items():
    print(f"{feature:20} {description}")
```

## 3. Exploratory Data Analysis for Medical Insights

```python
# Distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['disease_progression'], bins=30, kde=True)
plt.title('Distribution of Diabetes Progression Measure')
plt.xlabel('Disease Progression')
plt.ylabel('Frequency')
plt.show()

# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Diabetes Features')
plt.tight_layout()
plt.show()

# Individual feature relationships with progression
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

for i, feature in enumerate(df.columns):
    if feature == 'disease_progression':
        continue
    
    ax = axes[i]
    sns.scatterplot(x=df[feature], y=df['disease_progression'], alpha=0.6, ax=ax)
    ax.set_title(f'{feature} vs Disease Progression')
    ax.set_xlabel(feature)
    ax.set_ylabel('Disease Progression')
    
    # Add regression line
    sns.regplot(x=df[feature], y=df['disease_progression'], 
                scatter=False, ax=ax, color='red', line_kws={'linewidth': 2})
    
    # Add correlation coefficient
    corr = df[feature].corr(df['disease_progression'])
    ax.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, backgroundcolor='white')

plt.tight_layout()
plt.show()
```

## 4. Data Preprocessing for Medical ML

```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Create train/test split
X = df.drop('disease_progression', axis=1)
y = df['disease_progression']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} patients")
print(f"Testing set: {X_test.shape[0]} patients")

# Create preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle potential missing values
    ('scaler', StandardScaler())                    # Standardize features
])

# Apply preprocessing pipeline
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

print("\nProcessed data shape:")
print(f"X_train_processed: {X_train_processed.shape}")
print(f"X_test_processed: {X_test_processed.shape}")
```

## 5. Model Development for Disease Prediction

We'll evaluate several regression models to predict diabetes progression.

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Define evaluation metric display function
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    return {'model': model_name, 'mae': mae, 'rmse': rmse, 'r2': r2}

# Create a dictionary of models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf')
}

# Train and evaluate each model
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_processed, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Evaluate the model
    result = evaluate_model(y_test, y_pred, name)
    results.append(result)

# Convert results to DataFrame for easier comparison
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.sort_values('r2', ascending=False).reset_index(drop=True))
```

### Model Comparison Visualization

```python
# Visualize model performance comparison
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

# R-squared comparison
results_df_sorted = results_df.sort_values('r2', ascending=False)
sns.barplot(x='model', y='r2', data=results_df_sorted, ax=ax[0])
ax[0].set_title('Model Comparison: R-squared (higher is better)')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
ax[0].set_ylim(0, 1)
ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# MAE comparison
results_df_sorted = results_df.sort_values('mae')
sns.barplot(x='model', y='mae', data=results_df_sorted, ax=ax[1])
ax[1].set_title('Model Comparison: Mean Absolute Error (lower is better)')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

### Selecting the Best Model

```python
# Identify best performing model based on R²
best_model_name = results_df.loc[results_df['r2'].idxmax()]['model']
print(f"\nBest performing model: {best_model_name}")

# Get the best model
best_model = models[best_model_name]

# Analyze predictions vs actual values
y_pred_best = best_model.predict(X_test_processed)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Actual vs Predicted Disease Progression ({best_model_name})')
plt.xlabel('Actual Disease Progression')
plt.ylabel('Predicted Disease Progression')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot residuals
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## 6. Feature Importance for Medical Interpretation

Understanding which features drive predictions is crucial in medical applications.

```python
import shap
from sklearn.inspection import permutation_importance

# Feature importance for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Direct feature importance from tree-based model
    importances = best_model.feature_importances_
    
    # Create dataframe for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance from {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    # Print feature importance with medical context
    print("\nFeature Importance with Clinical Context:")
    for index, row in feature_importance_df.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        description = feature_descriptions.get(feature, "N/A")
        print(f"{feature:10} {importance:.4f}  - {description}")

# For non-tree-based models, use permutation importance
else:
    # Calculate permutation importance
    perm_importance = permutation_importance(best_model, X_test_processed, y_test, 
                                            n_repeats=10, random_state=42)
    
    # Create dataframe for visualization
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    # Plot permutation importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=perm_importance_df)
    plt.title(f'Permutation Feature Importance from {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    # Print feature importance with medical context
    print("\nFeature Importance with Clinical Context:")
    for index, row in perm_importance_df.iterrows():
        feature = row['Feature']
        importance = row['Importance']
        description = feature_descriptions.get(feature, "N/A")
        print(f"{feature:10} {importance:.4f}  - {description}")
```

### SHAP Values for Explainable AI in Healthcare

```python
# Calculate SHAP values for explainability (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Create explainer
    explainer = shap.TreeExplainer(best_model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_processed)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names)
    plt.title(f'SHAP Value Summary for {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    # Dependence plots for top features
    top_features = feature_importance_df['Feature'][:3].values
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, feature in enumerate(top_features):
        feature_idx = list(feature_names).index(feature)
        shap.dependence_plot(feature_idx, shap_values, X_test_processed, 
                            feature_names=feature_names, ax=axes[i])
        axes[i].set_title(f'SHAP Dependence for {feature}')
    
    plt.tight_layout()
    plt.show()
```

## 7. Model Optimization for Clinical Use

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Assuming Gradient Boosting was the best model
# Define hyperparameter search space
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Create RandomizedSearchCV
random_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    scoring='r2',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Perform hyperparameter tuning
print("Performing hyperparameter tuning...")
random_search.fit(X_train_processed, y_train)

# Print best parameters
print("\nBest hyperparameters:")
print(random_search.best_params_)

# Evaluate optimized model
optimized_model = random_search.best_estimator_
y_pred_optimized = optimized_model.predict(X_test_processed)
optimized_results = evaluate_model(y_test, y_pred_optimized, 'Optimized Model')

# Compare with previous best model
print("\nImprovement in R²:", 
      optimized_results['r2'] - results_df.loc[results_df['r2'].idxmax()]['r2'])

# Save the optimized model
joblib.dump(optimized_model, 'diabetes_progression_model.pkl')
print("\nOptimized model saved to 'diabetes_progression_model.pkl'")

# Save the preprocessing pipeline
joblib.dump(preprocessing_pipeline, 'diabetes_preprocessing_pipeline.pkl')
print("Preprocessing pipeline saved to 'diabetes_preprocessing_pipeline.pkl'")
```

## 8. Clinical Decision Support System Development

Let's create a simple function to provide clinical insights based on the model's predictions.

```python
def predict_progression(patient_data, model_path='diabetes_progression_model.pkl', 
                      pipeline_path='diabetes_preprocessing_pipeline.pkl'):
    """
    Predict diabetes progression and provide clinical insights
    
    Parameters:
    -----------
    patient_data : dict or pandas DataFrame
        Patient's medical measurements
    model_path : str
        Path to saved model file
    pipeline_path : str
        Path to saved preprocessing pipeline
        
    Returns:
    --------
    dict
        Prediction results and clinical insights
    """
    # Load model and pipeline
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    
    # Convert input to DataFrame if dict
    if isinstance(patient_data, dict):
        patient_data = pd.DataFrame([patient_data])
    
    # Preprocess data
    X_processed = pipeline.transform(patient_data)
    
    # Make prediction
    prediction = model.predict(X_processed)[0]
    
    # Define risk levels based on prediction
    if prediction < 50:
        risk_level = "Low"
        recommendation = "Continue regular monitoring and maintain healthy lifestyle."
    elif prediction < 150:
        risk_level = "Moderate"
        recommendation = "Schedule follow-up in 3 months. Consider lifestyle interventions."
    else:
        risk_level = "High"
        recommendation = "Immediate clinical intervention required. Consider medication adjustment."
    
    # Calculate feature contributions if using tree-based model
    feature_contributions = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for i, feature in enumerate(feature_names):
            feature_contributions[feature] = importances[i]
    
    # Prepare response
    result = {
        'predicted_progression': prediction,
        'risk_level': risk_level,
        'clinical_recommendation': recommendation,
        'feature_contributions': feature_contributions
    }
    
    return result

# Test with sample patient data
sample_patient = {
    'age': 0.05, # Standardized values
    'sex': 0.05,
    'bmi': 0.06,
    'bp': 0.1,
    's1': 0.05,
    's2': -0.01,
    's3': -0.05,
    's4': 0.02,
    's5': 0.01,
    's6': -0.01
}

# Make prediction
prediction_result = predict_progression(sample_patient)

# Display clinical interpretation
print("\nClinical Decision Support Output:")
print(f"Predicted Disease Progression: {prediction_result['predicted_progression']:.1f}")
print(f"Risk Level: {prediction_result['risk_level']}")
print(f"Recommendation: {prediction_result['clinical_recommendation']}")
print("\nKey Contributing Factors:")
sorted_features = sorted(prediction_result['feature_contributions'].items(), 
                        key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features[:3]:
    description = feature_descriptions.get(feature, "")
    print(f"- {feature}: {description}")
```

## 9. Web Application for Medical Staff

Let's create a simple Flask application for healthcare providers to use our model.

```python
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    # Simple HTML form for input
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Diabetes Progression Prediction</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-group { margin-bottom: 15px; }
            label { display: inline-block; width: 180px; }
            input { padding: 5px; width: 200px; }
            button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <h1>Clinical Decision Support: Diabetes Progression Prediction</h1>
        <form id="predictionForm">
            <div class="form-group">
                <label>Age (standardized):</label>
                <input type="number" id="age" step="0.01" value="0.05">
            </div>
            <div class="form-group">
                <label>Sex (standardized):</label>
                <input type="number" id="sex" step="0.01" value="0.05">
            </div>
            <div class="form-group">
                <label>BMI (standardized):</label>
                <input type="number" id="bmi" step="0.01" value="0.06">
            </div>
            <div class="form-group">
                <label>Blood Pressure (standardized):</label>
                <input type="number" id="bp" step="0.01" value="0.1">
            </div>
            <div class="form-group">
                <label>Total Cholesterol (s1):</label>
                <input type="number" id="s1" step="0.01" value="0.05">
            </div>
            <div class="form-group">
                <label>LDL (s2):</label>
                <input type="number" id="s2" step="0.01" value="-0.01">
            </div>
            <div class="form-group">
                <label>HDL (s3):</label>
                <input type="number" id="s3" step="0.01" value="-0.05">
            </div>
            <div class="form-group">
                <label>Cholesterol/HDL Ratio (s4):</label>
                <input type="number" id="s4" step="0.01" value="0.02">
            </div>
            <div class="form-group">
                <label>Triglycerides (s5):</label>
                <input type="number" id="s5" step="0.01" value="0.01">
            </div>
            <div class="form-group">
                <label>Blood Sugar (s6):</label>
                <input type="number" id="s6" step="0.01" value="-0.01">
            </div>
            <button type="button" onclick="predict()">Predict Progression</button>
        </form>
        
        <div id="resultSection" class="result" style="display: none;">
            <h2>Clinical Assessment</h2>
            <p><strong>Predicted Disease Progression:</strong> <span id="progression"></span></p>
            <p><strong>Risk Level:</strong> <span id="riskLevel"></span></p>
            <p><strong>Recommendation:</strong> <span id="recommendation"></span></p>
            <h3>Key Contributing Factors</h3>
            <div id="factors"></div>
        </div>
        
        <script>
            function predict() {
                const data = {
                    age: parseFloat(document.getElementById('age').value),
                    sex: parseFloat(document.getElementById('sex').value),
                    bmi: parseFloat(document.getElementById('bmi').value),
                    bp: parseFloat(document.getElementById('bp').value),
                    s1: parseFloat(document.getElementById('s1').value),
                    s2: parseFloat(document.getElementById('s2').value),
                    s3: parseFloat(document.getElementById('s3').value),
                    s4: parseFloat(document.getElementById('s4').value),
                    s5: parseFloat(document.getElementById('s5').value),
                    s6: parseFloat(document.getElementById('s6').value)
                };
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(result => {
                    document.getElementById('progression').textContent = result.predicted_progression.toFixed(1);
                    document.getElementById('riskLevel').textContent = result.risk_level;
                    document.getElementById('recommendation').textContent = result.clinical_recommendation;
                    
                    // Display factors
                    const factorsDiv = document.getElementById('factors');
                    factorsDiv.innerHTML = '';
                    
                    const factors = Object.entries(result.feature_contributions)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 3);
                        
                    factors.forEach(([feature, importance]) => {
                        const p = document.createElement('p');
                        p.innerHTML = `<strong>${feature}:</strong> ${result.feature_descriptions[feature] || ''}`;
                        factorsDiv.appendChild(p);
                    });
                    
                    document.getElementById('resultSection').style.display = 'block';
                });
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    # Get patient data from request
    patient_data = request.json
    
    # Make prediction
    prediction_result = predict_progression(patient_data)
    
    # Add feature descriptions for frontend
    prediction_result['feature_descriptions'] = feature_descriptions
    
    return jsonify(prediction_result)

# To run the app:
# if __name__ == '__main__':
#     app.run(debug=True)
```

## 10. Ethical Considerations in Medical AI

When deploying ML models in healthcare, ethical considerations are paramount:

```python
def ethical_assessment(model, data, target_column='disease_progression'):
    """
    Performs an ethical assessment of a medical prediction model
    
    Parameters:
    -----------
    model : trained ML model
        The model to evaluate
    data : pandas DataFrame
        The dataset to assess for bias
    target_column : str
        Name of the target variable column
        
    Returns:
    --------
    dict
        Assessment results
    """
    # Split data for analysis
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # 1. Model Fairness - Check performance across demographics
    # In a real scenario, you would have demographic features (race, gender, etc.)
    # Here we'll use 'sex' as a proxy
    fairness_results = {}
    
    # Get predictions
    predictions = model.predict(preprocessing_pipeline.transform(X))
    
    # Create DataFrame with actual and predicted values
    df_results = pd.DataFrame({
        'actual': y,
        'predicted': predictions,
        'sex': X['sex']  # Using standardized values
    })
    
    # Create binary sex groups for demonstration
    df_results['sex_group'] = df_results['sex'].apply(lambda x: 'male' if x > 0 else 'female')
    
    # Calculate MAE per group
    group_performance = df_results.groupby('sex_group').apply(
        lambda x: mean_absolute_error(x['actual'], x['predicted'])
    )
    
    fairness_results['group_mae'] = group_performance.to_dict()
    fairness_results['max_disparity'] = group_performance.max() - group_performance.min()
    
    # 2. Assess Potential for Harm
    # In a real clinical setting, you would calculate the rate of false positives/negatives
    # and their potential clinical impact
    
    # For regression, we'll calculate critical errors
    # Define critical error as prediction being off by >100 units
    df_results['error'] = abs(df_results['actual'] - df_results['predicted'])
    df_results['critical_error'] = df_results['error'] > 100
    
    harm_assessment = {
        'critical_error_rate': df_results['critical_error'].mean(),
        'max_error': df_results['error'].max()
    }
    
    # 3. Privacy Assessment (simplified)
    privacy_assessment = {
        'contains_personal_identifiers': False,  # Assumption for this dataset
        'anonymization_level': 'High',           # Standardized values, no identifiers
        'reidentification_risk': 'Low'           # Based on dataset features
    }
    
    return {
        'fairness_assessment': fairness_results,
        'harm_assessment': harm_assessment,
        'privacy_assessment': privacy_assessment,
        'recommendations': [
            "Regular model auditing for performance disparities across demographic groups",
            "Implement confidence intervals with predictions to communicate uncertainty",
            "Use model as decision support, not as a replacement for clinical judgment",
            "Monitor for concept drift as clinical practices and populations change",
            "Ensure patient consent for data usage and model applications"
        ]
    }

# Perform ethical assessment
ethical_results = ethical_assessment(optimized_model, df)

# Display results
print("\nEthical Assessment Results:")
print("\nFairness Assessment:")
print(f"  MAE by demographic group: {ethical_results['fairness_assessment']['group_mae']}")
print(f"  Maximum performance disparity: {ethical_results['fairness_assessment']['max_disparity']:.2f}")

print("\nHarm Assessment:")
print(f"  Critical error rate: {ethical_results['harm_assessment']['critical_error_rate']:.2%}")
print(f"  Maximum prediction error: {ethical_results['harm_assessment']['max_error']:.2f}")

print("\nPrivacy Assessment:")
for key, value in ethical_results['privacy_assessment'].items():
    print(f"  {key}: {value}")

print("\nRecommendations for Ethical Deployment:")
for i, rec in enumerate(ethical_results['recommendations'], 1):
    print(f"  {i}. {rec}")
```

## 11. Model Monitoring for Clinical Safety

In a medical setting, continuous monitoring of model performance is critical for patient safety.

```python
def simulate_model_drift(model, X, y, n_periods=6):
    """
    Simulates data drift over time and monitors model performance
    
    Parameters:
    -----------
    model : trained ML model
        The model to monitor
    X : array-like
        Features used for prediction
    y : array-like
        True target values
    n_periods : int
        Number of time periods to simulate
        
    Returns:
    --------
    pd.DataFrame
        Monitoring results over time
    """
    import datetime
    
    # Initialize results storage
    results = []
    
    # Start date for simulation
    start_date = datetime.datetime.now()
    
    # Create a copy of the data that we'll gradually modify
    X_current = X.copy()
    y_current = y.copy()
    
    # Process X data for model input
    X_processed = preprocessing_pipeline.transform(X_current)
    
    # Calculate baseline performance
    y_pred = model.predict(X_processed)
    baseline_mae = mean_absolute_error(y_current, y_pred)
    baseline_r2 = r2_score(y_current, y_pred)
    
    # Add baseline to results
    results.append({
        'period': 0,
        'date': start_date,
        'mae': baseline_mae,
        'r2': baseline_r2,
        'drift_detected': False,
        'alert_level': 'Normal'
    })
    
    # Simulate drift over time periods
    for period in range(1, n_periods + 1):
        # Calculate current date
        current_date = start_date + datetime.timedelta(days=30 * period)
        
        # Simulate data drift by modifying some feature distributions
        # Focus on BMI and blood pressure which might change seasonally
        drift_factor = 1 + (period / 10)  # Increases over time
        
        # Apply drift to specific columns
        X_current['bmi'] = X_current['bmi'] * drift_factor
        X_current['bp'] = X_current['bp'] * drift_factor
        
        # Process drifted data
        X_processed_current = preprocessing_pipeline.transform(X_current)
        
        # Make predictions with drifted data
        y_pred_current = model.predict(X_processed_current)
        
        # Calculate performance metrics
        current_mae = mean_absolute_error(y_current, y_pred_current)
        current_r2 = r2_score(y_current, y_pred_current)
        
        # Determine if drift is significant
        mae_change = (current_mae - baseline_mae) / baseline_mae
        r2_change = (baseline_r2 - current_r2) / baseline_r2 if baseline_r2 > 0 else 0
        
        # Set alert levels
        if mae_change > 0.2 or r2_change > 0.2:
            alert_level = 'Critical'
            drift_detected = True
        elif mae_change > 0.1 or r2_change > 0.1:
            alert_level = 'Warning'
            drift_detected = True
        else:
            alert_level = 'Normal'
            drift_detected = False
        
        # Store results
        results.append({
            'period': period,
            'date': current_date,
            'mae': current_mae,
            'r2': current_r2,
            'mae_change': mae_change,
            'r2_change': r2_change,
            'drift_detected': drift_detected,
            'alert_level': alert_level
        })
    
    return pd.DataFrame(results)

# Run model monitoring simulation
monitoring_results = simulate_model_drift(optimized_model, X, y, n_periods=12)

# Display monitoring results
print("\nModel Monitoring Results:")
print(monitoring_results[['period', 'date', 'mae', 'r2', 'alert_level']])

# Visualize performance drift
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(monitoring_results['period'], monitoring_results['mae'], marker='o')
plt.axhline(y=monitoring_results.iloc[0]['mae'], color='r', linestyle='--')
plt.fill_between(monitoring_results['period'], 
                monitoring_results.iloc[0]['mae'], 
                monitoring_results.iloc[0]['mae'] * 1.1, 
                alpha=0.2, color='yellow', label='Warning Threshold')
plt.fill_between(monitoring_results['period'], 
                monitoring_results.iloc[0]['mae'] * 1.1, 
                monitoring_results.iloc[0]['mae'] * 1.2, 
                alpha=0.2, color='red', label='Critical Threshold')
plt.title('Model MAE Over Time')
plt.xlabel('Monitoring Period')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(monitoring_results['period'], monitoring_results['r2'], marker='o')
plt.axhline(y=monitoring_results.iloc[0]['r2'], color='r', linestyle='--')
plt.title('Model R² Over Time')
plt.xlabel('Monitoring Period')
plt.ylabel('R-squared')
plt.grid(True, alpha=0.3)

# Plot alert levels
plt.subplot(2, 2, 3)
colors = {'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'}
alert_colors = [colors[level] for level in monitoring_results['alert_level']]
plt.bar(monitoring_results['period'], [1] * len(monitoring_results), color=alert_colors)
plt.title('Model Health Status')
plt.xlabel('Monitoring Period')
plt.yticks([])
plt.tight_layout()

# Add a custom legend for alert levels
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Normal'),
    Patch(facecolor='orange', label='Warning'),
    Patch(facecolor='red', label='Critical')
]
plt.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()

# Determine when retraining is needed
retraining_needed = monitoring_results[monitoring_results['drift_detected'] == True]
if not retraining_needed.empty:
    first_drift = retraining_needed.iloc[0]
    print(f"\nModel retraining recommended after period {first_drift['period']}")
    print(f"Date: {first_drift['date'].strftime('%Y-%m-%d')}")
    print(f"Alert level: {first_drift['alert_level']}")
    print(f"Performance degradation: MAE increased by {first_drift['mae_change']:.1%}")
else:
    print("\nNo significant drift detected. Model remains reliable.")
```

## 12. Clinical Impact Assessment

Finally, let's assess the potential clinical impact of our model:

```python
def clinical_impact_analysis(baseline_error, model_error, patient_count):
    """
    Analyze the clinical impact of improved predictions
    
    Parameters:
    -----------
    baseline_error : float
        Error rate of current clinical practice
    model_error : float
        Error rate of ML model
    patient_count : int
        Estimated number of patients per year
        
    Returns:
    --------
    dict
        Impact metrics
    """
    # Assumptions about clinical outcomes and costs
    cost_per_overtreatment = 1500  # $ per patient
    cost_per_undertreatment = 5000  # $ per patient (includes complications)
    
    # Estimate error distribution
    # Assume 50% overtreatment, 50% undertreatment in errors
    error_reduction = baseline_error - model_error
    overtreatment_reduction = error_reduction * 0.5 * patient_count
    undertreatment_reduction = error_reduction * 0.5 * patient_count
    
    # Calculate cost savings
    overtreatment_savings = overtreatment_reduction * cost_per_overtreatment
    undertreatment_savings = undertreatment_reduction * cost_per_undertreatment
    total_savings = overtreatment_savings + undertreatment_savings
    
    # Calculate other metrics
    improved_outcomes = undertreatment_reduction  # Patients with better outcomes
    efficiency_improvement = overtreatment_reduction / patient_count
    
    # Calculate ROI (assuming model development and maintenance costs)
    development_cost = 250000  # $
    annual_maintenance = 50000  # $ per year
    first_year_roi = (total_savings - development_cost - annual_maintenance) / (development_cost + annual_maintenance)
    subsequent_roi = (total_savings - annual_maintenance) / annual_maintenance
    
    return {
        'patient_count': patient_count,
        'error_reduction_rate': error_reduction,
        'overtreatment_cases_avoided': overtreatment_reduction,
        'undertreatment_cases_avoided': undertreatment_reduction,
        'overtreatment_savings': overtreatment_savings,
        'undertreatment_savings': undertreatment_savings,
        'total_annual_savings': total_savings,
        'improved_patient_outcomes': improved_outcomes,
        'efficiency_improvement': efficiency_improvement,
        'first_year_roi': first_year_roi,
        'subsequent_years_roi': subsequent_roi
    }

# Calculate clinical impact
# Assume baseline clinical error rate is 20% and our model's error rate is ~15%
baseline_error_rate = 0.20
model_error_rate = 0.15  # Based on our MAE relative to the outcome range
annual_patients = 10000  # Hypothetical diabetes patient count

impact = clinical_impact_analysis(baseline_error_rate, model_error_rate, annual_patients)

# Display impact results
print("\nClinical Impact Assessment:")
print(f"Annual patient population: {impact['patient_count']:,}")
print(f"Error reduction rate: {impact['error_reduction_rate']:.1%}")
print(f"Overtreatment cases avoided annually: {impact['overtreatment_cases_avoided']:.0f}")
print(f"Undertreatment cases avoided annually: {impact['undertreatment_cases_avoided']:.0f}")
print(f"Total cost savings: ${impact['total_annual_savings']:,.2f}")
print(f"First year ROI: {impact['first_year_roi']:.1%}")
print(f"Subsequent years ROI: {impact['subsequent_years_roi']:.1%}")
print(f"Patients with improved outcomes: {impact['improved_patient_outcomes']:.0f}")
print(f"Healthcare efficiency improvement: {impact['efficiency_improvement']:.1%}")

# Visualize impact
plt.figure(figsize=(15, 10))

# Cost savings breakdown
plt.subplot(2, 2, 1)
savings = [impact['overtreatment_savings'], impact['undertreatment_savings']]
labels = ['Overtreatment Savings', 'Undertreatment Savings']
plt.pie(savings, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
plt.title('Cost Savings Breakdown')

# ROI by year
plt.subplot(2, 2, 2)
years = range(1, 6)
roi_values = [impact['first_year_roi']] + [impact['subsequent_years_roi']] * 4
plt.bar(years, [r * 100 for r in roi_values])  # Convert to percentage
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Return on Investment by Year')
plt.xlabel('Year')
plt.ylabel('ROI (%)')
plt.grid(True, alpha=0.3)

# Cases avoided
plt.subplot(2, 2, 3)
cases = [impact['overtreatment_cases_avoided'], impact['undertreatment_cases_avoided']]
labels = ['Overtreatment Avoided', 'Undertreatment Avoided']
plt.bar(labels, cases)
plt.title('Cases Avoided Annually')
plt.ylabel('Number of Patients')
plt.grid(True, alpha=0.3)

# Cumulative savings
plt.subplot(2, 2, 4)
years = range(1, 6)
savings_y1 = impact['total_annual_savings'] - impact['development_cost']
savings_subsequent = impact['total_annual_savings']
yearly_savings = [savings_y1] + [savings_subsequent] * 4
cumulative_savings = [sum(yearly_savings[:i+1]) for i in range(len(yearly_savings))]
plt.bar(years, yearly_savings, alpha=0.7, label='Annual Net Savings')
plt.plot(years, cumulative_savings, 'ro-', label='Cumulative Savings')
plt.title('Projected Financial Impact')
plt.xlabel('Year')
plt.ylabel('Savings ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 13. Key Learnings and Best Practices

This case study demonstrated how to build a ML model for predicting diabetes disease progression through the entire machine learning lifecycle. Key takeaways include:

1. **Medical Context Matters**: Understanding the clinical significance of features and predictions is essential for creating useful healthcare models.

2. **Feature Importance Analysis**: In medical applications, model interpretability is crucial for clinical adoption and ethical considerations.

3. **Model Evaluation**: Healthcare models require rigorous evaluation using multiple metrics and validation approaches.

4. **Ethical Considerations**: Special attention must be paid to fairness, bias, and potential harms when deploying ML in healthcare.

5. **Model Monitoring**: Healthcare AI systems require continuous monitoring for data drift and performance degradation to ensure patient safety.

6. **Clinical Integration**: Effective decision support systems must present predictions alongside interpretable insights for healthcare providers.

7. **Impact Assessment**: Quantifying the potential clinical and financial impact helps justify implementation of ML in healthcare settings.

### Best Practices for Medical ML:

- Always involve healthcare professionals in all stages of the project
- Prioritize model interpretability over small performance gains
- Implement thorough validation and testing before deployment
- Design systems that augment, rather than replace, clinical judgment
- Continuously monitor model performance and data distributions
- Maintain clear documentation of model limitations and intended use
- Consider regulatory requirements (like FDA approval for medical AI)
- Build in safeguards for identifying edge cases and unusual predictions

This project provides a foundation for applying machine learning in healthcare contexts, demonstrating both the technical implementation and essential considerations for responsible deployment.