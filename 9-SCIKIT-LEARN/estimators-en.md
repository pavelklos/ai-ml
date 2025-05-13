# [Scikit-learn] Estimators (algorithms)

- Classification
  - SGD Classifier
  - Kernel Approximation
  - Linear SVC
  - KNeighbors Classifier
  - Ensemble Classifiers
  - SVC
  - Naive Bayes

- Regression
  - SGD Regressor
  - ElasticNet
  - Lasso
  - Ridge Regression
  - SVR (kernel="linear")
  - Ensemble Regressors
  - SVR (kernel="rbf")

- Clustering
  - KMeans
  - Spectral Clustering
  - GMM
  - MiniBatch KMeans
  - MeanShift
  - VBGMM

- Dimensionality Reduction
  - Randomized PCA
  - Spectral Embedding
  - IsoMap
  - LLE
  - Kernel Approximation

# Scikit-learn Estimators (Algorithms)

## Choosing the Right Algorithm Category

### Classification
**When to use:** Use classification algorithms when you need to predict discrete class labels or categories. Classification is suitable when your target variable is categorical (e.g., spam/not spam, fraud/legitimate, disease/no disease).

### Regression
**When to use:** Use regression algorithms when you need to predict continuous numerical values. Regression is appropriate when your target variable is a real number (e.g., house prices, temperature, sales figures).

### Clustering
**When to use:** Use clustering algorithms when you need to discover inherent groupings in your data without labeled examples. Clustering is an unsupervised learning approach that organizes similar data points into groups.

### Dimensionality Reduction
**When to use:** Use dimensionality reduction techniques when you need to reduce the number of features in your dataset while preserving meaningful information. These techniques help with visualization, addressing the curse of dimensionality, and speeding up model training.

## Classification Algorithms

### SGD Classifier
**When to use:** When dealing with large datasets or when you need online learning capabilities.
**Suitable for:** Linear classification problems with large datasets where memory efficiency is important.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and SGD classifier
clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(max_iter=1000, tol=1e-3)
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

### Kernel Approximation
**When to use:** When you want to use kernel methods on large datasets but can't afford the computational cost of traditional kernel SVMs.
**Suitable for:** Large datasets where you want kernel-like performance with linear scalability.

```python
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

# Create a pipeline with RBF kernel approximation and linear classifier
rbf_feature = RBFSampler(gamma=1, random_state=1)
clf = make_pipeline(
    rbf_feature,
    SGDClassifier(max_iter=1000)
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

### Linear SVC
**When to use:** For linear classification problems when you need better control over regularization and penalties.
**Suitable for:** High-dimensional datasets with clear linear separation.

```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Linear SVC
clf = make_pipeline(
    StandardScaler(),
    LinearSVC(dual=False, tol=1e-3)
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

### KNeighbors Classifier
**When to use:** When your data has complex local structure and you don't need an explicit model.
**Suitable for:** Lower-dimensional datasets where proximity is meaningful for classification.

```python
from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
clf = KNeighborsClassifier(n_neighbors=5)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

### Ensemble Classifiers
**When to use:** When you want to improve model performance by combining multiple models.
**Suitable for:** Complex problems where a single model might not capture all patterns.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_clf.predict(X_test)
gb_predictions = gb_clf.predict(X_test)
```

### SVC
**When to use:** When you need a powerful non-linear classifier with various kernel options.
**Suitable for:** Smaller datasets with complex decision boundaries.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and SVC
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1, gamma='scale')
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

### Naive Bayes
**When to use:** When you have independent features and need a fast, simple classifier.
**Suitable for:** Text classification, spam filtering, and situations with relatively independent features.

```python
from sklearn.naive_bayes import GaussianNB

# Create Naive Bayes classifier
clf = GaussianNB()

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

## Regression Algorithms

### SGD Regressor
**When to use:** For large datasets when you need online learning capabilities.
**Suitable for:** Linear regression problems with large datasets where memory efficiency is important.

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and SGD regressor
reg = make_pipeline(
    StandardScaler(),
    SGDRegressor(max_iter=1000, tol=1e-3)
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)
```

### ElasticNet
**When to use:** When you want a balance between Ridge and Lasso regression.
**Suitable for:** Datasets with many correlated features where you want both feature selection and regularization.

```python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and ElasticNet
reg = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)
```

### Lasso
**When to use:** When you need feature selection and sparse models.
**Suitable for:** High-dimensional datasets where many features may be irrelevant.

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Lasso
reg = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1)
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)
```

### Ridge Regression
**When to use:** When you want to penalize large coefficients but keep all features.
**Suitable for:** Datasets with many correlated features.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Ridge regression
reg = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)
```

### SVR (kernel="linear")
**When to use:** When you need a linear regression model that's robust to outliers.
**Suitable for:** Datasets where you want the benefits of SVM in a regression context with linear relationships.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and linear SVR
reg = make_pipeline(
    StandardScaler(),
    SVR(kernel='linear', C=1.0)
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)
```

### Ensemble Regressors
**When to use:** When you want to improve regression performance by combining multiple models.
**Suitable for:** Complex regression problems where a single model might not capture all patterns.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_reg.predict(X_test)
gb_predictions = gb_reg.predict(X_test)
```

### SVR (kernel="rbf")
**When to use:** When your data has non-linear relationships that require a flexible regression model.
**Suitable for:** Complex regression problems with non-linear patterns.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and RBF SVR
reg = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=1.0, gamma='scale')
)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
predictions = reg.predict(X_test)
```

## Clustering Algorithms

### KMeans
**When to use:** When you need a simple, fast clustering algorithm with spherical clusters.
**Suitable for:** Data with well-separated, roughly equal-sized, globular clusters.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and KMeans
clustering = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, random_state=42)
)

# Fit the clustering model
clustering.fit(X)

# Get cluster assignments
labels = clustering.predict(X)
```

### Spectral Clustering
**When to use:** When your data forms complex, non-globular shapes.
**Suitable for:** Data where clusters have complex shapes that KMeans would fail to identify.

```python
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Spectral Clustering
clustering = make_pipeline(
    StandardScaler(),
    SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
)

# Fit the clustering model
labels = clustering.fit_predict(X)
```

### GMM (Gaussian Mixture Models)
**When to use:** When your data consists of overlapping clusters of different sizes and shapes.
**Suitable for:** Data that can be modeled as a mixture of Gaussian distributions.

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and GMM
clustering = make_pipeline(
    StandardScaler(),
    GaussianMixture(n_components=3, random_state=42)
)

# Fit the model
clustering.fit(X)

# Get cluster assignments
labels = clustering[-1].predict(clustering[0].transform(X))
```

### MiniBatch KMeans
**When to use:** When you have large datasets and need a faster version of KMeans.
**Suitable for:** Very large datasets where standard KMeans would be too slow.

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and MiniBatch KMeans
clustering = make_pipeline(
    StandardScaler(),
    MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)
)

# Fit the model
clustering.fit(X)

# Get cluster assignments
labels = clustering.predict(X)
```

### MeanShift
**When to use:** When you don't know the number of clusters in advance and want to discover them.
**Suitable for:** Data with an unknown number of clusters of varying shapes and sizes.

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Estimate bandwidth for MeanShift
bandwidth = estimate_bandwidth(X, quantile=0.2)

# Create a pipeline with standardization and MeanShift
clustering = make_pipeline(
    StandardScaler(),
    MeanShift(bandwidth=bandwidth, bin_seeding=True)
)

# Fit the model
clustering.fit(X)

# Get cluster assignments
labels = clustering.predict(X)
```

### VBGMM (Variational Bayesian Gaussian Mixture)
**When to use:** When you want to automatically determine the number of clusters.
**Suitable for:** Data where the number of components is not known in advance.

```python
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Bayesian GMM
clustering = make_pipeline(
    StandardScaler(),
    BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.1, random_state=42)
)

# Fit the model
clustering.fit(X)

# Get cluster assignments
labels = clustering[-1].predict(clustering[0].transform(X))
```

## Dimensionality Reduction Algorithms

### Randomized PCA
**When to use:** When you need to reduce dimensions efficiently on large datasets.
**Suitable for:** High-dimensional datasets where computational efficiency is important.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Randomized PCA
dim_reduction = make_pipeline(
    StandardScaler(),
    PCA(n_components=2, svd_solver='randomized', random_state=42)
)

# Fit and transform the data
X_reduced = dim_reduction.fit_transform(X)
```

### Spectral Embedding
**When to use:** When you need a non-linear dimensionality reduction that preserves local relationships.
**Suitable for:** Data where local structure is important and linear methods fail.

```python
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Spectral Embedding
dim_reduction = make_pipeline(
    StandardScaler(),
    SpectralEmbedding(n_components=2, random_state=42)
)

# Fit and transform the data
X_reduced = dim_reduction.fit_transform(X)
```

### IsoMap
**When to use:** When you want to preserve geodesic distances between points.
**Suitable for:** Data that lies on a non-linear manifold, like a "Swiss roll".

```python
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Isomap
dim_reduction = make_pipeline(
    StandardScaler(),
    Isomap(n_components=2, n_neighbors=5)
)

# Fit and transform the data
X_reduced = dim_reduction.fit_transform(X)
```

### LLE (Locally Linear Embedding)
**When to use:** When you want to preserve local properties of the data.
**Suitable for:** Non-linear data where local structure is important.

```python
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and LLE
dim_reduction = make_pipeline(
    StandardScaler(),
    LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
)

# Fit and transform the data
X_reduced = dim_reduction.fit_transform(X)
```

### Kernel Approximation
**When to use:** When you want to use kernel methods for dimensionality reduction efficiently.
**Suitable for:** Large datasets where explicit kernel computations would be too expensive.

```python
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline with standardization and Nystroem kernel approximation
dim_reduction = make_pipeline(
    StandardScaler(),
    Nystroem(kernel='rbf', n_components=2, random_state=42)
)

# Fit and transform the data
X_reduced = dim_reduction.fit_transform(X)
```