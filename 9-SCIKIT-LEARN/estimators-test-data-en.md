# Scikit-learn Estimator Test Data

This guide provides appropriate test data for each scikit-learn estimator mentioned in the document. Each section includes sample data generation code that creates suitable datasets for the specific algorithm type.

## Classification Estimators

### SGD Classifier
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a large synthetic dataset (10,000 samples) with 20 features
# SGD works well with large datasets and standardized features
X, y = make_classification(
    n_samples=10000, 
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Kernel Approximation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a non-linear classification dataset with 5000 samples
# Suitable for kernel methods with clear non-linear separation
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=3,
    class_sep=0.8,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Linear SVC
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a high-dimensional dataset with linear separation
# Linear SVC works well with high-dimensional data that has clear linear boundaries
X, y = make_classification(
    n_samples=1000,
    n_features=100,  # High-dimensional data
    n_informative=30,
    n_redundant=10,
    class_sep=1.0,  # Clear separation
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### KNeighbors Classifier
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a dataset with complex local structure
# KNN works well with datasets where local proximity matters
X, y = make_classification(
    n_samples=1000,
    n_features=5,  # Lower-dimensional data
    n_informative=4,
    n_redundant=1,
    n_clusters_per_class=4,  # Complex local structure
    class_sep=0.8,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Ensemble Classifiers
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a complex dataset with multiple informative features
# Ensemble methods excel with complex datasets that have multiple patterns
X, y = make_classification(
    n_samples=2000,
    n_features=30,
    n_informative=20,
    n_redundant=5,
    n_classes=2,
    n_clusters_per_class=3,
    class_sep=0.75,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### SVC
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a non-linear dataset with clear decision boundaries
# SVC with RBF kernel works well with smaller datasets having non-linear patterns
X, y = make_classification(
    n_samples=800,  # Smaller dataset
    n_features=10,
    n_informative=7,
    n_redundant=3,
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Naive Bayes
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a dataset with independent features
# Naive Bayes works well when features are relatively independent
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=0,  # No redundant features
    n_clusters_per_class=1,  # Simple structure
    random_state=42
)

# For text classification, you could use:
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# news = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'talk.religion.misc'])
# X = CountVectorizer().fit_transform(news.data)
# y = news.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Regression Estimators

### SGD Regressor
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a large synthetic regression dataset
# SGD works well with large datasets
X, y = make_regression(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    noise=5.0,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ElasticNet
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a dataset with many correlated features
# ElasticNet works well with correlated features
X, y = make_regression(
    n_samples=1000,
    n_features=50,
    n_informative=10,
    effective_rank=5,  # Introduces correlation
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Lasso
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a high-dimensional sparse dataset
# Lasso works well when many features are irrelevant
X, y = make_regression(
    n_samples=1000,
    n_features=100,  # High-dimensional
    n_informative=10,  # Only 10 features are relevant
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Ridge Regression
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a dataset with correlated features
# Ridge works well with correlated features
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    effective_rank=5,  # Creates correlation
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### SVR (kernel="linear")
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a regression dataset with outliers
# Linear SVR is robust to outliers
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    random_state=42
)

# Add some outliers
outlier_indices = np.random.choice(len(y), size=50, replace=False)
y[outlier_indices] += np.random.normal(0, 50, size=50)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Ensemble Regressors
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a complex regression dataset
# Ensemble methods excel with complex relationships
X, y = make_regression(
    n_samples=2000,
    n_features=30,
    n_informative=15,
    random_state=42
)

# Add some non-linearity
y = y + 0.5 * np.sin(X[:, 0]) + 0.5 * np.square(X[:, 1])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### SVR (kernel="rbf")
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a non-linear regression dataset
# RBF SVR works well with non-linear patterns
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    random_state=42
)

# Add non-linearity
y = y + 0.7 * np.sin(X[:, 0] * 2) + 0.3 * np.cos(X[:, 1] * 3) + 0.2 * np.square(X[:, 2])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Clustering Estimators

### KMeans
```python
import numpy as np
from sklearn.datasets import make_blobs

# Create data with well-separated, spherical clusters
# KMeans works best with globular clusters
X, y_true = make_blobs(
    n_samples=1000,
    centers=3,
    cluster_std=0.7,
    random_state=42
)

# For KMeans, we often don't need a test set as it's unsupervised
# But you can split if you want to evaluate on a test set
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### Spectral Clustering
```python
import numpy as np
from sklearn.datasets import make_moons

# Create data with non-globular, moon-shaped clusters
# Spectral clustering works well with complex shapes
X, y_true = make_moons(
    n_samples=500,
    noise=0.05,
    random_state=42
)

# For spectral clustering, we typically don't need a test set
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### GMM (Gaussian Mixture Models)
```python
import numpy as np
from sklearn.datasets import make_blobs

# Create overlapping clusters of different sizes and shapes
# GMM works well with Gaussian-distributed clusters
X, y_true = make_blobs(
    n_samples=[100, 200, 300],  # Different cluster sizes
    centers=3,
    cluster_std=[0.5, 1.0, 1.5],  # Different standard deviations
    random_state=42
)

# Add some random rotation to create different shapes
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# For GMM, we typically don't need a test set
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### MiniBatch KMeans
```python
import numpy as np
from sklearn.datasets import make_blobs

# Create a large dataset with well-separated clusters
# MiniBatch KMeans is designed for large datasets
X, y_true = make_blobs(
    n_samples=100000,  # Large dataset
    centers=3,
    cluster_std=0.7,
    random_state=42
)

# For MiniBatch KMeans, we often don't need a test set
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### MeanShift
```python
import numpy as np
from sklearn.datasets import make_blobs

# Create data with clusters of different densities
# MeanShift automatically finds cluster centers based on density
X, y_true = make_blobs(
    n_samples=1000,
    centers=[[0, 0], [5, 5], [-5, -5]],
    cluster_std=[0.5, 1.0, 1.5],  # Different densities
    random_state=42
)

# For MeanShift, we typically don't need a test set
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### VBGMM (Variational Bayesian Gaussian Mixture)
```python
import numpy as np
from sklearn.datasets import make_blobs

# Create data with an unknown number of clusters
# VBGMM can automatically determine the number of components
X, y_true = make_blobs(
    n_samples=1000,
    centers=5,  # Set a higher number than we expect to find
    cluster_std=[0.5, 0.8, 1.0, 1.2, 1.5],
    random_state=42
)

# For VBGMM, we typically don't need a test set
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

## Dimensionality Reduction Estimators

### Randomized PCA
```python
import numpy as np
from sklearn.datasets import make_classification

# Create a high-dimensional dataset
# Randomized PCA is efficient for high-dimensional data
X, y = make_classification(
    n_samples=2000,
    n_features=100,  # High-dimensional
    n_informative=10,
    n_redundant=90,
    random_state=42
)

# For dimensionality reduction, we typically fit on the full dataset
# No need for train/test split, unless you're using the reduced features for a supervised task
```

### Spectral Embedding
```python
import numpy as np
from sklearn.datasets import make_swiss_roll

# Create a Swiss roll dataset
# Spectral embedding works well with manifold data
X, color = make_swiss_roll(
    n_samples=1000,
    noise=0.05,
    random_state=42
)

# For dimensionality reduction, we typically fit on the full dataset
# No need for train/test split, unless you're using the reduced features for a supervised task
```

### IsoMap
```python
import numpy as np
from sklearn.datasets import make_swiss_roll

# Create a Swiss roll dataset
# IsoMap is designed for manifold data like the Swiss roll
X, color = make_swiss_roll(
    n_samples=1000,
    noise=0.05,
    random_state=42
)

# For dimensionality reduction, we typically fit on the full dataset
# No need for train/test split, unless you're using the reduced features for a supervised task
```

### LLE (Locally Linear Embedding)
```python
import numpy as np
from sklearn.datasets import make_swiss_roll

# Create a dataset that lies on a non-linear manifold
# LLE works well with manifold data where local structure is important
X, color = make_swiss_roll(
    n_samples=1000,
    noise=0.05,
    random_state=42
)

# For dimensionality reduction, we typically fit on the full dataset
# No need for train/test split, unless you're using the reduced features for a supervised task
```

### Kernel Approximation
```python
import numpy as np
from sklearn.datasets import make_classification

# Create a large dataset with non-linear patterns
# Kernel approximation is useful for large datasets with non-linear relationships
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    n_clusters_per_class=3,
    random_state=42
)

# Split the data if you're using the transformed features for a supervised task
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```