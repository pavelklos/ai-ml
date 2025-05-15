# Testovací data pro Scikit-learn Estimatory

Tato příručka poskytuje vhodná testovací data pro každý scikit-learn estimator zmíněný v dokumentu. Každá sekce obsahuje ukázkový kód pro generování dat, který vytváří vhodné datasety pro konkrétní typ algoritmu.

## Klasifikační Estimatory

### SGD Classifier
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Vytvoření velkého syntetického datasetu (10 000 vzorků) s 20 příznaky
# SGD funguje dobře s velkými datasety a standardizovanými příznaky
X, y = make_classification(
    n_samples=10000, 
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Kernel Approximation
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Vytvoření nelineárního klasifikačního datasetu s 5000 vzorky
# Vhodné pro kernelové metody s jasným nelineárním oddělením
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=3,
    class_sep=0.8,
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Linear SVC
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Vytvoření vysokodimenzionálního datasetu s lineárním oddělením
# Linear SVC funguje dobře s vysokodimenzionálními daty, která mají jasné lineární hranice
X, y = make_classification(
    n_samples=1000,
    n_features=100,  # Vysokodimenzionální data
    n_informative=30,
    n_redundant=10,
    class_sep=1.0,  # Jasné oddělení
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### KNeighbors Classifier
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Vytvoření datasetu s komplexní lokální strukturou
# KNN funguje dobře s datasety, kde záleží na lokální blízkosti
X, y = make_classification(
    n_samples=1000,
    n_features=5,  # Nízkodimenzionální data
    n_informative=4,
    n_redundant=1,
    n_clusters_per_class=4,  # Komplexní lokální struktura
    class_sep=0.8,
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Ensemble Classifiers
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Vytvoření komplexního datasetu s více informativními příznaky
# Ensemble metody vynikají s komplexními datasety, které mají více vzorů
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

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### SVC
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Vytvoření nelineárního datasetu s jasnými rozhodovacími hranicemi
# SVC s RBF kernelem funguje dobře s menšími datasety s nelineárními vzory
X, y = make_classification(
    n_samples=800,  # Menší dataset
    n_features=10,
    n_informative=7,
    n_redundant=3,
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Naive Bayes
```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Vytvoření datasetu s nezávislými příznaky
# Naive Bayes funguje dobře, když jsou příznaky relativně nezávislé
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=0,  # Žádné redundantní příznaky
    n_clusters_per_class=1,  # Jednoduchá struktura
    random_state=42
)

# Pro textovou klasifikaci byste mohli použít:
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# news = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'talk.religion.misc'])
# X = CountVectorizer().fit_transform(news.data)
# y = news.target

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Regresní Estimatory

### SGD Regressor
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Vytvoření velkého syntetického regresního datasetu
# SGD funguje dobře s velkými datasety
X, y = make_regression(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    noise=5.0,
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### ElasticNet
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Vytvoření datasetu s mnoha korelovanými příznaky
# ElasticNet funguje dobře s korelovanými příznaky
X, y = make_regression(
    n_samples=1000,
    n_features=50,
    n_informative=10,
    effective_rank=5,  # Zavádí korelaci
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Lasso
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Vytvoření vysokodimenzionálního řídkého datasetu
# Lasso funguje dobře, když je mnoho příznaků irelevantních
X, y = make_regression(
    n_samples=1000,
    n_features=100,  # Vysokodimenzionální
    n_informative=10,  # Pouze 10 příznaků je relevantních
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Ridge Regression
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Vytvoření datasetu s korelovanými příznaky
# Ridge funguje dobře s korelovanými příznaky
X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    effective_rank=5,  # Vytváří korelaci
    random_state=42
)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### SVR (kernel="linear")
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Vytvoření regresního datasetu s odlehlými hodnotami
# Linear SVR je robustní vůči odlehlým hodnotám
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    random_state=42
)

# Přidání odlehlých hodnot
outlier_indices = np.random.choice(len(y), size=50, replace=False)
y[outlier_indices] += np.random.normal(0, 50, size=50)

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Ensemble Regressors
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Vytvoření komplexního regresního datasetu
# Ensemble metody vynikají v komplexních vztazích
X, y = make_regression(
    n_samples=2000,
    n_features=30,
    n_informative=15,
    random_state=42
)

# Přidání nelinearity
y = y + 0.5 * np.sin(X[:, 0]) + 0.5 * np.square(X[:, 1])

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### SVR (kernel="rbf")
```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Vytvoření nelineárního regresního datasetu
# RBF SVR funguje dobře s nelineárními vzory
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    random_state=42
)

# Přidání nelinearity
y = y + 0.7 * np.sin(X[:, 0] * 2) + 0.3 * np.cos(X[:, 1] * 3) + 0.2 * np.square(X[:, 2])

# Rozdělení dat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Shlukovací Estimatory

### KMeans
```python
import numpy as np
from sklearn.datasets import make_blobs

# Vytvoření dat s dobře oddělenými, sférickými shluky
# KMeans funguje nejlépe s kulovitými shluky
X, y_true = make_blobs(
    n_samples=1000,
    centers=3,
    cluster_std=0.7,
    random_state=42
)

# Pro KMeans často nepotřebujeme testovací sadu, protože jde o neřízené učení
# Ale můžete rozdělit data, pokud chcete vyhodnocovat na testovací sadě
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### Spectral Clustering
```python
import numpy as np
from sklearn.datasets import make_moons

# Vytvoření dat s nekompaktními, půlměsícovitými shluky
# Spectral clustering funguje dobře s komplexními tvary
X, y_true = make_moons(
    n_samples=500,
    noise=0.05,
    random_state=42
)

# Pro spectral clustering typicky nepotřebujeme testovací sadu
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### GMM (Gaussian Mixture Models)
```python
import numpy as np
from sklearn.datasets import make_blobs

# Vytvoření překrývajících se shluků různých velikostí a tvarů
# GMM funguje dobře se shluky s normálním rozdělením
X, y_true = make_blobs(
    n_samples=[100, 200, 300],  # Různé velikosti shluků
    centers=3,
    cluster_std=[0.5, 1.0, 1.5],  # Různé směrodatné odchylky
    random_state=42
)

# Přidání náhodné rotace pro vytvoření různých tvarů
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

# Pro GMM typicky nepotřebujeme testovací sadu
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### MiniBatch KMeans
```python
import numpy as np
from sklearn.datasets import make_blobs

# Vytvoření velkého datasetu s dobře oddělenými shluky
# MiniBatch KMeans je navržen pro velké datasety
X, y_true = make_blobs(
    n_samples=100000,  # Velký dataset
    centers=3,
    cluster_std=0.7,
    random_state=42
)

# Pro MiniBatch KMeans často nepotřebujeme testovací sadu
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### MeanShift
```python
import numpy as np
from sklearn.datasets import make_blobs

# Vytvoření dat se shluky různých hustot
# MeanShift automaticky nalézá centra shluků na základě hustoty
X, y_true = make_blobs(
    n_samples=1000,
    centers=[[0, 0], [5, 5], [-5, -5]],
    cluster_std=[0.5, 1.0, 1.5],  # Různé hustoty
    random_state=42
)

# Pro MeanShift typicky nepotřebujeme testovací sadu
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

### VBGMM (Variational Bayesian Gaussian Mixture)
```python
import numpy as np
from sklearn.datasets import make_blobs

# Vytvoření dat s neznámým počtem shluků
# VBGMM může automaticky určit počet komponent
X, y_true = make_blobs(
    n_samples=1000,
    centers=5,  # Nastavení vyššího počtu, než očekáváme najít
    cluster_std=[0.5, 0.8, 1.0, 1.2, 1.5],
    random_state=42
)

# Pro VBGMM typicky nepotřebujeme testovací sadu
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

## Estimatory pro redukci dimenzionality

### Randomized PCA
```python
import numpy as np
from sklearn.datasets import make_classification

# Vytvoření vysokodimenzionálního datasetu
# Randomized PCA je efektivní pro vysokodimenzionální data
X, y = make_classification(
    n_samples=2000,
    n_features=100,  # Vysokodimenzionální
    n_informative=10,
    n_redundant=90,
    random_state=42
)

# Pro redukci dimenzionality typicky provádíme fit na celém datasetu
# Není potřeba train/test rozdělení, pokud nepoužíváte redukované příznaky pro řízený úkol
```

### Spectral Embedding
```python
import numpy as np
from sklearn.datasets import make_swiss_roll

# Vytvoření datasetu typu Swiss roll
# Spectral embedding funguje dobře s manifoldovými daty
X, color = make_swiss_roll(
    n_samples=1000,
    noise=0.05,
    random_state=42
)

# Pro redukci dimenzionality typicky provádíme fit na celém datasetu
# Není potřeba train/test rozdělení, pokud nepoužíváte redukované příznaky pro řízený úkol
```

### IsoMap
```python
import numpy as np
from sklearn.datasets import make_swiss_roll

# Vytvoření datasetu typu Swiss roll
# IsoMap je navržen pro manifoldová data jako Swiss roll
X, color = make_swiss_roll(
    n_samples=1000,
    noise=0.05,
    random_state=42
)

# Pro redukci dimenzionality typicky provádíme fit na celém datasetu
# Není potřeba train/test rozdělení, pokud nepoužíváte redukované příznaky pro řízený úkol
```

### LLE (Locally Linear Embedding)
```python
import numpy as np
from sklearn.datasets import make_swiss_roll

# Vytvoření datasetu, který leží na nelineárním manifoldu
# LLE funguje dobře s manifoldovými daty, kde je důležitá lokální struktura
X, color = make_swiss_roll(
    n_samples=1000,
    noise=0.05,
    random_state=42
)

# Pro redukci dimenzionality typicky provádíme fit na celém datasetu
# Není potřeba train/test rozdělení, pokud nepoužíváte redukované příznaky pro řízený úkol
```

### Kernel Approximation
```python
import numpy as np
from sklearn.datasets import make_classification

# Vytvoření velkého datasetu s nelineárními vzory
# Kernel approximation je užitečná pro velké datasety s nelineárními vztahy
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    n_clusters_per_class=3,
    random_state=42
)

# Rozdělení dat, pokud používáte transformované příznaky pro řízený úkol
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```