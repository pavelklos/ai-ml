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

# Scikit-learn Estimátory (Algoritmy)

## Jak zvolit správnou kategorii algoritmu

### Klasifikace
**Kdy použít:** Klasifikační algoritmy použijte, když potřebujete predikovat diskrétní třídy nebo kategorie. Klasifikace je vhodná, když je vaše cílová proměnná kategorická (např. spam/ne spam, podvod/legitimní, nemoc/bez nemoci).

### Regrese
**Kdy použít:** Regresní algoritmy použijte, když potřebujete predikovat spojité numerické hodnoty. Regrese je vhodná, když je vaše cílová proměnná reálné číslo (např. ceny domů, teplota, prodejní čísla).

### Shlukování
**Kdy použít:** Shlukovací algoritmy použijte, když potřebujete objevit přirozené skupiny ve vašich datech bez označených příkladů. Shlukování je přístup strojového učení bez učitele, který organizuje podobná data do skupin.

### Redukce dimenzionality
**Kdy použít:** Techniky redukce dimenzionality použijte, když potřebujete snížit počet příznaků v datové sadě při zachování smysluplných informací. Tyto techniky pomáhají s vizualizací, řešením prokletí dimenzionality a zrychlením trénování modelů.

## Klasifikační algoritmy

### SGD Classifier
**Kdy použít:** Při práci s velkými datovými sadami nebo když potřebujete schopnosti online učení.
**Vhodné pro:** Lineární klasifikační problémy s velkými datovými sadami, kde je důležitá efektivita paměti.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a SGD klasifikátorem
clf = make_pipeline(
    StandardScaler(),
    SGDClassifier(max_iter=1000, tol=1e-3)
)

# Trénování modelu
clf.fit(X_train, y_train)

# Predikce
predictions = clf.predict(X_test)
```

### Kernel Approximation
**Kdy použít:** Když chcete použít jádrové metody na velké datové sady, ale nemůžete si dovolit výpočetní náklady tradičních jádrových SVM.
**Vhodné pro:** Velké datové sady, kde chcete výkon podobný jádrovým metodám s lineární škálovatelností.

```python
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline s aproximací RBF jádra a lineárním klasifikátorem
rbf_feature = RBFSampler(gamma=1, random_state=1)
clf = make_pipeline(
    rbf_feature,
    SGDClassifier(max_iter=1000)
)

# Trénování modelu
clf.fit(X_train, y_train)

# Predikce
predictions = clf.predict(X_test)
```

### Linear SVC
**Kdy použít:** Pro lineární klasifikační problémy, když potřebujete lepší kontrolu nad regularizací a penalizacemi.
**Vhodné pro:** Vysoko-dimenzionální datové sady s jasným lineárním oddělením.

```python
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Linear SVC
clf = make_pipeline(
    StandardScaler(),
    LinearSVC(dual=False, tol=1e-3)
)

# Trénování modelu
clf.fit(X_train, y_train)

# Predikce
predictions = clf.predict(X_test)
```

### KNeighbors Classifier
**Kdy použít:** Když vaše data mají komplexní lokální strukturu a nepotřebujete explicitní model.
**Vhodné pro:** Nižší dimenzionální datové sady, kde je blízkost významná pro klasifikaci.

```python
from sklearn.neighbors import KNeighborsClassifier

# Vytvoření KNN klasifikátoru
clf = KNeighborsClassifier(n_neighbors=5)

# Trénování modelu
clf.fit(X_train, y_train)

# Predikce
predictions = clf.predict(X_test)
```

### Ensemble Classifiers
**Kdy použít:** Když chcete zlepšit výkon modelu kombinací více modelů.
**Vhodné pro:** Komplexní problémy, kde jeden model nemusí zachytit všechny vzory.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)

# Predikce
rf_predictions = rf_clf.predict(X_test)
gb_predictions = gb_clf.predict(X_test)
```

### SVC
**Kdy použít:** Když potřebujete výkonný nelineární klasifikátor s různými možnostmi jader.
**Vhodné pro:** Menší datové sady s komplexními hranicemi rozhodování.

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a SVC
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1, gamma='scale')
)

# Trénování modelu
clf.fit(X_train, y_train)

# Predikce
predictions = clf.predict(X_test)
```

### Naive Bayes
**Kdy použít:** Když máte nezávislé příznaky a potřebujete rychlý, jednoduchý klasifikátor.
**Vhodné pro:** Klasifikaci textů, filtrování spamu a situace s relativně nezávislými příznaky.

```python
from sklearn.naive_bayes import GaussianNB

# Vytvoření Naive Bayes klasifikátoru
clf = GaussianNB()

# Trénování modelu
clf.fit(X_train, y_train)

# Predikce
predictions = clf.predict(X_test)
```

## Regresní algoritmy

### SGD Regressor
**Kdy použít:** Pro velké datové sady, když potřebujete schopnosti online učení.
**Vhodné pro:** Lineární regresní problémy s velkými datovými sadami, kde je důležitá efektivita paměti.

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a SGD regresorem
reg = make_pipeline(
    StandardScaler(),
    SGDRegressor(max_iter=1000, tol=1e-3)
)

# Trénování modelu
reg.fit(X_train, y_train)

# Predikce
predictions = reg.predict(X_test)
```

### ElasticNet
**Kdy použít:** Když chcete rovnováhu mezi Ridge a Lasso regresí.
**Vhodné pro:** Datové sady s mnoha korelovanými příznaky, kde chcete jak výběr příznaků, tak regularizaci.

```python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a ElasticNet
reg = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
)

# Trénování modelu
reg.fit(X_train, y_train)

# Predikce
predictions = reg.predict(X_test)
```

### Lasso
**Kdy použít:** Když potřebujete výběr příznaků a řídké modely.
**Vhodné pro:** Vysoko-dimenzionální datové sady, kde mnoho příznaků může být irelevantních.

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Lasso
reg = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1)
)

# Trénování modelu
reg.fit(X_train, y_train)

# Predikce
predictions = reg.predict(X_test)
```

### Ridge Regression
**Kdy použít:** Když chcete penalizovat velké koeficienty, ale zachovat všechny příznaky.
**Vhodné pro:** Datové sady s mnoha korelovanými příznaky.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Ridge regresí
reg = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Trénování modelu
reg.fit(X_train, y_train)

# Predikce
predictions = reg.predict(X_test)
```

### SVR (kernel="linear")
**Kdy použít:** Když potřebujete lineární regresní model, který je odolný vůči odlehlým hodnotám.
**Vhodné pro:** Datové sady, kde chcete výhody SVM v regresním kontextu s lineárními vztahy.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a lineárním SVR
reg = make_pipeline(
    StandardScaler(),
    SVR(kernel='linear', C=1.0)
)

# Trénování modelu
reg.fit(X_train, y_train)

# Predikce
predictions = reg.predict(X_test)
```

### Ensemble Regressors
**Kdy použít:** Když chcete zlepšit regresní výkon kombinací více modelů.
**Vhodné pro:** Komplexní regresní problémy, kde jeden model nemusí zachytit všechny vzory.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train, y_train)

# Predikce
rf_predictions = rf_reg.predict(X_test)
gb_predictions = gb_reg.predict(X_test)
```

### SVR (kernel="rbf")
**Kdy použít:** Když vaše data obsahují nelineární vztahy, které vyžadují flexibilní regresní model.
**Vhodné pro:** Komplexní regresní problémy s nelineárními vzory.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a RBF SVR
reg = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=1.0, gamma='scale')
)

# Trénování modelu
reg.fit(X_train, y_train)

# Predikce
predictions = reg.predict(X_test)
```

## Shlukovací algoritmy

### KMeans
**Kdy použít:** Když potřebujete jednoduchý, rychlý shlukovací algoritmus se sférickými shluky.
**Vhodné pro:** Data s dobře oddělenými, přibližně stejně velkými, kulovitými shluky.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a KMeans
clustering = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, random_state=42)
)

# Aplikace shlukovacího modelu
clustering.fit(X)

# Získání přiřazení do shluků
labels = clustering.predict(X)
```

### Spectral Clustering
**Kdy použít:** Když vaše data tvoří komplexní, nekulovité tvary.
**Vhodné pro:** Data, kde shluky mají komplexní tvary, které by KMeans nedokázal identifikovat.

```python
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Spektrálním shlukováním
clustering = make_pipeline(
    StandardScaler(),
    SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
)

# Aplikace shlukovacího modelu
labels = clustering.fit_predict(X)
```

### GMM (Gaussian Mixture Models)
**Kdy použít:** Když vaše data obsahují překrývající se shluky různých velikostí a tvarů.
**Vhodné pro:** Data, která lze modelovat jako směs Gaussovských rozdělení.

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a GMM
clustering = make_pipeline(
    StandardScaler(),
    GaussianMixture(n_components=3, random_state=42)
)

# Aplikace modelu
clustering.fit(X)

# Získání přiřazení do shluků
labels = clustering[-1].predict(clustering[0].transform(X))
```

### MiniBatch KMeans
**Kdy použít:** Když máte velké datové sady a potřebujete rychlejší verzi KMeans.
**Vhodné pro:** Velmi velké datové sady, kde by standardní KMeans byl příliš pomalý.

```python
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a MiniBatch KMeans
clustering = make_pipeline(
    StandardScaler(),
    MiniBatchKMeans(n_clusters=3, batch_size=100, random_state=42)
)

# Aplikace modelu
clustering.fit(X)

# Získání přiřazení do shluků
labels = clustering.predict(X)
```

### MeanShift
**Kdy použít:** Když předem neznáte počet shluků a chcete je objevit.
**Vhodné pro:** Data s neznámým počtem shluků různých tvarů a velikostí.

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Odhad šířky pásma pro MeanShift
bandwidth = estimate_bandwidth(X, quantile=0.2)

# Vytvoření pipeline se standardizací a MeanShift
clustering = make_pipeline(
    StandardScaler(),
    MeanShift(bandwidth=bandwidth, bin_seeding=True)
)

# Aplikace modelu
clustering.fit(X)

# Získání přiřazení do shluků
labels = clustering.predict(X)
```

### VBGMM (Variational Bayesian Gaussian Mixture)
**Kdy použít:** Když chcete automaticky určit počet shluků.
**Vhodné pro:** Data, kde není předem znám počet komponent.

```python
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Bayesovským GMM
clustering = make_pipeline(
    StandardScaler(),
    BayesianGaussianMixture(n_components=10, weight_concentration_prior=0.1, random_state=42)
)

# Aplikace modelu
clustering.fit(X)

# Získání přiřazení do shluků
labels = clustering[-1].predict(clustering[0].transform(X))
```

## Algoritmy redukce dimenzionality

### Randomized PCA
**Kdy použít:** Když potřebujete efektivně snížit dimenzionalitu na velkých datových sadách.
**Vhodné pro:** Vysoko-dimenzionální datové sady, kde je důležitá výpočetní efektivita.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a randomizovanou PCA
dim_reduction = make_pipeline(
    StandardScaler(),
    PCA(n_components=2, svd_solver='randomized', random_state=42)
)

# Aplikace a transformace dat
X_reduced = dim_reduction.fit_transform(X)
```

### Spectral Embedding
**Kdy použít:** Když potřebujete nelineární redukci dimenzionality, která zachovává lokální vztahy.
**Vhodné pro:** Data, kde je důležitá lokální struktura a lineární metody selhávají.

```python
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Spektrálním vkládáním
dim_reduction = make_pipeline(
    StandardScaler(),
    SpectralEmbedding(n_components=2, random_state=42)
)

# Aplikace a transformace dat
X_reduced = dim_reduction.fit_transform(X)
```

### IsoMap
**Kdy použít:** Když chcete zachovat geodetické vzdálenosti mezi body.
**Vhodné pro:** Data, která leží na nelineárním rozložení, jako je "Swiss roll".

```python
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Isomap
dim_reduction = make_pipeline(
    StandardScaler(),
    Isomap(n_components=2, n_neighbors=5)
)

# Aplikace a transformace dat
X_reduced = dim_reduction.fit_transform(X)
```

### LLE (Locally Linear Embedding)
**Kdy použít:** Když chcete zachovat lokální vlastnosti dat.
**Vhodné pro:** Nelineární data, kde je důležitá lokální struktura.

```python
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a LLE
dim_reduction = make_pipeline(
    StandardScaler(),
    LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
)

# Aplikace a transformace dat
X_reduced = dim_reduction.fit_transform(X)
```

### Kernel Approximation
**Kdy použít:** Když chcete efektivně použít jádrové metody pro redukci dimenzionality.
**Vhodné pro:** Velké datové sady, kde by explicitní výpočty jádrových funkcí byly příliš nákladné.

```python
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Vytvoření pipeline se standardizací a Nystroemovou aproximací jádra
dim_reduction = make_pipeline(
    StandardScaler(),
    Nystroem(kernel='rbf', n_components=2, random_state=42)
)

# Aplikace a transformace dat
X_reduced = dim_reduction.fit_transform(X)
```