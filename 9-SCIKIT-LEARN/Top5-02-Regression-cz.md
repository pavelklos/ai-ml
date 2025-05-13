# Regression

## Regresní modely v knihovně Scikit-learn

Scikit-learn poskytuje mnoho regresních algoritmů pro předpovídání spojitých hodnot. V tomto článku se zaměříme na 5 nejpoužívanějších regresních modelů, jejich vlastnosti, kód a praktické využití.

## 1. LinearRegression

### Popis

LinearRegression je nejjednodušší regresní model, který předpokládá lineární vztah mezi vstupními proměnnými a cílovou proměnnou. Model se snaží najít nejlepší přímku (nebo nadrovinu ve více dimenzích), která minimalizuje součet čtverců chyb mezi skutečnými a předpovězenými hodnotami.

### Kdy a kde použít

- Když existuje lineární závislost mezi proměnnými
- Pro jednoduché a interpretovatelné modely
- Jako výchozí (baseline) model
- Pro pochopení vlivu jednotlivých vlastností na cílovou proměnnou

### Výhody a nevýhody

**Výhody:**
- Jednoduchá implementace a rychlý výpočet
- Vysoká interpretovatelnost (koeficienty přímo určují vliv proměnných)
- Nízké riziko přetrénování pro jednoduché problémy
- Malý počet hyperparametrů

**Nevýhody:**
- Nedokáže modelovat nelineární vztahy
- Citlivý na odlehlé hodnoty
- Předpokládá nezávislost vlastností
- Může podtrénovat u komplexnějších dat

### Ukázka kódu

````python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Načtení datové sady
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Rozdělení dat na trénovací a testovací část
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vytvoření a trénování modelu
model = LinearRegression()
model.fit(X_train, y_train)

# Predikce
y_pred = model.predict(X_test)

# Vyhodnocení modelu
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Střední kvadratická chyba: {mse:.4f}")
print(f"Koeficient determinace R²: {r2:.4f}")

# Analýza koeficientů
coefficients = pd.DataFrame({
    'Vlastnost': housing.feature_names,
    'Koeficient': model.coef_
})
print("\nKoeficienty modelu:")
print(coefficients)

# Vizualizace skutečných vs. předpovězených hodnot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Předpovězené hodnoty')
plt.title('Skutečné vs. předpovězené hodnoty')
plt.grid(True)
plt.show()

# Vizualizace reziduí
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test - y_pred, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
plt.xlabel('Předpovězené hodnoty')
plt.ylabel('Rezidua')
plt.title('Rezidua')
plt.grid(True)
plt.show()
````

## 2. RandomForestRegressor

### Popis

RandomForestRegressor je ensemble metoda založená na rozhodovacích stromech. Vytváří několik rozhodovacích stromů na náhodných podmnožinách trénovacích dat a používá průměr jejich výstupů k finální předpovědi. Tento přístup pomáhá snižovat rozptyl a zlepšuje generalizaci.

### Kdy a kde použít

- Pro komplexní nelineární vztahy mezi daty
- Když existuje mnoho vlastností, včetně potenciálně irelevantních
- Pro úlohy vyžadující dobrou generalizaci bez nadměrného ladění
- Když potřebujete stanovit důležitost jednotlivých vlastností

### Výhody a nevýhody

**Výhody:**
- Zvládá nelineární závislosti bez nutnosti transformace dat
- Robustní vůči odlehlým hodnotám
- Automaticky vybírá důležité vlastnosti
- Nepotřebuje škálování vstupních dat

**Nevýhody:**
- Méně interpretovatelný než lineární modely
- Výpočetně náročnější
- Sklon k přetrénování při nevhodných hyperparametrech
- Vyžaduje více paměti a času při trénování na velkých datech

### Ukázka kódu

````python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Načtení datové sady (v novějších verzích scikit-learn použijte alternativní dataset)
# V případě nedostupnosti lze použít: from sklearn.datasets import fetch_california_housing
try:
    boston = load_boston()
    X, y = boston.data, boston.target
    feature_names = boston.feature_names
except:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names

# Rozdělení dat na trénovací a testovací část
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vytvoření a trénování modelu
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predikce
y_pred = rf.predict(X_test)

# Vyhodnocení modelu
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Střední kvadratická chyba: {mse:.4f}")
print(f"Střední absolutní chyba: {mae:.4f}")
print(f"Koeficient determinace R²: {r2:.4f}")

# Ladění hyperparametrů
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nNejlepší hyperparametry: {grid_search.best_params_}")

# Použití nejlepšího modelu
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"MSE nejlepšího modelu: {mse_best:.4f}")
print(f"R² nejlepšího modelu: {r2_best:.4f}")

# Důležitost vlastností
feature_importances = pd.DataFrame({
    'Vlastnost': feature_names,
    'Důležitost': best_rf.feature_importances_
})
feature_importances = feature_importances.sort_values('Důležitost', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Vlastnost'], feature_importances['Důležitost'])
plt.xlabel('Důležitost')
plt.title('Důležitost vlastností')
plt.tight_layout()
plt.show()

# Vizualizace skutečných vs. předpovězených hodnot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Předpovězené hodnoty')
plt.title('Skutečné vs. předpovězené hodnoty')
plt.grid(True)
plt.show()
````

## 3. SVR (Support Vector Regressor)

### Popis

SVR je implementace algoritmu Support Vector Machine pro regresi. Cílem je najít funkci, která se od všech trénovacích bodů odchyluje maximálně o hodnotu epsilon, přičemž je co nejvíce plochá. Používá tzv. "kernel trick" pro transformaci dat do vyšší dimenze, kde může být problém řešitelný lineárně.

### Kdy a kde použít

- Pro nelineární regresní problémy
- Pro datasety střední velikosti
- Když potřebujete vysokou přesnost a regularizaci
- Když jsou data zašuměná

### Výhody a nevýhody

**Výhody:**
- Efektivní ve vysokodimenzionálních prostorech
- Dobře funguje i s omezeným množstvím dat
- Versatilní díky různým typům kernelů
- Odolný vůči odlehlým hodnotám (s vhodným nastavením)

**Nevýhody:**
- Výpočetně náročný pro velké datasety
- Vyžaduje pečlivé ladění hyperparametrů
- Složitá interpretace modelu
- Citlivý na škálování vstupních dat

### Ukázka kódu

````python
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, randint

# Generování syntetických dat
X, y = make_regression(n_samples=500, n_features=5, noise=0.1, random_state=42)

# Rozdělení dat na trénovací a testovací část
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizace dat
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Vytvoření a trénování základních modelů s různými kernely
kernels = ['linear', 'poly', 'rbf']
svr_models = {}

for kernel in kernels:
    svr = SVR(kernel=kernel)
    svr.fit(X_train_scaled, y_train_scaled)
    svr_models[kernel] = svr

# Vyhodnocení modelů
for kernel, model in svr_models.items():
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"SVR s '{kernel}' kernelem:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")

# Ladění hyperparametrů pro RBF kernel
param_distributions = {
    'C': uniform(0.1, 100),
    'gamma': uniform(0.001, 1),
    'epsilon': uniform(0.01, 1)
}

random_search = RandomizedSearchCV(
    SVR(kernel='rbf'),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train_scaled)

print(f"\nNejlepší hyperparametry: {random_search.best_params_}")

# Použití nejlepšího modelu
best_svr = random_search.best_estimator_
y_pred_scaled_best = best_svr.predict(X_test_scaled)
y_pred_best = scaler_y.inverse_transform(y_pred_scaled_best.reshape(-1, 1)).ravel()

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"MSE nejlepšího modelu: {mse_best:.4f}")
print(f"R² nejlepšího modelu: {r2_best:.4f}")

# Vizualizace skutečných vs. předpovězených hodnot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Předpovězené hodnoty')
plt.title('Skutečné vs. předpovězené hodnoty (SVR)')
plt.grid(True)
plt.show()

# Vizualizace reziduí
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_best, y_test - y_pred_best, alpha=0.5)
plt.hlines(y=0, xmin=y_pred_best.min(), xmax=y_pred_best.max(), colors='r', linestyles='--')
plt.xlabel('Předpovězené hodnoty')
plt.ylabel('Rezidua')
plt.title('Rezidua (SVR)')
plt.grid(True)
plt.show()
````

## 4. GradientBoostingRegressor

### Popis

GradientBoostingRegressor je pokročilá ensemble metoda, která buduje model postupně v sekvenci. Každý nový model se učí z chyb předchozích modelů, což vede k postupnému zlepšování prediktivní schopnosti. Typicky používá rozhodovací stromy jako základní modely.

### Kdy a kde použít

- Pro komplexní regresní úlohy
- Když požadujete vysokou přesnost predikce
- Pro strukturovaná tabulární data
- V soutěžích strojového učení (často patří mezi nejúspěšnější algoritmy)

### Výhody a nevýhody

**Výhody:**
- Vynikající prediktivní výkon
- Zvládá různé typy dat a vztahů
- Robustnost vůči odlehlým hodnotám
- Poskytuje důležitost vlastností

**Nevýhody:**
- Výpočetně náročný při trénování
- Riziko přetrénování při špatném nastavení
- Složitější ladění hyperparametrů
- Méně interpretovatelný než jednodušší modely

### Ukázka kódu

````python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Načtení datové sady
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Rozdělení dat na trénovací a testovací část
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vytvoření a trénování modelu
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

# Predikce
y_pred = gb.predict(X_test)

# Vyhodnocení modelu
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Střední kvadratická chyba: {mse:.4f}")
print(f"Koeficient determinace R²: {r2:.4f}")

# Sledování chyby během trénování
test_score = np.zeros((gb.n_estimators,), dtype=np.float64)
for i, y_pred in enumerate(gb.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

train_score = np.zeros((gb.n_estimators,), dtype=np.float64)
for i, y_pred in enumerate(gb.staged_predict(X_train)):
    train_score[i] = mean_squared_error(y_train, y_pred)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(gb.n_estimators) + 1, train_score, 'b-',
         label='Trénovací MSE')
plt.plot(np.arange(gb.n_estimators) + 1, test_score, 'r-',
         label='Testovací MSE')
plt.xlabel('Počet stromů')
plt.ylabel('Střední kvadratická chyba')
plt.title('Vývoj MSE během trénování')
plt.legend()
plt.grid(True)
plt.show()

# Důležitost vlastností
feature_importances = pd.DataFrame({
    'Vlastnost': housing.feature_names,
    'Důležitost': gb.feature_importances_
})
feature_importances = feature_importances.sort_values('Důležitost', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Vlastnost'], feature_importances['Důležitost'])
plt.xlabel('Důležitost')
plt.title('Důležitost vlastností')
plt.tight_layout()
plt.show()

# Křivka učení
train_sizes, train_scores, test_scores = learning_curve(
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='neg_mean_squared_error')

train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="b")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="r")
plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Trénovací MSE")
plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Validační MSE")
plt.xlabel("Počet trénovacích vzorků")
plt.ylabel("Střední kvadratická chyba")
plt.title("Křivka učení pro GradientBoostingRegressor")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Vizualizace skutečných vs. předpovězených hodnot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Předpovězené hodnoty')
plt.title('Skutečné vs. předpovězené hodnoty')
plt.grid(True)
plt.show()
````

## 5. ElasticNet

### Popis

ElasticNet je lineární regresní model s regularizací, která kombinuje penalty L1 a L2. Tento model je užitečný, když existuje mnoho korelovaných vlastností. Kombinuje vlastnosti Lasso (výběr vlastností) a Ridge regrese (zmenšování koeficientů), což pomáhá řešit multikolinearitu a předcházet přetrénování.

### Kdy a kde použít

- Pro data s mnoha korelovanými vlastnostmi
- Pro řídké modely (s mnoha nulovými koeficienty)
- Když chcete automatický výběr vlastností
- Pro finanční modelování, genomická data a další vysocedimenzionální problémy

### Výhody a nevýhody

**Výhody:**
- Kombinuje to nejlepší z Lasso a Ridge regrese
- Efektivní výběr vlastností
- Zvládá multikolinearitu
- Poskytuje interpretovatelnější modely než složitější algoritmy

**Nevýhody:**
- Stále omezen na lineární vztahy
- Vyžaduje pečlivé ladění dvou regularizačních parametrů (alpha a l1_ratio)
- Méně výkonný pro nelineární data
- Pro vysokou přesnost potřebuje správně škálovaná data

### Ukázka kódu

````python
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generování syntetických dat s korelovanými vlastnostmi
X, y = make_regression(n_samples=1000, n_features=50, n_informative=10, 
                      noise=0.5, random_state=42)

# Rozdělení dat na trénovací a testovací část
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizace dat
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Vytvoření a trénování různých modelů pro porovnání
models = {
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'Ridge': Ridge(alpha=0.1, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'model': model
    }
    print(f"{name}:")
    print(f"  MSE: {results[name]['mse']:.4f}")
    print(f"  R²: {results[name]['r2']:.4f}")
    print(f"  Počet nulových koeficientů: {np.sum(model.coef_ == 0)}/{len(model.coef_)}")

# Ladění hyperparametrů pro ElasticNet
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid_search = GridSearchCV(
    ElasticNet(random_state=42, max_iter=10000),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nNejlepší hyperparametry: {grid_search.best_params_}")

# Použití nejlepšího modelu
best_elastic = grid_search.best_estimator_
y_pred_best = best_elastic.predict(X_test_scaled)

mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"MSE nejlepšího modelu: {mse_best:.4f}")
print(f"R² nejlepšího modelu: {r2_best:.4f}")
print(f"Počet nulových koeficientů: {np.sum(best_elastic.coef_ == 0)}/{len(best_elastic.coef_)}")

# Vizualizace koeficientů různých modelů
non_zero_coef = np.where(np.abs(results['ElasticNet']['model'].coef_) > 1e-10)[0]
selected_features = non_zero_coef[:10]  # Zobrazíme jen prvních 10 nenulových vlastností

coefs = pd.DataFrame()
for name, result in results.items():
    coefs[name] = result['model'].coef_
coefs['Vlastnost'] = [f"X{i}" for i in range(X.shape[1])]

plt.figure(figsize=(12, 8))
for i, feature in enumerate(selected_features):
    plt.subplot(2, 5, i + 1)
    for name in models.keys():
        plt.bar(name, coefs.loc[feature, name])
    plt.title(f"Vlastnost {feature}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Vizualizace cesty regularizace
from sklearn.linear_model import enet_path

alphas, coefs, _ = enet_path(X_train_scaled, y_train, l1_ratio=0.5, 
                            fit_intercept=False, return_models=False)

plt.figure(figsize=(12, 6))
plt.plot(-np.log10(alphas), coefs.T)
plt.xlabel('-log(alpha)')
plt.ylabel('Koeficienty')
plt.title('Cesta regularizace pro ElasticNet (l1_ratio=0.5)')
plt.tight_layout()
plt.show()

# Vizualizace skutečných vs. předpovězených hodnot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Předpovězené hodnoty')
plt.title('Skutečné vs. předpovězené hodnoty (ElasticNet)')
plt.grid(True)
plt.show()
````

## Srovnávací tabulka regresních modelů

| Model | Rychlost trénování | Rychlost predikce | Interpretovatelnost | Nelineární vztahy | Odlehlé hodnoty | Vysokodim. data | Hyperparametry | Hlavní výhody | Hlavní nevýhody |
|-------|-------------------|-------------------|---------------------|-------------------|----------------|-----------------|--------------------|--------------|----------------|
| LinearRegression | Velmi rychlé | Velmi rychlé | Vysoká | Ne | Citlivý | Průměrné | Minimum | Jednoduchost, interpretace | Pouze lineární vztahy |
| RandomForestRegressor | Střední | Rychlé | Střední | Ano | Robustní | Dobré | Více | Vysoká přesnost, robustnost | Výpočetní náročnost, černá skříňka |
| SVR | Pomalé | Střední | Nízká | Ano | Robustní | Průměrné | Více | Přesnost pro komplexní data | Výpočetní náročnost, nutnost škálování |
| GradientBoostingRegressor | Pomalé | Střední | Nízká | Ano | Středně robustní | Dobré | Mnoho | Vysoká přesnost, versatilita | Složité ladění, černá skříňka |
| ElasticNet | Rychlé | Velmi rychlé | Vysoká | Ne | Středně citlivý | Velmi dobré | Několik | Výběr vlastností, regularizace | Omezen na lineární vztahy |