# Analýza datové sady Iris

## Hlavní body notebooku

Tento Jupyter Notebook představuje komplexní framework pro analýzu datové sady Iris. Zaměřuje se na všechny klíčové fáze projektu strojového učení:

1. **Import knihoven** - používá pandas, numpy, matplotlib, seaborn, scikit-learn a další pro analýzu dat
2. **Načtení a průzkum dat** - pracuje s datovou sadou Iris
3. **Předzpracování dat** - zahrnuje identifikaci typů sloupců, rozdělení na numerické a kategorické
4. **Vytvoření a trénování modelů** - implementuje několik klasifikačních algoritmů
5. **Vyhodnocení modelů** - využívá metriky jako přesnost, F1 skóre a matici záměn
6. **Vizualizace výsledků** - pomocí grafů porovnává výkon modelů
7. **Předpovědi** - ukazuje, jak používat natrénovaný model pro predikce na nových datech

## Použité techniky a algoritmy

- **Předzpracování**: StandardScaler, OneHotEncoder, SimpleImputer
- **Modely**: 
  - LogisticRegression
  - RandomForestClassifier 
  - DecisionTreeClassifier
  - SVC (Support Vector Classification)
  - KNeighborsClassifier
- **Křížová validace**: rozdělení dat na trénovací a testovací sadu
- **Vyhodnocení**: přesnost, přesnost, úplnost, F1 skóre, matice záměn
- **Vizualizace**: porovnání výkonu modelů, křivka učení, matice korelace, důležitost příznaků

## Výsledky

Notebook analyzuje klasifikační problém se třemi třídami květin Iris (setosa, versicolor, virginica) podle jejich fyzických charakteristik. Modely dosahují vysoké přesnosti díky jednoduchosti a dobré separovatelnosti datové sady Iris.

Z výsledků vyplývá, že:
- RandomForest a SVM modely obvykle dosahují nejvyšší přesnosti
- Důležité příznaky pro klasifikaci jsou délka a šířka okvětních lístků
- Většina modelů dosahuje přesnosti přes 90%

## Shrnutí

Tento notebook představuje komplexní a znovupoužitelný framework pro klasifikační úlohy strojového učení. Demonstruje celý proces od načtení dat po nasazení modelu na příkladu klasické datové sady Iris. Zahrnuje všechny důležité kroky datové analýzy a poskytuje dobrý základ pro pokročilejší projekty strojového učení.

### Další kroky

1. **Vylepšení modelu**:
   - Ladění hyperparametrů
   - Pokročilejší inženýrství příznaků
   - Vyzkoušení ensemble metod

2. **Další analýza**:
   - Hlubší zkoumání chybných klasifikací
   - Detailnější analýza příznaků

3. **Nasazení**:
   - Serializace modelu pro produkci
   - Vytvoření API pro inferenci modelu
   - Nastavení monitorování výkonu modelu

Tento notebook je ideální jako šablona pro podobné klasifikační úlohy nebo jako výukový materiál pro pochopení základních konceptů strojového učení.