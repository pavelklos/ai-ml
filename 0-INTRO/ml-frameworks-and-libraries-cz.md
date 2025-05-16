# Knihovny a frameworky pro strojové učení

## 1. Obecné knihovny pro strojové učení

- **Scikit-learn**: Komplexní knihovna s implementacemi pro regresi, klasifikaci, shlukování, redukci dimenzionality, výběr modelu a předzpracování dat
- **TensorFlow/Keras**: Framework pro hluboké učení podporující všechna paradigmata strojového učení s přístupy neuronových sítí
- **PyTorch**: Flexibilní framework pro hluboké učení s dynamickým výpočetním grafem
- **JAX**: Vysoce výkonné numerické výpočty a výzkum strojového učení
- **XGBoost**: Optimalizovaná knihovna pro gradientní posilování pro regresi a klasifikaci
- **LightGBM**: Vysoce výkonný framework pro gradientní posilování od Microsoftu
- **CatBoost**: Knihovna pro gradientní posilování s vynikající podporou pro kategorické proměnné
- **H2O**: Škálovatelná, distribuovaná platforma pro strojové učení
- **PyCaret**: Knihovna s nízkým množstvím kódu pro rychlé prototypování
- **RAPIDS cuML**: GPU-akcelerované algoritmy strojového učení

## 2. Knihovny zaměřené na regresi

- **Statsmodels**: Specializace na statistické modely a testování hypotéz
- **Scikit-Garden**: Rozšíření scikit-learn, včetně kvantilové regrese
- **PyMC**: Pravděpodobnostní programování pro bayesovskou regresi
- **Edward**: Pravděpodobnostní programovací jazyk pro bayesovskou regresi
- **Prophet**: Nástroj od Facebooku pro předpovídání časových řad

## 3. Knihovny zaměřené na klasifikaci

- **Vowpal Wabbit**: Rychlé online učení pro klasifikaci
- **FastText**: Knihovna pro klasifikaci textu a učení reprezentace
- **Thundersvm**: Knihovna SVM podporující GPU
- **LIBSVM**: Populární knihovna SVM
- **Imbalanced-learn**: Specializovaná pro problémy s nevyváženou klasifikací

## 4. Knihovny zaměřené na shlukování

- **HDBSCAN**: Hierarchické shlukování založené na hustotě
- **FAISS**: Facebook AI Similarity Search pro efektivní shlukování velkých datasetů
- **pyclustering**: Kolekce shlukovacích algoritmů
- **SciPy**: Obsahuje několik shlukovacích algoritmů
- **UMAP**: Manifold learning a shlukování
- **BIRCH**: Implementace shlukovacího algoritmu BIRCH

## 5. Knihovny pro redukci dimenzionality

- **UMAP-learn**: Uniform Manifold Approximation and Projection
- **Scikit-dim**: Odhad vnitřní dimenze
- **TensorFlow Embedding Projector**: Pro vizualizaci vysokodimenzionálních dat
- **Manifold**: Různé metody manifold learning
- **Ivis**: Strukturu zachovávající redukce dimenzionality

## 6. Knihovny pro souborové učení (Ensemble Learning)

- **ML-Ensemble**: Meta-estimátory souborů kompatibilní se scikit-learn
- **DESlib**: Knihovna pro dynamický výběr souborů
- **Stacking**: Implementace technik vrstvení souborů
- **SuperLearner**: Pythonová implementace algoritmu Super Learner
- **Auto-Sklearn**: Automatizovaná konstrukce souborů a ladění hyperparametrů

## 7. Knihovny pro detekci anomálií

- **PyOD**: Komplexní knihovna pro detekci odlehlých hodnot
- **TODS**: Systém pro detekci odlehlých hodnot v časových řadách
- **Alibi Detect**: Algoritmy pro detekci odlehlých hodnot, nepřátelských příkladů a posunu dat
- **Luminaire**: Detekce anomálií pro časové řady
- **GluonTS**: Sada nástrojů Amazonu pro detekci anomálií v časových řadách

## 8. Knihovny pro analýzu časových řad

- **Sktime**: Jednotné rozhraní pro strojové učení časových řad
- **Darts**: Uživatelsky přívětivá moderní knihovna pro časové řady
- **STUMPY**: Výkonná a efektivní analýza časových řad
- **Kats**: Sada nástrojů Facebooku pro analýzu časových řad
- **Greykite**: Flexibilní knihovna pro předpovídání od LinkedInu
- **Orbit**: Framework Uberu pro předpovídání časových řad
- **Tslearn**: Sada nástrojů strojového učení věnovaná datům časových řad
- **PyTS**: Balíček Python pro klasifikaci časových řad

## 9. Knihovny pro semi-supervizované učení

- **Semi-Supervised-Learning**: Implementace různých semi-supervizovaných algoritmů
- **FixMatch**: Implementace algoritmu FixMatch
- **Structured Semi-Supervised Learning**: Balíček pro strukturované předpovídání výstupů
- **Label Propagation**: Různé implementace algoritmů šíření štítků
- **Snorkel**: Framework pro programové označování dat

## 10. Knihovny pro nasazení modelů

- **MLflow**: Platforma pro kompletní životní cyklus strojového učení
- **BentoML**: Framework pro poskytování a nasazování modelů strojového učení
- **TensorFlow Serving**: Systém pro poskytování modelů strojového učení
- **Cortex**: Nasazování modelů strojového učení v produkci
- **TorchServe**: Flexibilní a snadno použitelná knihovna pro poskytování modelů pro PyTorch

Tento seznam pokrývá širokou škálu frameworků pro každé paradigma strojového učení, od obecných knihoven až po specializované nástroje pro konkrétní úkoly. Výběr závisí na vašich specifických požadavcích, charakteristikách datasetu a výpočetních zdrojích.

---

Scikit-learn je skutečně jednou z nejpoužívanějších a nejvšestrannějších knihoven pro strojové učení v Pythonu. Je výbornou volbou pro mnoho úloh strojového učení z několika důvodů:

### Přednosti scikit-learn:

- **Komplexní algoritmy**: Pokrývá většinu tradičních algoritmů ML (regrese, klasifikace, shlukování, redukce dimenzionality)
- **Konzistentní API**: Používá jednotné rozhraní napříč všemi modely (fit, predict, transform)
- **Výborná dokumentace**: Skvělá dokumentace s příklady
- **Připraveno pro produkci**: Stabilní, optimalizované implementace
- **Integrace**: Bezproblémová spolupráce s pandas, NumPy a dalšími nástroji pro práci s daty v Pythonu

### Kdy používat scikit-learn:

- **Problémy s tabulkovými daty**: Vynikající pro strukturovaná data
- **Standardní úlohy ML**: Pro většinu potřeb regrese, klasifikace a shlukování
- **Vývoj prototypů**: Rychlé experimentování a základní modely
- **Malé až střední datasety**: Efektivně pracuje s datasety, které se vejdou do paměti

### Kdy zvážit jiné knihovny:

- **Hluboké učení**: TensorFlow/PyTorch jsou specializované pro neuronové sítě
- **Velká data**: Pro datasety, které se nevejdou do paměti, zvažte PySpark nebo Dask
- **Gradientní boosting**: I když má scikit-learn implementace, XGBoost, LightGBM a CatBoost často poskytují lepší výkon
- **Časové řady**: Specializované knihovny jako Prophet nebo statsmodels nabízejí specifičtější funkcionalitu
- **Pokročilé NLP**: HuggingFace Transformers nebo spaCy pro moderní úlohy zpracování přirozeného jazyka

Scikit-learn je vynikající základ pro většinu ML projektů a dobrý výchozí bod, ale v závislosti na vašich specifických potřebách možná budete potřebovat doplnit jej o jiné specializované knihovny, jak vaše projekty rostou v komplexitě.