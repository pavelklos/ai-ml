# Další paradigmata strojového učení

## 1. Redukce dimenzionality

Techniky redukce dimenzionality transformují vysokodimenzionální data do prostoru s nižší dimenzí při zachování důležitých informací. Tyto metody pomáhají vizualizovat složitá data, snížit výpočetní nároky a zmírnit "prokletí dimenzionality".

### Příklady z reálného světa pro redukci dimenzionality

1. **Rozpoznávání obličejů**: Redukce vysokodimenzionálních obrazových dat obličejů na klíčové rysy, které obličeje odlišují
2. **Analýza genové exprese**: Kondenzace tisíců genových expresí do smysluplných vzorů pro klasifikaci nemocí
3. **Analýza textových dokumentů**: Transformace velkých matic dokument-termín do konceptuálních prostorů pro modelování témat
4. **Marketingová segmentace**: Redukce dimenzí atributů zákazníků pro vizualizaci a identifikaci zákaznických segmentů
5. **Zpracování signálů**: Komprese vysokodimenzionálních senzorických dat při zachování důležitých vlastností signálu

### Nejpoužívanější modely Scikit-learn pro redukci dimenzionality

1. `PCA` (Analýza hlavních komponent): Identifikuje lineární kombinace vlastností, které zachycují maximální rozptyl
2. `TruncatedSVD`: Pracuje s řídkými maticemi pro redukci dimenzionality textových dat
3. `TSNE` (t-Distributed Stochastic Neighbor Embedding): Zachovává lokální podobnosti pro vizualizaci
4. `UMAP`: Rychlejší alternativa k t-SNE, která lépe zachovává globální strukturu
5. `FactorAnalysis`: Modeluje korelace mezi proměnnými pomocí latentních faktorů

## 2. Souborové učení

Souborové učení kombinuje více modelů strojového učení k dosažení lepších prediktivních výsledků, než by bylo možné získat z jediného modelu. Snižuje rozptyl, zkreslení nebo zlepšuje predikce prostřednictvím různých kombinačních technik.

### Příklady z reálného světa pro souborové učení

1. **Hodnocení úvěrového rizika**: Kombinace více modelů pro přesnější predikci nesplácení úvěrů
2. **Lékařská diagnostika**: Spojení predikcí z více systémů pro zlepšení diagnostické přesnosti
3. **Předpověď počasí**: Agregace různých meteorologických modelů pro spolehlivější predikce
4. **Doporučovací systémy**: Kombinace různých doporučovacích algoritmů pro zlepšení kvality návrhů
5. **Detekce podvodů**: Využití více detekčních metod k identifikaci neobvyklých vzorů v transakcích

### Nejpoužívanější modely Scikit-learn pro souborové učení

1. `RandomForestClassifier` / `RandomForestRegressor`: Soubor rozhodovacích stromů využívající bagging
2. `GradientBoostingClassifier` / `GradientBoostingRegressor`: Sekvenční budování stromů pro opravování chyb předchůdců
3. `VotingClassifier` / `VotingRegressor`: Kombinuje různé modely prostřednictvím většinového hlasování nebo průměrování
4. `AdaBoostClassifier` / `AdaBoostRegressor`: Zaměřuje se na těžko klasifikovatelné příklady pomocí převážení
5. `StackingClassifier` / `StackingRegressor`: Používá predikce z více modelů jako příznaky pro meta-model

## 3. Detekce anomálií

Detekce anomálií identifikuje vzácné položky, události nebo pozorování, které se výrazně odchylují od většiny dat a vzbuzují podezření tím, že se liší od normálního chování. Tyto techniky jsou zásadní pro nalezení odlehlých hodnot nebo neobvyklých vzorů v datech.

### Příklady z reálného světa pro detekci anomálií

1. **Detekce podvodů**: Identifikace neobvyklých bankovních nebo kreditních transakcí, které mohou naznačovat podvod
2. **Síťová bezpečnost**: Detekce neobvyklých vzorů v síťovém provozu, které by mohly signalizovat pokusy o průnik
3. **Kontrola kvality výroby**: Nalezení vadných výrobků s anomálními charakteristikami
4. **Monitorování zdravotního stavu**: Detekce neobvyklých životních funkcí pacientů nebo laboratorních výsledků, které mohou indikovat naléhavé stavy
5. **Detekce poruch senzorů**: Identifikace nefunkčních senzorů v průmyslovém zařízení nebo systémech IoT

### Nejpoužívanější modely Scikit-learn pro detekci anomálií

1. `IsolationForest`: Efektivně izoluje anomálie pomocí rekurzivního dělení
2. `OneClassSVM`: Učí se hranici kolem normálních datových bodů
3. `LocalOutlierFactor`: Identifikuje lokální odchylky v hustotě dat
4. `EllipticEnvelope`: Předpokládá, že data pocházejí z Gaussovského rozdělení a identifikuje odlehlé hodnoty
5. `DBSCAN`: Shlukování založené na hustotě, které může označit jako odlehlé hodnoty body nepatřící do žádného shluku

## 4. Analýza časových řad

Analýza časových řad zahrnuje analýzu datových bodů sebraných v čase za účelem extrakce smysluplných statistik, identifikace vzorů a predikce budoucích hodnot. Tyto techniky zohledňují časové závislosti mezi pozorováními.

### Příklady z reálného světa pro analýzu časových řad

1. **Predikce cen akcií**: Předpovídání trendů finančních trhů na základě historických cenových dat
2. **Předpověď poptávky**: Predikce budoucí poptávky po produktech na základě sezónních a historických prodejních dat
3. **Modelování spotřeby energie**: Analýza a předpověď vzorců využití elektřiny nebo zdrojů
4. **Analýza návštěvnosti webových stránek**: Studium vzorů návštěvníků a predikce budoucího zatížení
5. **Predikce šíření nemocí**: Analýza míry infekce v čase pro předpověď progrese epidemie

### Nejpoužívanější modely kompatibilní se Scikit-learn pro analýzu časových řad

1. `Prophet` (Facebook): Zpracovává sezónnost a efekty svátků pro obchodní časové řady
2. Modely `ARIMA` (přes statsmodels): Tradiční statistický přístup k předpovědi časových řad
3. `TimeSeriesSplit`: Křížová validace pro časové řady (nejde o model, ale je zásadní pro validaci)
4. `HistGradientBoostingRegressor`: Dokáže efektivně pracovat s časově založenými příznaky
5. `RidgeCV` s časovými příznaky: Jednoduché lineární modely s regularizací pro časové řady

## 5. Semi-supervizované učení

Semi-supervizované učení používá kombinaci označených a neoznačených dat pro trénink. Je obzvláště cenné, když jsou označená data omezená, ale neoznačených dat je dostatek, přičemž využívá vzory v neoznačených datech ke zlepšení výkonu modelu.

### Příklady z reálného světa pro semi-supervizované učení

1. **Klasifikace webových stránek**: Využití malé sady označených webových stránek ke kategorizaci velkého množství neoznačených stránek
2. **Analýza medicínských snímků**: Využití omezených anotovaných medicínských skenů s většími neanotovanými datasety
3. **Rozpoznávání řeči**: Zlepšení modelů s omezeným přepisem zvuku pomocí rozsáhlých nepřepsaných nahrávek
4. **Predikce struktury proteinů**: Využití omezených známých struktur k pomoci při predikci neznámých proteinových struktur
5. **Klasifikace textových dokumentů**: Kategorizace velkých kolekcí dokumentů, kde je ručně označena pouze podmnožina

### Nejpoužívanější modely Scikit-learn pro semi-supervizované učení

1. `LabelPropagation`: Šíří informace o štítcích do neoznačených dat pomocí metod založených na grafech
2. `LabelSpreading`: Podobné jako LabelPropagation, ale odolnější vůči šumu
3. `SelfTrainingClassifier`: Iterativně označuje neoznačená data s vysokou důvěrou pro přetrénování modelu
4. `Co-training` (vlastní implementace): Používá různé pohledy na data k bootstrappingu učení
5. `Semi-supervised SVM` (vlastní implementace): Přizpůsobuje SVM k využití neoznačených dat během tréninku