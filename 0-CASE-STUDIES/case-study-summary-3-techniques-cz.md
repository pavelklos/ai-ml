# Případové studie strojového učení: Regrese, klasifikace a shlukování

## 1. Regrese

Regrese je technika učení s učitelem používaná k predikci spojitých numerických hodnot na základě vstupních vlastností. Model se učí porozumět vztahu mezi nezávislými proměnnými (vlastnostmi) a závislou proměnnou (cílem).

### Příklady z reálného světa pro regresi

1. **Predikce cen nemovitostí**: Předpovídání cen nemovitostí na základě vlastností jako výměra, lokalita, počet ložnic atd.

2. **Předpověď prodeje**: Předpovídání budoucího objemu prodejů na základě historických prodejních dat, marketingových výdajů, sezónnosti a ekonomických ukazatelů.

3. **Predikce lékařského dávkování**: Určení vhodných dávek léků na základě charakteristik pacienta jako věk, váha, anamnéza a biomarkery.

4. **Předpověď spotřeby energie**: Předpovídání spotřeby elektřiny nebo plynu pro budovy na základě povětrnostních podmínek, denní doby, obsazenosti a historických vzorců.

5. **Predikce cen akcií**: Předpovídání cen akcií nebo tržních indexů na základě historických cenových dat, objemů obchodování, finančních výkazů společností a tržních ukazatelů.

### Nejpoužívanější modely Scikit-learn pro regresi

1. `LinearRegression`: Jednoduchý a interpretovatelný model pro lineární vztahy
2. `RandomForestRegressor`: Metoda souborových modelů, která dobře funguje pro složité nelineární vztahy
3. `GradientBoostingRegressor`: Výkonný boosting algoritmus, který často dosahuje vysoké přesnosti
4. `ElasticNet`: Regularizovaná regrese, která řeší multikolinearitu a výběr vlastností
5. `SVR` (Support Vector Regressor): Efektivní pro středně velké datasety se složitými vzory

## 2. Klasifikace

Klasifikace je technika učení s učitelem používaná ke kategorizaci dat do diskrétních tříd nebo štítků. Model se učí rozhodovací hranice, které oddělují různé třídy na základě vstupních vlastností.

### Příklady z reálného světa pro klasifikaci

1. **Detekce spamu v emailech**: Klasifikace emailů jako spam nebo legitimní na základě obsahu, informací o odesílateli a metadat.

2. **Predikce odchodu zákazníků**: Předpovídání, zda zákazníci opustí službu na základě vzorců používání, demografických údajů a interakcí se zákaznickou podporou.

3. **Diagnostika nemocí**: Klasifikace zdravotních stavů na základě příznaků, laboratorních výsledků, obrazových dat a anamnézy pacienta.

4. **Hodnocení úvěrového rizika**: Určení, zda žadatelé o půjčku pravděpodobně nesplatí své závazky, na základě úvěrové historie, příjmu, existujících dluhů a dalších finančních ukazatelů.

5. **Analýza sentimentu**: Kategorizace textových recenzí nebo příspěvků na sociálních sítích jako pozitivní, negativní nebo neutrální na základě obsahu.

### Nejpoužívanější modely Scikit-learn pro klasifikaci

1. `LogisticRegression`: Jednoduchý a interpretovatelný základní model s dobrým výkonem
2. `RandomForestClassifier`: Robustní metoda souborových modelů, která dobře zvládá nelineární vztahy
3. `GradientBoostingClassifier`: Vysoce výkonný boosting algoritmus, který často vede žebříčky
4. `SVC` (Support Vector Classifier): Výkonný nástroj pro složité rozhodovací hranice ve středně velkých datasetech
5. `KNeighborsClassifier`: Jednoduchý, ale efektivní přístup založený na učení z instancí

## 3. Shlukování

Shlukování je technika učení bez učitele používaná k seskupování podobných datových bodů na základě jejich vnitřních vlastností. Na rozdíl od učení s učitelem shlukování nevyužívá označená data a místo toho identifikuje přirozená seskupení.

### Příklady z reálného světa pro shlukování

1. **Segmentace zákazníků**: Seskupování zákazníků na základě nákupního chování, demografických údajů a vzorců zapojení pro cílený marketing.

2. **Komprese obrazu**: Snížení barevné složitosti v obrazech seskupováním podobných barev a reprezentací každé skupiny jedinou barvou.

3. **Detekce anomálií**: Identifikace neobvyklých vzorů v datech nalezením bodů, které dobře nepatří do žádného shluku, užitečné při detekci podvodů a monitorování systémů.

4. **Shlukování dokumentů**: Seskupování podobných dokumentů na základě obsahové podobnosti pro organizování velkých kolekcí nebo doporučovací systémy.

5. **Analýza genové exprese**: Shlukování genů s podobnými expresními vzorci napříč různými experimentálními podmínkami k identifikaci funkčně příbuzných genů.

### Nejpoužívanější modely Scikit-learn pro shlukování

1. `KMeans`: Rychlý a jednoduchý algoritmus pro vytváření sférických shluků
2. `DBSCAN`: Přístup založený na hustotě, který dokáže najít shluky libovolného tvaru a identifikovat odlehlé hodnoty
3. `AgglomerativeClustering`: Technika hierarchického shlukování, která vytváří vnořené shluky
4. `MeanShift`: Technika pro nalezení hustých oblastí v datech bez nutnosti specifikovat počet shluků
5. `GaussianMixture`: Pravděpodobnostní model pro měkké shlukování pomocí Gaussovských distribucí