# Predykcja założenia lokaty terminowej — UCI Bank Marketing

Klasyfikacja binarna na silnie niezbalansowanym zbiorze danych (88.7% / 11.3%): czy klient banku założy lokatę terminową w wyniku kampanii telemarketingowej. Projekt porównuje pięć modeli — RandomForest, SVC, KNN, MLP (scikit-learn) oraz sieć neuronową w TensorFlow/Keras — w pełnym pipeline: EDA → feature engineering → strojenie hiperparametrów (GridSearchCV) → dobór progu decyzyjnego → ewaluacja na zbiorze testowym.

**Stack:** Python · pandas · scikit-learn · imbalanced-learn · TensorFlow/Keras · matplotlib/seaborn · Google Colab (GPU)

---

## 1. Problem

Bank prowadzi telefoniczne kampanie marketingowe. Celem jest model, który wskaże klientów najbardziej skłonnych do założenia lokaty — tak, by call center dzwoniło tam, gdzie ma to sens.

Kluczowa trudność: **niezbalansowanie klas 89/11**. Naiwny klasyfikator mówiący zawsze „nie" osiąga 89% accuracy, nie ucząc się niczego. Dlatego accuracy nie jest w tym projekcie metryką oceny — **metryką wiodącą jest PR-AUC** (pole pod krzywą precyzja–czułość), które jako jedyne nie jest zawyżane przez dużą liczbę prawdziwych negatywów. Losowy klasyfikator osiąga tu PR-AUC = 0.113; wszystkie modele przekroczyły 0.43.

## 2. Dane

- **Źródło:** [Bank Marketing Dataset (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing) — wariant `bank-additional-full.csv`
- **Rozmiar:** 41 188 obserwacji × 21 kolumn (20 cech + cel `y`)
- **Cechy:** demografia klienta (wiek, zawód, wykształcenie), produkty kredytowe, historia kontaktów oraz wskaźniki makroekonomiczne (Euribor 3M, stopa zatrudnienia, indeksy konsumenckie)
- **Rozkład celu:** 36 548 „no" (88.7%) / 4 640 „yes" (11.3%)

Plik CSV nie jest częścią repozytorium — instrukcja pobrania w sekcji [Jak uruchomić](#8-jak-uruchomić).

## 3. EDA — najważniejsze obserwacje

![Analiza zmiennych numerycznych](wykresy/analiza_numeryczna.png)

![Analiza zmiennych kategorycznych](wykresy/analiza_kategoryczna.png)

- **`poutcome` = success to najsilniejszy pojedynczy predyktor** — klienci, którzy kupili w poprzedniej kampanii, konwertują w 65% przypadków (≈ 6× średnia).
- **Wskaźniki makroekonomiczne mają najsilniejszą realną korelację z celem** (`nr.employed` −0.35, `euribor3m` −0.31, `emp.var.rate` −0.30) — gorsza koniunktura sprzyja lokatom. Jednocześnie są silnie współliniowe między sobą (0.91–0.97).
- **Kontakt komórkowy konwertuje ~3× lepiej** niż stacjonarny (15% vs 5%); miesiące o niskim wolumenie kampanii (mar, dec, sep, oct) mają skuteczność 44–51%.
- **`duration` (czas rozmowy) ma najwyższą korelację z celem (0.41), ale to klasyczny data leakage** — czas rozmowy znamy dopiero po jej zakończeniu.
- Cechy bez sygnału: `housing`, `loan`, `day_of_week`, `marital`, `cons.conf.idx` — usunięte z modelu.

## 4. Przygotowanie danych

| Przekształcenie | Uzasadnienie |
|---|---|
| Usunięcie `duration` | **Data leakage** — cecha znana dopiero po rozmowie; mimo najsilniejszej korelacji z celem (0.41) jej użycie zawyżyłoby wyniki w sposób nieosiągalny w praktyce (ROC-AUC ~0.93) |
| `pdays` → binarna `contacted_before` | Wartość 999 (brak wcześniejszego kontaktu) dotyczy ~96% rekordów — placeholder zaburzał sygnał |
| Winsoryzacja `campaign` (clip do 6) | Długi ogon outlierów do 56 prób kontaktu — prawdopodobne błędy CRM |
| Binning `age` → `age_group` | Sygnał nieliniowy (młodzi i seniorzy konwertują wyżej); surowy `age` zostaje dla modeli drzewiastych |

**Finalne cechy: 15** (8 numerycznych + 7 kategorycznych), po one-hot encoding **49 kolumn**. Kodowanie i skalowanie przez `ColumnTransformer` (StandardScaler + OneHotEncoder) **dopasowany wyłącznie na zbiorze treningowym** — brak wycieku do testu. Podział train/test 80/20 ze stratyfikacją (`random_state=42`).

## 5. Metodyka

**Strojenie:** każdy model scikit-learn przeszedł `GridSearchCV` z 10-krotną stratyfikowaną walidacją krzyżową (`StratifiedKFold`), z monitorowaniem 7 metryk równolegle i wyborem najlepszej kombinacji wg **PR-AUC** (`refit='average_precision'`). Sieć Keras: przeszukanie 4 architektur (ReLU + Dropout, wyjście sigmoid) z `EarlyStopping` na walidacyjnym PR-AUC.

**Dwie strategie obsługi niezbalansowania:**

| Strategia | Modele | Mechanizm |
|---|---|---|
| `class_weight='balanced'` | RandomForest, SVC | większa kara za błędy na klasie mniejszościowej w funkcji straty |
| SMOTE (oversampling) | KNN, MLP, Keras | syntetyczne przykłady klasy „yes" generowane **wyłącznie na foldzie treningowym** (pipeline `imblearn`) — fold walidacyjny i test nietknięte |

**Dobór progu decyzyjnego:** standardowy próg 0.5 jest nieoptymalny przy 89/11. Próg dobrano osobno dla każdego modelu, maksymalizując **F2-score (β = 2)** — czułość ważona 4× silniej niż precyzja, bo w marketingu bankowym koszt pominięcia klienta (FN) przewyższa koszt zbędnego telefonu (FP). Próg dobierany na predykcjach z walidacji krzyżowej / holdoutu (nigdy na teście) i ograniczony z góry do 0.5.

## 6. Wyniki (zbiór testowy: 8 238 obs., 928 klasy „tak")

| Model | Próg | F1-macro | PR-AUC | ROC-AUC | Recall (tak) | Precyzja (tak) | Brier ↓ |
|---|---|---|---|---|---|---|---|
| SVC | 0.147 | **0.702** | 0.434 | 0.784 | 0.501 | **0.452** | **0.082** |
| MLP | 0.500 | 0.695 | 0.457 | 0.799 | 0.634 | 0.388 | 0.150 |
| RandomForest | 0.448 | 0.686 | **0.482** | **0.812** | 0.666 | 0.365 | 0.143 |
| Keras MLP | 0.500 | 0.676 | 0.468 | 0.800 | **0.682** | 0.348 | 0.167 |
| KNN | 0.500 | 0.647 | 0.448 | 0.781 | 0.667 | 0.307 | 0.161 |

![Porównanie modeli na zbiorze testowym](wykresy/analiza_porownawcza_modele.png)

**Najlepsze hiperparametry:**

| Model | Hiperparametry | PR-AUC (CV) |
|---|---|---|
| RandomForest | `n_estimators=350`, `max_depth=10` | 0.455 |
| Keras MLP | architektura 64-32-1, ReLU + Dropout, 33 epoki (EarlyStopping), batch 64 | 0.454 |
| SVC | `kernel='poly'`, `C=0.1`, `gamma='auto'` | 0.440 |
| MLP | `hidden_layer_sizes=(50,)`, `activation='logistic'` | 0.438 |
| KNN | `n_neighbors=71`, `metric='manhattan'`, `weights='uniform'` | 0.417 |

> **Nota:** ze względu na złożoność obliczeniową (~O(n²)–O(n³)) SVC był strojony na 10-procentowej próbce zbioru (`bank-additional.csv`, 4 119 rekordów), a oceniany na tym samym pełnym zbiorze testowym co pozostałe modele. Łączny czas strojenia pozostałych modeli na pełnym zbiorze: ~3 h (Colab; sieć Keras na GPU: ~5 min).

## 7. Wnioski

**Rekomendacja zależy od celu biznesowego:**

- **Ranking i priorytetyzacja listy klientów → RandomForest.** Najlepsze metryki rankingowe niezależne od progu (PR-AUC 0.482, ROC-AUC 0.812), pełna interpretowalność (feature importance), brak wymogu skalowania.
- **Wdrożenie przy stałym progu i ograniczonej przepustowości call center → SVC.** Po doborze progu najwyższy F1-macro (0.702), najmniej zbędnych telefonów (563 FP) i najlepsza kalibracja prawdopodobieństw (Brier 0.082). Słabość: niższa czułość (0.50).
- **Maksymalne pokrycie rynku → Keras MLP lub KNN.** Czułość ~0.67–0.68 (Keras wykrywa 633 z 928 klientów), ale kosztem ponad 1 100 zbędnych telefonów.

**Odpowiedzi na pytania badawcze projektu:**

- *Która funkcja aktywacji wygrywa?* W MLP (sklearn) — **logistyczna (sigmoid)**, nie ReLU; przy płytkiej sieci (50 neuronów) gładka aktywacja sprzyjała generalizacji. W głębszej sieci Keras najlepiej działał ReLU z Dropoutem.
- *Które jądro SVC wygrywa?* **Wielomianowe (`poly`)** z silną regularyzacją (`C=0.1`) — pokonało RBF i liniowe.

**Wnioski metodologiczne:**

- Przy niezbalansowaniu 89/11 **accuracy jest myląca** — kluczowe są PR-AUC i F-score.
- **Usunięcie `duration` obniżyło metryki, ale uczyniło model realistycznym** — świadoma decyzja przeciw wyciekowi danych.
- **Kierunek doboru progu zależy od kalibracji modelu.** Dobrze skalibrowany SVC wymagał obniżenia progu do 0.147 (recall 0.19 → 0.50, +289 trafionych klientów). Modele trenowane SMOTE mają zawyżone prawdopodobieństwa klasy mniejszościowej — ich optymalny próg pozostał na 0.5.
- **Wszystkie modele osiągnęły zbliżony wynik** (ROC-AUC 0.78–0.81), co wskazuje na wyczerpanie sygnału w cechach — wąskie gardło leży w danych, nie w modelach.

## 8. Jak uruchomić

```bash
git clone https://github.com/<twoj-login>/bank-term-deposit-prediction.git
cd bank-term-deposit-prediction

# zależności
pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn joblib

# dane (nie są częścią repo)
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
unzip bank-additional.zip
mkdir -p dane && cp bank-additional/bank-additional-full.csv dane/

jupyter notebook notebook.ipynb
```

Notebook zawiera też sekcję konfiguracji dla **Google Colab** (zalecane — trening sieci Keras korzysta z GPU). Wytrenowane modele zapisują się do `modele/` (joblib `.pkl` / Keras `.keras`) i mogą być wczytane bez ponownego treningu (sekcja 5.7 notebooka).

## 9. Struktura repozytorium

```
.
├── notebook.ipynb        # pełny pipeline: EDA → FE → trening → dobór progu → ewaluacja
├── dane/                 # bank-additional-full.csv (pobierane osobno)
├── modele/               # zapisane modele — generowane przez notebook (nie w repo)
├── wykresy/
│   ├── analiza_numeryczna.png
│   ├── analiza_kategoryczna.png
│   └── analiza_porownawcza_modele.png
└── README.md
```

## Źródło danych

> Moro, S., Cortez, P., Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 62, 22–31.

## Autor

**Marek Karolczuk** — projekt zrealizowany w ramach kursu uczenia maszynowego (SGGW, 2026).
