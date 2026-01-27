# Uczenie Nienadzorowane - Projekt WikiArt Inpainting

Projekt realizowany w ramach przedmiotu Uczenie Nienadzorowane - System uzupełniania uszkodzeń dzieł sztuki z platformy WikiArt.

## 📋 Opis projektu

Celem projektu jest zbudowanie modelu uzupełniającego (inpainting) uszkodzenia dzieł sztuki przy użyciu metod uczenia nienadzorowanego i samonadzorowanego. System składa się z następujących komponentów:

1. **Generator uszkodzeń** - symulacja białych plam/masek na obrazach
2. **Autoenkoder** - budowa zredukowanej reprezentacji obrazów
3. **Klasteryzacja** - grupowanie obrazów według stylu
4. **Model inpainting** - uzupełnianie uszkodzeń dla każdej grupy
5. **Super-resolution** - zwiększanie rozdzielczości obrazów

## 🏗️ Struktura projektu

```
UN/
├── data/                      # Zbiory danych (gitignored)
│   ├── raw/                   # Surowe dane z WikiArt
│   ├── processed/             # Przetworzone dane
│   └── splits/                # Podział train/val/test
├── src/                       # Kod źródłowy
│   ├── encoder/               # Implementacja autoenkodera
│   ├── clustering/            # Klasteryzacja reprezentacji
│   ├── damage_generator/      # Generator uszkodzeń
│   ├── inpainting/            # Modele inpaintingu
│   ├── superres/              # Modele super-resolution
│   └── gui/                   # Proste GUI demo
├── notebooks/                 # Jupyter notebooks
├── models/                    # Zapisane modele (gitignored)
├── configs/                   # Pliki konfiguracyjne
└── requirements.txt           # Zależności projektu
```

## 🚀 Instalacja

1. Utwórz środowisko wirtualne:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

## 📊 Zbiór danych

Projekt wykorzystuje zbiór WikiArt dostępny na Hugging Face:

- **Podstawowy**: `Artificio/WikiArt_Full` (256x256)
- **Wysoka rozdzielczość**: `huggan/wikiart` lub Internet Archive

## 🛠️ Wykorzystywane metody

- Autoenkodery (samonadzorowane budowanie reprezentacji)
- Klasteryzacja (k-means, DBSCAN, Gaussian Mixture, Spectral Clustering)
- UMAP/t-SNE (wizualizacja przestrzeni wysokowymiarowych)
- Neural Inpainting (uzupełnianie uszkodzeń)
- Super-resolution (zwiększanie rozdzielczości)

## 📝 Wymagania techniczne

- Python >= 3.11
- PyTorch >= 2.0.0
- CUDA (opcjonalnie, dla GPU)

## 🎯 Plan rozwoju

### Etap 1 (obecny)

- [x] Struktura projektu
- [x] Generator prostych uszkodzeń (kwadratowe maski)
- [x] Autoenkoder podstawowy

### Etap 2

- [x] Klasteryzacja reprezentacji
- [x] Model inpainting dla prostych uszkodzeń
- [ ] Ocena: 3.0

### Etap 3

- [x] Super-resolution
- [ ] Ocena: 4.0

### Etap 4

- [x] Generator nieregularnych uszkodzeń
- [x] Inpainting dla nieregularnych masek
- [ ] Ocena: 5.0

## ▶️ Szybkie demo

```bash
streamlit run src/gui/app.py
```

Notebook end-to-end: `notebooks/end_to_end_demo.ipynb`.

## 🧪 Trening krok po kroku (Colab, pipeline jak w WikiArt_Inpainting)

Najprościej uruchomić pełny pipeline w Colabie przez notebook `notebooks/colab_training.ipynb`.

1. Otwórz notebook w Colabie i uruchom komórkę **Setup**, aby sklonować repo i zainstalować zależności.
2. Wczytaj dane WikiArt w sekcji **1. Wczytanie danych** (domyślnie `Artificio/WikiArt_Full`).
3. Wytrenuj autoenkoder w sekcji **2. Trening autoenkodera** – to buduje embeddingi.
4. Uruchom **3. Ekstrakcja embeddingów i klasteryzacja** – powstaną etykiety klastrów.
5. Wytrenuj **bazowy model inpainting** w sekcji **4. Trening bazowego modelu inpainting**.
6. (Opcjonalnie) W sekcji **5. Fine-tuning per klaster** ucz osobne impaintery na danych z klastrów.
7. Sprawdź wynik w sekcji **6. Szybki test impaintera**.

Klasteryzacja jest kluczowa – etykiety klastrów służą do przygotowania podzbiorów, na których dopracowujesz modele inpaintingu (tak jak w repozytorium WikiArt_Inpainting).
