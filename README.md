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
│   └── damage_generator/      # Generator uszkodzeń
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
- [ ] Generator prostych uszkodzeń (kwadratowe maski)
- [ ] Autoenkoder podstawowy

### Etap 2

- [ ] Klasteryzacja reprezentacji
- [ ] Model inpainting dla prostych uszkodzeń
- [ ] Ocena: 3.0

### Etap 3

- [ ] Super-resolution
- [ ] Ocena: 4.0

### Etap 4

- [ ] Generator nieregularnych uszkodzeń
- [ ] Inpainting dla nieregularnych masek
- [ ] Ocena: 5.0
