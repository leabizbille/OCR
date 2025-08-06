
# 🔎 OCR Benchmark Tool: Évaluation Multimoteur (Tesseract, EasyOCR, PaddleOCR)

Ce projet Python permet de comparer automatiquement plusieurs moteurs de reconnaissance optique de caractères (OCR) sur des documents PDF, en évaluant leurs performances via différentes métriques (CRR, CER, WER, F1-score, etc.), avant et après normalisation du texte.

## 📂 Arborescence du projet

ocr_comparator/
├── data/
│ ├── sample1.pdf # Fichiers PDF à analyser
│ └── sample2.pdf
│
├── ground_truths.json # Textes de référence (format JSON ou fichiers .txt individuels)
├── ocr_comparison.py # Script principal contenant tout le pipeline OCR & évaluation
├── results/
│ └── ocr_comparison_results.csv # Résultats agrégés de la comparaison
├── requirements_Sauvegarde.txt # Liste complète des dépendances
└── README.md


  
--- 
## Objectif

L'outil automatise :

- L’extraction d’images depuis un fichier PDF.
- L’application de **trois moteurs OCR** : `Pytesseract`, `EasyOCR`, `PaddleOCR`.
- Le calcul de **métriques d’évaluation textuelles** (brutes et normalisées).
- La **génération d’un rapport Excel** avec les résultats par moteur et par page.
- La **visualisation** des performances (F1-score).
  
---
## Installation & Prérequis

### 1. Création d’un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # ou `venv\\Scripts\\activate` sous Windows

--- 
### 2. Installation des dépendances
Ce projet repose sur les bibliothèques suivantes :
pip install pytesseract easyocr paddleocr pdf2image pandas nltk matplotlib seaborn fuzzywuzzy python-Levenshtein openpyxl editdistance numpy opencv-python

```bash
pip install -r requirements_Sauvegarde.txt
 ---

## 3. Installation de Tesseract OCR

    Télécharger et installer : Tesseract pour Windows (UB Mannheim)
    Par défaut, le chemin est : C:\\Program Files\\Tesseract-OCR


```bash
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

--- 
## 4. Installation de Poppler (pour pdf2image)

Téléchargez les binaires : https://github.com/oschwartz10612/poppler-windows/releases/
Ajouter le dossier poppler/bin au PATH système (important pour Windows)

## ⚙️ Fonctionnalités principales

- 🔤 OCR avec Tesseract, EasyOCR, PaddleOCR.
- 📊 Comparaison automatique de textes OCR à une **vérité terrain** (`ground truth`).
- 📁 Traitement en **batch** de plusieurs fichiers PDF + `.txt`.
- 🧪 Évaluation via : CRR, CER, WRR, WER, Fuzzy Score, BLEU, Precision, Recall, F1-score.
- 📈 Visualisation des F1-scores via `matplotlib`/`seaborn`.

---
```markdown
## Ground Truth (texte de référence)

    Soit au format .txt un fichier par PDF

    Soit un fichier ground_truths.json de type :
    
```bash
{
  "sample1.pdf": "Texte attendu pour sample1...",
  "sample2.pdf": "Texte attendu pour sample2..."
}
```markdown
## Script principal
```bash
python ocr_comparison.py
```markdown
Ce script :

    Charge le texte de référence

    Convertit chaque page PDF en image

    Applique Tesseract, EasyOCR, PaddleOCR

    Calcule les métriques de comparaison

    Exporte les résultats dans un fichier .xlsx ou .csv

    Génère une visualisation comparative

Exemple de configuration dans le script
```markdown
```bash
files = [
    ("data/sample1.txt", "data/sample1.pdf"),
    ("data/sample2.txt", "data/sample2.pdf"),
]
batch_process(files, output_excel='results/ocr_comparison_results.xlsx')

