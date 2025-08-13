# Import des modules pour manipuler les fichiers et images
import os                 # Pour les opérations sur les chemins et dossiers
import cv2                # OpenCV pour traitement d'image (conversion couleurs, etc)
import numpy as np        # Pour les tableaux et matrices numériques
import pytesseract        # Wrapper Python de Tesseract OCR (reconnaissance texte)
import easyocr            # EasyOCR : autre librairie OCR, plus moderne et deep learning
from pdf2image import convert_from_path  # Convertit PDF en images (PIL)
import pandas as pd       # Gestion et analyse des données en DataFrame
import nltk               # Librairie NLP, notamment pour tokenizer (découper texte)
import editdistance       # Calcul distance d'édition (Levenshtein) entre deux séquences
from nltk.tokenize import word_tokenize  # Tokenisation en mots (NLTK)
from fuzzywuzzy import fuzz              # Calcul similarité "floue" entre chaînes (Levenshtein)
from paddleocr import PaddleOCR          # PaddleOCR : OCR basé sur deep learning (plus récent)
import matplotlib.pyplot as plt          # Visualisation graphiques
import seaborn as sns                    # Visualisation avancée (graphique statistique)
from nltk.translate.bleu_score import sentence_bleu  # Score BLEU pour similarité texte
import unicodedata       # Pour normaliser/transformer les caractères (ex : supprimer accents)
import re                # Expressions régulières (traitement texte)
from nltk.metrics.distance import edit_distance  # Calcul distance d'édition Levenshtein
import uuid
import json
from Fonctions import (
# Configuration / Chemins
    TESSERACT_PATH,
# OCR Readers
    easyocr_reader,
# OCR Wrappers
    ocr_with_pytesseract,
    ocr_with_easyocr,
    ocr_with_paddleocr,
    # Extraction & Traitement
    extract_ocr_texts,
    normalize_text,
# Ground Truth & PDF
    load_ground_truth_text,
    load_images_from_file,
# Évaluation & Visualisation
    compute_metrics,
    plot_metrics_from_excel,
# Exécution
    batch_process,
    generate_output_filename
)

# --- Gestion des warnings spécifiques ---
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
# On ignore certains warnings liés à l'utilisation mémoire GPU (utile si GPU activé)

# --- Téléchargement des ressources NLTK ---
nltk.download('punkt')
# --- Configuration du chemin Tesseract OCR ---
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# Point d'entrée pour exécuter le script directement
if __name__ == "__main__":
    files = [
        ("GROUND_TRUTH/Berville_L_CV_IA-avril.txt", "Original/Berville_L_CV_IA-avril.jpg")#,
        #("GROUND_TRUTH/Exemple_ocr.docx", "Original/Exemple_ocr.jpg"),
        #("GROUND_TRUTH/Recommandations et pistes.docx", "Original/Recommandations et pistes image.pdf"),
    ]
    output_file = generate_output_filename()
    batch_process(files, output_excel=output_file)