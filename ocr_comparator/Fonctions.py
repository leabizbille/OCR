# Import des modules pour manipuler les fichiers et images
import os                 # Pour les opérations sur les chemins et dossiers
import cv2                # OpenCV pour traitement d'image (conversion couleurs, etc)
import numpy as np        # Pour les tableaux et matrices numériques
import pytesseract        # Wrapper Python de Tesseract OCR (reconnaissance texte)
import easyocr            # EasyOCR : autre librairie OCR, plus moderne et deep learning
from pdf2image import convert_from_path  # Convertit PDF en images (PIL)
import pandas as pd       # Gestion et analyse des données en DataFrame
import nltk               # Librairie NLP, notamment pour tokenizer (découper texte)
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
from pdf2image import convert_from_path
from PIL import Image
import time

# --- Gestion des warnings spécifiques ---
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
# On ignore certains warnings liés à l'utilisation mémoire GPU

# --- Téléchargement des ressources NLTK ---
nltk.download('punkt')
# On télécharge la ressource 'punkt' pour tokenizer les textes en mots

# --- Configuration du chemin Tesseract OCR ---
# Tesseract nécessite d'indiquer où est l'exécutable (Windows ici)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# --- Initialisation des moteurs OCR pour éviter de réinitialiser à chaque appel ---
# EasyOCR configuré pour langue française et sans GPU (CPU uniquement)
easyocr_reader = easyocr.Reader(['fr'], gpu=False)

# === Définition des fonctions ===
def ocr_with_pytesseract(image):
    """
    Applique Tesseract OCR sur une image donnée.
    - Convertit en niveaux de gris (améliore précision OCR)
    - Retourne la chaîne de texte reconnue
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Conversion RGB -> gris
    return pytesseract.image_to_string(gray).strip()  # OCR puis nettoyage espaces

def ocr_with_paddleocr(image_path_or_array):
    """
    Extrait le texte d'une image via PaddleOCR et renvoie un paragraphe concatené.

    Args:
        image_path_or_array (str ou numpy.ndarray): chemin vers l'image ou image déjà chargée (BGR).

    Returns:
        str: texte extrait sous forme de paragraphe.
    """
    ocr = PaddleOCR(lang='fr', use_textline_orientation=False)

    # Chargement de l'image si un chemin est donné
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
        if image is None:
            raise ValueError(f"Impossible de lire l'image au chemin : {image_path_or_array}")
    else:
        image = image_path_or_array

    result = ocr.predict(image)
    data = result[0]
    texts = data['rec_texts']

    texte_paragraphe = " ".join(texts)
    return texte_paragraphe

def ocr_with_easyocr(image):
    """
    Utilise EasyOCR pour extraire texte de l'image.
    Retourne la concaténation des textes reconnus sur toutes les zones détectées.
    """
    result = easyocr_reader.readtext(image)  # Reconnaissance texte EasyOCR
    # result = liste de tuples (box, texte, confiance)
    return " ".join([text[1] for text in result]).strip()
 
def extract_ocr_texts(image):
    """
    Applique les 3 OCR sur la même image.
    Renvoie un dict avec clé = moteur OCR, valeur = texte reconnu.
    """
    texts = {}
    texts['Pytesseract'] = ocr_with_pytesseract(image)  # Tesseract
    texts['EasyOCR'] = ocr_with_easyocr(image)          # EasyOCR
    texts['PaddleOCR'] = ocr_with_paddleocr(image) or ""
    return texts

def normalize_text(text: str) -> str:
    """
    Normalise un texte pour faciliter les comparaisons :
    - Supprime les accents
    - Supprime la ponctuation
    - Met en minuscules
    - Supprime espaces multiples
    Si le texte est une liste, on concatène en chaîne avant traitement.
    """
    if isinstance(text, list):
        text = " ".join(text)  # liste -> chaîne
    text = unicodedata.normalize('NFKD', text)  # décompose les accents
    text = text.encode('ascii', 'ignore').decode('utf-8')  # supprime accents
    text = re.sub(r'\s+', ' ', text)  # supprime espaces multiples
    text = re.sub(r'[^\w\s]', '', text)  # supprime ponctuation
    return text.lower().strip()  # minuscule et supprime espaces début/fin

def compute_metrics(reference: str, predicted: str) -> dict:
    """
    Calcule plusieurs métriques de comparaison texte entre référence et prédiction :
    - CRR (Character Recognition Rate) : % caractères corrects
    - CER (Character Error Rate) : taux d'erreur caractères (distance de Levenshtein normalisée)
    - WRR (Word Recognition Rate) : % mots corrects
    - WER (Word Error Rate) : taux d'erreur mots
    - Fuzzy : score de similarité textuelle (0-100)
    - BLEU : score BLEU n-grammes (0-1)
    - Precision / Recall / F1 : métriques basées sur ensemble de mots
    - Robustesse : 1 - CER (indicateur de qualité)
    """
    # Gérer le cas où les entrées sont des listes
    if isinstance(reference, list):
        reference = " ".join(reference)
    if isinstance(predicted, list):
        predicted = " ".join(predicted)

    # Mise en minuscules pour homogénéité
    reference = reference.lower()
    predicted = predicted.lower()

    # Tokenisation en mots
    reference_words = word_tokenize(reference)
    predicted_words = word_tokenize(predicted)

    # Calcul CRR : proportion de caractères identiques à la même position
    char_matches = sum(1 for a, b in zip(reference, predicted) if a == b)
    crr = char_matches / max(len(reference), 1)

    # Calcul CER : distance d'édition caractères normalisée
    cer = edit_distance(reference, predicted) / max(len(reference), 1)

    # Calcul WRR : proportion de mots identiques à la même position
    word_matches = sum(1 for a, b in zip(reference_words, predicted_words) if a == b)
    wrr = word_matches / max(len(reference_words), 1)

    # Calcul WER : distance d'édition sur mots normalisée
    wer = edit_distance(reference_words, predicted_words) / max(len(reference_words), 1)

    # Calcul similarité fuzzy (Levenshtein ratio, 0-100)
    fuzzy_score = fuzz.ratio(reference, predicted)

    # Calcul BLEU (score de similarité n-grammes)
    bleu_score = sentence_bleu([reference_words], predicted_words) if predicted_words else 0

    # Calcul Precision, Recall, F1 à partir des ensembles de mots (ignore doublons)
    ref_set = set(reference_words)
    pred_set = set(predicted_words)
    true_positive = len(ref_set & pred_set)  # mots en commun
    false_positive = len(pred_set - ref_set) # mots faux détectés
    false_negative = len(ref_set - pred_set) # mots manquants

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Robustesse = 1 - taux d'erreur caractères
    robustness = 1 - cer

    # Retourne un dictionnaire avec toutes les métriques arrondies
    return {
        'CRR': round(crr, 3),
        'CER': round(cer, 3),
        'WRR': round(wrr, 3),
        'WER': round(wer, 3),
        'Fuzzy': round(fuzzy_score, 2),
        'BLEU': round(bleu_score, 3),
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'F1-score': round(f1, 3),
        'Robustesse': round(robustness, 3)
    }

def plot_metrics_from_excel(file):
    df = pd.read_excel(file)
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x='OCR_engine', y='F1-score', hue='Normalized')
    plt.title("F1-score par moteur OCR (normalisé vs brut)")
    plt.show()

import os
from docx import Document

def load_ground_truth_text(file_path):
    # Ajouter automatiquement le dossier si nécessaire
    if not os.path.isabs(file_path) and not os.path.exists(file_path):
        file_path = os.path.join("résultats", file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Fichier introuvable : {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.docx':
        # Lecture avec python-docx
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text).strip()
        except Exception as e:
            raise RuntimeError(f"❌ Échec de lecture du fichier DOCX {file_path} : {e}")

    else:
        # Lecture classique pour fichiers texte
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            print(f"⚠️ Encodage UTF-8 échoué pour {file_path}, tentative en 'latin-1'")
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                raise RuntimeError(f"❌ Échec de lecture du fichier {file_path} avec 'latin-1' : {e}")


def batch_process(files_list, output_excel=None):
    """
    files_list : liste de tuples (chemin_ground_truth_txt, chemin_pdf)
    Retourne le DataFrame combiné et exporte tout en Excel.
    """
    all_results = []
    for gt_path, pdf_path in files_list:
        print(f"Traitement du couple:\n- Ground truth: {gt_path}\n- PDF: {pdf_path}")
        df = main(gt_path, pdf_path, output_excel=None)  # Pas d'export dans main
        df['Source_GT'] = os.path.basename(gt_path)     # Pour tracer la source plus tard
        df['Source_PDF'] = os.path.basename(pdf_path)
        all_results.append(df)
    
    if output_excel is None:
        output_excel = generate_output_filename()  # <-- indentation corrigée
    
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_excel(output_excel, index=False)
    print(f"\nTous les résultats combinés exportés vers : {output_excel}")
    return combined_df


def load_images_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        # Convertir PDF en liste d'images PIL
        return convert_from_path(file_path)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # Charger directement une image (retourner liste avec 1 élément)
        return [Image.open(file_path)]
    else:
        raise ValueError(f"Format de fichier non supporté : {file_path}")


def generate_output_filename(prefix="resultats_ocr_", extension=".xlsx", folder="data"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}{timestamp}{extension}"
    return os.path.join(folder, filename)

def main(ground_truth_txt_path, pdf_path, output_excel):
    """
    Fonction principale exécutant le pipeline complet :
    - Chargement du texte référence
    - Conversion PDF en images (pages)
    - Application des 3 moteurs OCR par page
    - Calcul métriques brutes et normalisées
    - Stockage des résultats dans un fichier Excel
    """
    ground_truth = load_ground_truth_text(ground_truth_txt_path)
    images = load_images_from_file(pdf_path)

    results = []  # Liste pour stocker tous les résultats

    # Itération sur chaque page (image) du PDF
    for i, pil_img in enumerate(images):
        image_rgb = np.array(pil_img)  # Conversion PIL -> numpy array RGB
        print(f"\n--- Page {i + 1} ---")

        # Application des 3 OCR sur l'image
        ocr_texts = extract_ocr_texts(image_rgb)

        for engine, ocr_text in ocr_texts.items():
            # Calcul métriques sur texte brut non normalisé
            metrics_raw = compute_metrics(ground_truth, ocr_text)
            print(f"{engine} [BRUT] : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")

            # Stockage dans liste résultats
            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Normalized': False,
                **metrics_raw,
                'OCR_text': ocr_text
            })

            # Calcul métriques sur texte normalisé
            normalized_gt = normalize_text(ground_truth)
            normalized_pred = normalize_text(ocr_text)
            metrics_norm = compute_metrics(normalized_gt, normalized_pred)
            print(f"{engine} [NORMALISE] : CRR={metrics_norm['CRR']}, CER={metrics_norm['CER']}, F1={metrics_norm['F1-score']}")

            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Normalized': True,
                **metrics_norm,
                'OCR_text': normalized_pred
            })
    df = pd.DataFrame(results)
    if output_excel:  # Export seulement si output_excel est non None
        df.to_excel(output_excel, index=False)
        print(f"\nRésultats exportés vers : {output_excel}")
        if output_excel is not None:
            plot_metrics_from_excel(output_excel)
    return df


# Main pour l api
def main_return_texts(ground_truth_txt_path, pdf_path):
    """
    Exécute le pipeline OCR et renvoie les résultats sous forme de DataFrame
    (sans export Excel ni affichage graphique).
    """
    ground_truth = load_ground_truth_text(ground_truth_txt_path)
    images = load_images_from_file(pdf_path)

    results = []

    for i, pil_img in enumerate(images):
        image_rgb = np.array(pil_img)
        print(f"\n--- Page {i + 1} ---")

        ocr_texts = extract_ocr_texts(image_rgb)

        for engine, ocr_text in ocr_texts.items():
            metrics_raw = compute_metrics(ground_truth, ocr_text)
            print(f"{engine} [BRUT] : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")

            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Normalized': False,
                **metrics_raw,
                'OCR_text': ocr_text
            })

            normalized_gt = normalize_text(ground_truth)
            normalized_pred = normalize_text(ocr_text)
            metrics_norm = compute_metrics(normalized_gt, normalized_pred)
            print(f"{engine} [NORMALISE] : CRR={metrics_norm['CRR']}, CER={metrics_norm['CER']}, F1={metrics_norm['F1-score']}")

            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Normalized': True,
                **metrics_norm,
                'OCR_text': normalized_pred
            })

    return pd.DataFrame(results)
