import os
import cv2
import numpy as np
import pytesseract
import easyocr
from pdf2image import convert_from_path
import pandas as pd
import nltk
import editdistance
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import seaborn as sns
import editdistance
from paddleocr import PaddleOCR
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import unicodedata
import re
from nltk.metrics.distance import edit_distance



# ====== Configuration ======
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

nltk.download('punkt')

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Initialisation unique des OCR pour éviter surcharge
easyocr_reader = easyocr.Reader(['fr'], gpu=False)
paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='fr')

# ====== Fonctions OCR ======
def ocr_with_pytesseract(image):
    # image attendue en RGB, conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return pytesseract.image_to_string(gray).strip()

def ocr_with_easyocr(image):
    result = easyocr_reader.readtext(image)
    return " ".join([text[1] for text in result]).strip()

def ocr_with_paddleocr(image):
    import paddleocr
    ocr = paddleocr.PaddleOCR(use_textline_orientation=True, lang='fr')
    result = ocr.predict(image)
    # PaddleOCR renvoie une liste de pages
    if len(result) == 0:
        return []
    page_result = result[0]
    texts = []
    # page_result est une liste de lignes, chaque ligne est une liste où la 2e position (index 1) contient [texte, score]
    for line in page_result:
        text = line[1][0]  # texte reconnu
        texts.append(text)
    return texts

def extract_ocr_texts(image):
    # image au format numpy RGB
    texts = {}
    texts['Pytesseract'] = ocr_with_pytesseract(image)
    texts['EasyOCR'] = ocr_with_easyocr(image)
    texts['PaddleOCR'] = ocr_with_paddleocr(image)
    return texts
import unicodedata
import re

def normalize_text(text: str) -> str:
    if isinstance(text, list):
        text = " ".join(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')  # supprime accents
    text = re.sub(r'\s+', ' ', text)  # supprime espaces multiples
    text = re.sub(r'[^\w\s]', '', text)  # supprime ponctuation
    return text.lower().strip()



def main(ground_truth_txt_path, pdf_path, output_excel='ocr_metrics_report.xlsx'):
    ground_truth = load_ground_truth_text(ground_truth_txt_path)
    images = pdf_to_images(pdf_path)

    results = []
    for i, pil_img in enumerate(images):
        image_rgb = np.array(pil_img)
        print(f"\n--- Page {i + 1} ---")

        ocr_texts = extract_ocr_texts(image_rgb)

        for engine, ocr_text in ocr_texts.items():
            # --- Texte brut ---
            metrics_raw = compute_metrics(ground_truth, ocr_text)
            print(f"{engine} [BRUT] : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")
            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Normalized': False,
                **metrics_raw,
                'OCR_text': ocr_text
            })

            # --- Texte normalisé ---
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
    df.to_excel(output_excel, index=False)
    print(f"\nRésultats exportés vers : {output_excel}")

def compute_metrics(reference: str, predicted: str) -> dict:
    # Gestion des cas où les entrées sont des listes
    if isinstance(reference, list):
        reference = " ".join(reference)
    if isinstance(predicted, list):
        predicted = " ".join(predicted)

    # Mise en minuscules et tokenisation
    reference = reference.lower()
    predicted = predicted.lower()
    reference_words = word_tokenize(reference)
    predicted_words = word_tokenize(predicted)

    # CRR : Character Recognition Rate
    char_matches = sum(1 for a, b in zip(reference, predicted) if a == b)
    crr = char_matches / max(len(reference), 1)

    # CER : Character Error Rate
    cer = edit_distance(reference, predicted) / max(len(reference), 1)

    # WRR : Word Recognition Rate
    word_matches = sum(1 for a, b in zip(reference_words, predicted_words) if a == b)
    wrr = word_matches / max(len(reference_words), 1)

    # WER : Word Error Rate
    wer = edit_distance(reference_words, predicted_words) / max(len(reference_words), 1)

    # Fuzzy ratio (Levenshtein à granularité texte)
    fuzzy_score = fuzz.ratio(reference, predicted)

    # BLEU score (n-grammes)
    bleu_score = sentence_bleu([reference_words], predicted_words) if predicted_words else 0

    # Précision / Rappel / F1 (approche set)
    ref_set = set(reference_words)
    pred_set = set(predicted_words)
    true_positive = len(ref_set & pred_set)
    false_positive = len(pred_set - ref_set)
    false_negative = len(ref_set - pred_set)

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Robustesse : simple version = 1 - CER
    robustness = 1 - cer

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


# Les metriques en graphiques
def plot_ocr_metrics(df, output_folder='plots'):
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metrics_to_plot = ['CRR', 'CER', 'F1-score', 'WER', 'Precision', 'Recall', 'BLEU', 'Robustesse']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='OCR_engine', y=metric, hue='Version', data=df)
        plt.title(f'Distribution de {metric} par OCR Engine et Version')
        plt.ylabel(metric)
        plt.xlabel('OCR Engine')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/{metric}_boxplot.png")
        plt.close()

    print(f"Graphiques sauvegardés dans le dossier '{output_folder}'.")


# ====== Chargement texte ground truth ======
def load_ground_truth_text(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# ====== Conversion PDF -> images ======
def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def main(ground_truth_txt_path, pdf_path, output_excel='ocr_metrics_report.xlsx'):
    ground_truth = load_ground_truth_text(ground_truth_txt_path)
    images = pdf_to_images(pdf_path)

    results = []
    for i, pil_img in enumerate(images):
        image_rgb = np.array(pil_img)
        print(f"\n--- Page {i + 1} ---")

        ocr_texts = extract_ocr_texts(image_rgb)

        for engine, ocr_text in ocr_texts.items():
            # Version brute
            metrics_raw = compute_metrics(ground_truth, ocr_text)
            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Version': 'brute',
                **metrics_raw,
                'OCR_text': ocr_text
            })
            print(f"{engine} (brute) : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")

            # Version corrigée (normalisation simple, simule une correction manuelle)
            corrected_text = normalize_text(ocr_text)  
            metrics_corr = compute_metrics(ground_truth, corrected_text)
            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Version': 'corrigée',
                **metrics_corr,
                'OCR_text': corrected_text
            })
            print(f"{engine} (corrigée) : CRR={metrics_corr['CRR']}, CER={metrics_corr['CER']}, F1={metrics_corr['F1-score']}")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nRésultats exportés vers : {output_excel}")

    # Générer les graphiques
    plot_ocr_metrics(df)

if __name__ == "__main__":
    GROUND_TRUTH_TXT = "Berville_L_CV_IA-avril.txt"
    PDF_FILE = "Berville_L_CV_IA-avril.pdf"
    main(GROUND_TRUTH_TXT, PDF_FILE)
