import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
import nltk
from PIL import Image

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


paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='fr')


def ocr_with_paddleocr(image):
    ocr = PaddleOCR(use_angle_cls=False, lang='fr')
    result = ocr.predict(image)

    if not result:
        return {
            'full_text': '',
            'texts': [],
            'scores': [],
            'bboxes': []
        }

    rec_texts = []
    rec_scores = []
    rec_bboxes = []

    for line in result:
        text = line.get('text', '')
        score = line.get('score', 0.0)
        bbox = line.get('box', [])

        rec_texts.append(text)
        rec_scores.append(score)
        rec_bboxes.append(bbox)

    full_text = ' '.join(rec_texts)

    return {
        'full_text': full_text,
        'texts': rec_texts,
        'scores': rec_scores,
        'bboxes': rec_bboxes
    }



def extract_ocr_texts(image):
    # Extraction OCR
    rec = ocr_with_paddleocr(image)
    # On peut ici choisir de retourner les textes bruts (concaténés)
    full_text = " ".join(rec['texts'])
    # Et éventuellement renvoyer aussi les scores et bboxes dans un dict
    return {
        'full_text': full_text,
        'texts': rec['texts'],
        'scores': rec['scores'],
        'bboxes': rec['bboxes']
    }


def pdf_to_images(file_path):
    print(f"[INFO] Lecture du fichier : {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    print(f"[INFO] Extension du fichier détectée : {ext}")

    if ext == ".pdf":
        print("[INFO] Fichier reconnu comme PDF, conversion en images...")
        images = convert_from_path(file_path)
        print(f"[INFO] {len(images)} page(s) extraite(s) du PDF.")
        return images

    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
        print("[INFO] Fichier reconnu comme image unique.")
        img = Image.open(file_path).convert("RGB")
        print(f"[INFO] Image chargée avec dimensions : {img.size}")
        return [img]  # on retourne une liste pour rester compatible avec le reste du code

    else:
        raise ValueError(f"Type de fichier non supporté : {ext}")

def extract_ocr_texts(image):
    # image au format numpy RGB
    texts = {}
    texts['PaddleOCR'] = ocr_with_paddleocr(image)
    print(texts)
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
    print(text)
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
    longeur = len(reference)

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
        'Robustesse': round(robustness, 3),
        'longeur' :longeur
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


def main(ground_truth_txt_path, pdf_path, output_excel='ocr_metrics_report.xlsx'):
    ground_truth = load_ground_truth_text(ground_truth_txt_path)
    images = pdf_to_images(pdf_path)

    results = []

    for i, pil_img in enumerate(images):
        image_rgb = np.array(pil_img.convert("RGB"))  # ✅ PIL → RGB → numpy array
        print(f"\n--- Page {i + 1} ---")
        ocr_data = extract_ocr_texts(image_rgb)
        for engine, data in ocr_data.items():
            full_text = data['full_text']
            rec_texts = data['texts']
            rec_scores = data['scores']
            rec_bboxes = data['bboxes']

            metrics_raw = compute_metrics(ground_truth, full_text)

            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Version': 'brute',
                **metrics_raw,
                'OCR_text': full_text
            })

            print(f"{engine} (brute) : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")

            corrected_text = normalize_text(full_text)
            metrics_corr = compute_metrics(ground_truth, corrected_text)

            results.append({
                'Page': i + 1,
                'OCR_engine': engine,
                'Version': 'corrigée',
                **metrics_corr,
                'OCR_text': corrected_text
            })

            print(f"{engine} (corrigée) : CRR={metrics_corr['CRR']}, CER={metrics_corr['CER']}, F1={metrics_corr['F1-score']}")

        # Version brute (texte complet concaténé)
        metrics_raw = compute_metrics(ground_truth, full_text)
        results.append({
            'Page': i + 1,
            'OCR_engine': 'PaddleOCR',
            'Version': 'brute',
            **metrics_raw,
            'OCR_text': full_text
        })
        print(f"PaddleOCR (brute) : CRR={metrics_raw['CRR']}, CER={metrics_raw['CER']}, F1={metrics_raw['F1-score']}")

        # Version corrigée (normalisation simple)
        corrected_text = normalize_text(full_text)
        metrics_corr = compute_metrics(ground_truth, corrected_text)
        results.append({
            'Page': i + 1,
            'OCR_engine': 'PaddleOCR',
            'Version': 'corrigée',
            **metrics_corr,
            'OCR_text': corrected_text
        })
        print(f"PaddleOCR (corrigée) : CRR={metrics_corr['CRR']}, CER={metrics_corr['CER']}, F1={metrics_corr['F1-score']}")

        # Optionnel : Tu peux aussi sauvegarder les segments + scores
        # Exemple pour inspection/debug :
        # for t, s in zip(rec_texts, rec_scores):
        #     print(f"Texte: {t} - Score: {s:.3f}")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\nRésultats exportés vers : {output_excel}")

    # Générer les graphiques
    plot_ocr_metrics(df)


if __name__ == "__main__":
    GROUND_TRUTH_TXT = "Berville_L_CV_IA-avril.txt"
    PDF_FILE = "Berville_L_CV_IA-avril.jpg"

    main(GROUND_TRUTH_TXT, PDF_FILE)
