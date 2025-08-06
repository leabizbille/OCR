from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
import pandas as pd
import json
import unicodedata
from jiwer import wer
import Levenshtein
import pandas as pd


img_path = r"C:\Users\Lau\Documents\Moi\1-Travail (sept 23)\3- IA\1- Formation Greta\3- Projets\15- OCR\ocr_comparator\data\Berville_L_CV_IA-avril.jpg"

# Initialize PaddleOCR instance
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Run OCR inference on a sample image 
result = ocr.predict(
    input=img_path)

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")

# Imaginons res dict (extrait simplifié)
rec_texts = res['rec_texts']
rec_scores = res['rec_scores']

# rec_boxes est un array numpy de shape (N, 4) par exemple, on convertit chaque ligne en tuple
rec_boxes = [tuple(box) for box in res['rec_boxes']]

# Construction DataFrame
df_ocr = pd.DataFrame({
    'text': rec_texts,
    'score': rec_scores,
    'box': rec_boxes
})
#--------------------------------------------------
full_text = " ".join(df_ocr['text'].astype(str))

# === 2. Charger le texte ground truth ===
with open("Berville_L_CV_IA-avril.txt", encoding="utf-8") as f:
    ground_truth = f.read()
# === 3. Nettoyage simple (optionnel mais recommandé) ===
def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])  # enlever accents
    return text.lower().strip()

ocr_text = normalize_text(full_text)
ground_truth = normalize_text(ground_truth)

# === 4. Calcul des métriques ===s

# CER (Character Error Rate)
cer = Levenshtein.distance(ocr_text, ground_truth) / len(ground_truth)

# WER (Word Error Rate)
word_error = wer(ground_truth, ocr_text)

# === 5. Résultats ===
print(f"\n--- Évaluation OCR ---")
print(f"CER (Character Error Rate) : {cer:.2%}")
print(f"WER (Word Error Rate)      : {word_error:.2%}")
print(f"Levenshtein.distance : {Levenshtein.distance(ocr_text, ground_truth)}")
print(f"Longeur : {len(ground_truth)}")
