from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
import numpy as np
import shutil
from Fonctions import (
    batch_process,
    generate_output_filename,
    main_return_texts
)

app = FastAPI(
    title="OCR Processing API",
    description="API pour traiter et comparer des fichiers OCR avec des fichiers ground truth.",
    version="1.0.0"
)

GROUND_TRUTH_DIR = "uploads/GROUND_TRUTH"
ORIGINAL_DIR = "uploads/Original"
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)

@app.post("/process/", summary="Extraire le texte OCR",
          description="Extrait le texte OCR à partir de fichiers PDF et les compare à des fichiers ground truth (texte brut).")
async def process_ocr(
    ground_truth_files: List[UploadFile] = File(...,description="Fichiers ground truth "),
    ocr_files: List[UploadFile] = File(..., description="Fichiers OCR à analyser, généralement des PDF")
):
    """
    Cette route permet d'extraire les textes OCR et de les renvoyer sous forme de texte brut.
    Elle attend deux listes de fichiers : les ground truth et les fichiers OCR à comparer.
    """
    if len(ground_truth_files) != len(ocr_files):
        raise HTTPException(status_code=400, detail="Le nombre de fichiers ground truth doit être égal au nombre de fichiers OCR.")

    results = []

    # Sauvegarde des fichiers
    for gt_file in ground_truth_files:
        gt_path = os.path.join(GROUND_TRUTH_DIR, gt_file.filename)
        with open(gt_path, 'wb') as f:
            shutil.copyfileobj(gt_file.file, f)
    
    for ocr_file in ocr_files:
        ocr_path = os.path.join(ORIGINAL_DIR, ocr_file.filename)
        with open(ocr_path, 'wb') as f:
            shutil.copyfileobj(ocr_file.file, f)

    # Traitement par paires
    for gt_file, ocr_file in zip(ground_truth_files, ocr_files):
        gt_path = os.path.join(GROUND_TRUTH_DIR, gt_file.filename)
        ocr_path = os.path.join(ORIGINAL_DIR, ocr_file.filename)

        # Appel à la fonction principale de traitement OCR
        df = main_return_texts(ground_truth_txt_path=gt_path, pdf_path=ocr_path)

        # Vérification et extraction texte
        if 'OCR_text' not in df.columns:
            raise HTTPException(status_code=500, detail=f"Colonne 'OCR_text' absente dans le résultat pour {ocr_file.filename}")

        full_text = " ".join(df['OCR_text'].tolist())

        results.append({
            "ground_truth_file": gt_file.filename,
            "ocr_file": ocr_file.filename,
            "extracted_text": full_text
        })

    return {"results": results}


@app.post("/metrics/", summary="Calculer les métriques OCR",
          description="Calcule les métriques (par ex. distance, normalisation) entre les fichiers ground truth et OCR, et retourne aussi le moteur OCR utilisé.")
async def process_ocr(
    ground_truth_files: List[UploadFile] = File(...),
    ocr_files: List[UploadFile] = File(...)
):
    if len(ground_truth_files) != len(ocr_files):
        raise HTTPException(status_code=400, detail="Le nombre de fichiers ground truth doit être égal au nombre de fichiers OCR.")

    results = []

    # Sauvegarde des fichiers
    for gt_file in ground_truth_files:
        gt_path = os.path.join(GROUND_TRUTH_DIR, gt_file.filename)
        with open(gt_path, 'wb') as f:
            shutil.copyfileobj(gt_file.file, f)
    
    for ocr_file in ocr_files:
        ocr_path = os.path.join(ORIGINAL_DIR, ocr_file.filename)
        with open(ocr_path, 'wb') as f:
            shutil.copyfileobj(ocr_file.file, f)

    # Traitement par paires
    for gt_file, ocr_file in zip(ground_truth_files, ocr_files):
        gt_path = os.path.join(GROUND_TRUTH_DIR, gt_file.filename)
        ocr_path = os.path.join(ORIGINAL_DIR, ocr_file.filename)

        # Appel à la fonction principale de traitement OCR
        df = main_return_texts(ground_truth_txt_path=gt_path, pdf_path=ocr_path)

        # Sécurité sur les colonnes
        if 'metrics_norm' in df.columns:
            metrics_norm = " ".join(df['metrics_norm'].astype(str).tolist())
        else:
            metrics_norm = ""
            print(f"⚠️ La colonne 'metrics_norm' est absente pour {ocr_file.filename}")

        if 'OCR_text' not in df.columns:
            raise HTTPException(status_code=500, detail=f"Colonne 'OCR_text' absente dans le résultat pour {ocr_file.filename}")

        if 'OCR_engine' not in df.columns:
            raise HTTPException(status_code=500, detail=f"Colonne 'OCR_engine' absente dans le résultat pour {ocr_file.filename}")

        full_text = " ".join(df['OCR_text'].astype(str).tolist())
        engine = " ".join(df['OCR_engine'].astype(str).tolist())

        results.append({
            "ground_truth_file": gt_file.filename,
            "ocr_file": ocr_file.filename,
            "extracted_text": full_text,
            "engine": engine
        })

    return {"results": results}
