from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import os
import shutil
from Fonctions import (
    batch_process,
    generate_output_filename,
    main_return_texts
)

app = FastAPI()

GROUND_TRUTH_DIR = "uploads/GROUND_TRUTH"
ORIGINAL_DIR = "uploads/Original"
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
os.makedirs(ORIGINAL_DIR, exist_ok=True)

@app.post("/process/")
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
