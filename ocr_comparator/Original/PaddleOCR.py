from paddleocr import PaddleOCR
import cv2

def ocr_to_paragraph(image_path_or_array):
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

texte = ocr_to_paragraph('./Berville_L_CV_IA-avril.jpg')
print(texte)
