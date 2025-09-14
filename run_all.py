import subprocess
import webbrowser
import time
import os

# Chemin vers ton projet et l'environnement (mettre entre guillemets pour les espaces et parenthèses)
project_dir = r'"C:\Users\Lau\Documents\Moi\1-Travail (sept 23)\3- IA\1- Formation Greta\3- Projets\15- OCR\ocr_comparator"'
venv_activate = r'"C:\Users\Lau\Documents\Moi\1-Travail (sept 23)\3- IA\1- Formation Greta\3- Projets\15- OCR\venv\Scripts\activate.bat"'

# Construire la commande complète pour cmd
command = f'start cmd /k "cd /d {project_dir} && call {venv_activate} && uvicorn API_OCR:app --reload --port 8003"'

# Lancer FastAPI dans un nouveau terminal Windows
subprocess.Popen(command, shell=True)

# Attendre quelques secondes pour que FastAPI démarre
time.sleep(5)

# Ouvrir le navigateur sur la documentation FastAPI
webbrowser.open("http://127.0.0.1:8003/docs")

print("FastAPI est lancée et le navigateur ouvert sur /docs")
