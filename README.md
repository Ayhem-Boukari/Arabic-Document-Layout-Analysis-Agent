Arabic Document Layout Analysis Agent (FastAPI + YOLOv8)

Service FastAPI qui charge un modèle YOLOv8 (analyse de mise en page de documents, journaux, etc.) et expose 3 endpoints :

POST /infer → renvoie un JSON de détections

POST /infer_image → renvoie une image annotée (PNG)

POST /infer_yolo_txt → renvoie un fichier texte YOLO (format label)

✅ Version Python requise : 3.11 (recommandé)
✅ Poids YOLO attendus dans weights/best.pt
✅ Classes lues depuis data/data.yaml
✅ Seuils/filtres dans config/thresholds.json
✅ CORS ouvert (intégration plateforme type DataUP)

Sommaire

Aperçu des endpoints

Arborescence du projet

Prérequis & installation (Python 3.11)

Lancer le serveur (Uvicorn)

Tester rapidement

Formats de réponses (exemples)

Configuration & personnalisation

Dépendances & versions

Dépannage (FAQ)

Docker (optionnel)

Bonnes pratiques Git

Aperçu des endpoints

GET /health
Ping + infos rapides : classes, chemin des poids, version.

POST /infer
Entrée : multipart/form-data avec champ file (image).
Paramètres optionnels (form-data) :

imgsz (int, défaut 1280) : taille d’inférence YOLO

iou (float, défaut 0.5) : seuil IoU pour NMS

conf_min (float, défaut 0.001) : seuil de confiance minimal avant post-traitement
Sortie : JSON (bbox, scores, classes).

POST /infer_image
Même entrée que /infer, renvoie l’image annotée (PNG) avec les boxes en couleurs par classe.

POST /infer_yolo_txt
Même entrée que /infer, renvoie un texte au format label YOLO :
cls cx cy w h (coordonnées normalisées au range [0,1]).

L’app applique ensuite des filtres de confiance par classe + quelques règles “layout” simples (par ex. minimiser les doublons inutiles).

Arborescence du projet
repo_root/
├─ app/
│  ├─ __init__.py
│  ├─ main.py            # FastAPI (endpoints /health, /infer, /infer_image, /infer_yolo_txt)
│  ├─ utils.py           # lecture classes YAML + thresholds.json
│  ├─ postprocess.py     # filtres par classe + règles simples layout
│  └─ draw.py            # rendu des boxes (couleur par classe, épaisseur, labels)
├─ weights/
│  └─ best.pt            # <-- vos poids YOLOv8 finaux (≈ 20–30 MB)
├─ data/
│  └─ data.yaml          # <-- YAML Ultralytics avec la liste des classes (names: [...])
├─ config/
│  └─ thresholds.json    # <-- seuils par classe + éventuelles classes à exclure
├─ requirements.txt
└─ README.md

Prérequis & installation (Python 3.11)

Installer Python 3.11

Windows : python.org

Linux/Mac : via pyenv, asdf, ou gestionnaire de paquets.

Créer un environnement virtuel (recommandé)

Windows (PowerShell) :

py -3.11 -m venv venv311
venv311\Scripts\Activate.ps1


Linux / macOS :

python3.11 -m venv venv311
source venv311/bin/activate


Installer les dépendances

pip install --upgrade pip
pip install -r requirements.txt


Si torch échoue à s’installer automatiquement, essaye (CPU) :

pip install torch --index-url https://download.pytorch.org/whl/cpu


Déposer vos fichiers essentiels

Poids : weights/best.pt

Classes : data/data.yaml (contient names: [...])

Seuils : config/thresholds.json

Lancer le serveur (Uvicorn)

Depuis la racine du projet (dans le venv activé) :

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --reload


Swagger UI : http://127.0.0.1:8000/docs

ReDoc : http://127.0.0.1:8000/redoc

Health : http://127.0.0.1:8000/health

--reload redémarre automatiquement à chaque modification du code (utile en dev).

Tester rapidement
1) Ping & infos
curl http://127.0.0.1:8000/health

2) JSON de détections
curl -X POST "http://127.0.0.1:8000/infer" \
  -F "file=@/chemin/vers/mon_image.jpg" \
  -F "imgsz=1280" -F "iou=0.5" -F "conf_min=0.001"

3) Image annotée (PNG)
curl -X POST "http://127.0.0.1:8000/infer_image" \
  -F "file=@/chemin/vers/mon_image.jpg" \
  --output result.png

4) Fichier YOLO .txt
curl -X POST "http://127.0.0.1:8000/infer_yolo_txt" \
  -F "file=@/chemin/vers/mon_image.jpg"


Via Swagger UI (/docs), tu peux tester en uploadant un fichier directement dans l’interface.

Formats de réponses (exemples)
/infer (JSON)
{
  "width": 1654,
  "height": 2338,
  "detections": [
    {
      "cls_name": "Title",
      "cls_id": 1,
      "conf": 0.73,
      "x1": 120, "y1": 65, "x2": 1540, "y2": 210,
      "cx": 0.50, "cy": 0.06, "w": 0.86, "h": 0.06
    }
  ]
}


x1,y1,x2,y2 : coordonnées pixels (coin haut-gauche → bas-droit).

cx,cy,w,h : coordonnées normalisées (format YOLO, centre/largeur/hauteur ∈ [0,1]).

/infer_yolo_txt
1 0.500000 0.060000 0.860000 0.060000

Configuration & personnalisation
1) data/data.yaml (classes)

Exemple minimal :

names:
  - Header
  - Title
  - Text
  - Table
  - Image
  - Footer
  - Stamp or Signature
  - Caption
  - Keyvalue
  - List-item
  - Check-box

2) config/thresholds.json (seuils & exclusions)

Exemple :

{
  "per_class_conf": {
    "Header": 0.25,
    "Title": 0.30,
    "Text": 0.35,
    "Table": 0.30,
    "Image": 0.30,
    "Footer": 0.25,
    "Stamp or Signature": 0.30,
    "Caption": 0.25,
    "Keyvalue": 0.25,
    "List-item": 0.25,
    "Check-box": 0.15
  },
  "exclude": []
}


per_class_conf : seuil de confiance après prédiction (post-filtre).

exclude : liste de noms de classes à ignorer totalement.

3) Couleurs des boxes (par classe)

Dans app/draw.py, la map couleur est définie (ex CLASS_COLOR_MAP).
Tu peux ajuster couleur/épaisseur/label. Exemple :

CLASS_COLOR_MAP = {
    "Header": (255, 0, 0),
    "Title": (0, 165, 255),
    "Text": (0, 255, 0),
    ...
}
BOX_THICKNESS = 2
FONT_SCALE = 0.6


OpenCV utilise BGR (et non RGB).

Dépendances & versions

Dans requirements.txt (ex. recommandé) :

fastapi==0.115.0
uvicorn[standard]==0.30.6
ultralytics==8.3.175
opencv-python-headless==4.10.0.84
pydantic==2.8.2
python-multipart==0.0.9
PyYAML==6.0.2
numpy==1.26.4


ultralytics installera torch automatiquement.
Si souci, installe torch CPU manuellement :

pip install torch --index-url https://download.pytorch.org/whl/cpu

Dépannage (FAQ)

Erreur poids manquants : vérifie que weights/best.pt existe et correspond bien à ton modèle.

Classe inconnue : la classe prédite doit exister dans data/data.yaml (names: [...]).

Performance lente (CPU) : c’est normal sur des images 1280px et YOLOv8. Réduis imgsz (ex. 960) pour aller plus vite.

Erreur OpenCV (GTK/Qt) sur serveur : utilise opencv-python-headless.

CORS : est déjà permissif. Pour restreindre, ajuste allow_origins dans app/main.py.

Docker (optionnel)

Un Dockerfile CPU minimal :

FROM python:3.11-slim

# Dépendances système utiles (libgl pour OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie d'abord requirements pour profiter du cache build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copie du code et des assets
COPY app ./app
COPY weights ./weights
COPY data ./data
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


Build & run :

docker build -t newspaper_yolo_api .
docker run --rm -p 8000:8000 newspaper_yolo_api

Bonnes pratiques Git

Ne pas commiter d’environnements virtuels (venv/, .venv/, venv311/).

Ne pas commiter de gros fichiers/lots de données (datasets bruts, caches).

Ajouter un .gitignore :

__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
*.log

.venv/
venv/
venv311/

runs/
*.cache
.DS_Store
Thumbs.db
.vscode/
.idea/
unified_dataset/


⚠️ GitHub refuse tout fichier > 100 MB (ex. DLLs de PyTorch si tu commits le venv).
Tes poids best.pt ≈ 20–30 MB sont OK.

Licence

Au choix (MIT recommandée). Ajoute un LICENSE si besoin.
