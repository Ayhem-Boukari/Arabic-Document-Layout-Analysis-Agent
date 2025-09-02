# 📰 Arabic Document Layout Analysis API (FastAPI + YOLOv8)

Service **FastAPI** exposant des endpoints pour l’analyse de mise en page de documents (journaux, formulaires, PDF convertis en images, etc.) avec un modèle **YOLOv8**.

---

## 📑 Sommaire
- [Aperçu des endpoints](#-aperçu-des-endpoints)
- [Arborescence du projet](#-arborescence-du-projet)
- [Prérequis & installation](#-prérequis--installation)
- [Lancer le serveur](#-lancer-le-serveur)
- [Tester rapidement](#-tester-rapidement)
- [Formats de réponses](#-formats-de-réponses)
- [Configuration & personnalisation](#-configuration--personnalisation)
- [Dépendances & versions](#-dépendances--versions)
- [Dépannage (FAQ)](#-dépannage-faq)
- [Docker (optionnel)](#-docker-optionnel)
- [Bonnes pratiques Git](#-bonnes-pratiques-git)
- [Licence](#-licence)

---

## 🔗 Aperçu des endpoints

### `GET /health`
Retourne un ping et des infos rapides (chemin des poids, classes chargées, version).

### `POST /infer`
- **Entrée :** `multipart/form-data` avec champ `file` (image)
- **Paramètres optionnels :**
  - `imgsz` *(int, défaut 1280)* — taille d’inférence YOLO
  - `iou` *(float, défaut 0.5)* — seuil IoU pour NMS
  - `conf_min` *(float, défaut 0.001)* — confiance minimale avant post-traitement
- **Sortie :** JSON contenant largeur/hauteur, bboxes, classes, scores.

### `POST /infer_image`
Même entrée que `/infer` → renvoie **l’image annotée** (PNG).

### `POST /infer_yolo_txt`
Même entrée que `/infer` → renvoie un **fichier texte YOLO** :

```

cls cx cy w h

````
(coordonnées normalisées ∈ [0,1]).

---

## 📂 Arborescence du projet

```text
repo_root/
├─ app/
│  ├─ __init__.py
│  ├─ main.py            # FastAPI (endpoints /health, /infer, /infer_image, /infer_yolo_txt)
│  ├─ utils.py           # lecture classes YAML + thresholds.json
│  ├─ postprocess.py     # filtres par classe + règles simples layout
│  └─ draw.py            # rendu des boxes (couleur par classe)
├─ weights/
│  └─ best.pt            # <-- vos poids YOLOv8 finaux (~20-30 MB)
├─ data/
│  └─ data.yaml          # <-- classes YOLO (names: [...])
├─ config/
│  └─ thresholds.json    # <-- seuils par classe + exclusions
├─ requirements.txt
└─ README.md
````

---

## 🛠 Prérequis & installation

### 1️⃣ Installer **Python 3.11**

* **Windows :** [python.org](https://www.python.org/downloads/)
* **Linux / macOS :** via `pyenv`, `asdf`, ou package manager.

### 2️⃣ Créer un environnement virtuel (recommandé)

```bash
# Windows (PowerShell)
py -3.11 -m venv venv311
venv311\Scripts\Activate.ps1

# Linux / macOS
python3.11 -m venv venv311
source venv311/bin/activate
```

### 3️⃣ Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Si torch ne s’installe pas automatiquement (CPU) :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4️⃣ Déposer vos fichiers essentiels

* **Poids YOLO :** `weights/best.pt`
* **Classes :** `data/data.yaml`
* **Seuils :** `config/thresholds.json`

---

## 🚀 Lancer le serveur

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --reload
```

* Swagger UI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* ReDoc : [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* Health : [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## ⚡ Tester rapidement

### 1) Ping

```bash
curl http://127.0.0.1:8000/health
```

### 2) JSON de détections

```bash
curl -X POST "http://127.0.0.1:8000/infer" \
  -F "file=@/chemin/vers/mon_image.jpg" \
  -F "imgsz=1280" -F "iou=0.5" -F "conf_min=0.001"
```

### 3) Image annotée (PNG)

```bash
curl -X POST "http://127.0.0.1:8000/infer_image" \
  -F "file=@/chemin/vers/mon_image.jpg" \
  --output result.png
```

### 4) Fichier YOLO .txt

```bash
curl -X POST "http://127.0.0.1:8000/infer_yolo_txt" \
  -F "file=@/chemin/vers/mon_image.jpg"
```

---

## 📊 Formats de réponses

### Exemple `/infer` (JSON)

```json
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
```

---

## ⚙️ Configuration & personnalisation

### 1) data/data.yaml (classes)

```yaml
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
```

### 2) config/thresholds.json (seuils & exclusions)

```json
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
```

### 3) Couleurs (app/draw\.py)

```python
CLASS_COLOR_MAP = {
    "Header": (255, 0, 0),
    "Title": (0, 165, 255),
    "Text": (0, 255, 0),
    ...
}
BOX_THICKNESS = 2
FONT_SCALE = 0.6
```

*(OpenCV utilise BGR et non RGB.)*

---

## 📦 Dépendances & versions

Exemple `requirements.txt` :

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
ultralytics==8.3.175
opencv-python-headless==4.10.0.84
pydantic==2.8.2
python-multipart==0.0.9
PyYAML==6.0.2
numpy==1.26.4
```

---

## 🩺 Dépannage (FAQ)

* **Poids manquants :** vérifier `weights/best.pt`
* **Classe inconnue :** doit exister dans `data/data.yaml`
* **Performance lente (CPU) :** réduire `imgsz` (ex. 960)
* **Erreur OpenCV (GTK/Qt) :** utiliser `opencv-python-headless`
* **CORS :** ouvert par défaut (modifier dans `main.py` si besoin)

---

## 🐳 Docker (optionnel)

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY weights ./weights
COPY data ./data
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & run :

```bash
docker build -t newspaper_yolo_api .
docker run --rm -p 8000:8000 newspaper_yolo_api
```

---

## 🧹 Bonnes pratiques Git

`.gitignore` minimal :

```
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
```

⚠️ **Ne pas committer :**

* environnements virtuels
* datasets bruts
* fichiers > 100 MB (GitHub les refuse)

---

## 📜 Licence

MIT (recommandé).
Ajoute un fichier `LICENSE` si nécessaire.

---

```

Veux-tu que je te génère **un `requirements.txt` complet et figé** (versions exactes de torch incluses pour CPU) pour éviter les incompatibilités lors du déploiement ?
```
