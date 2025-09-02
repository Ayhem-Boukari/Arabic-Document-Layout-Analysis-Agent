# Arabic Document Layout Analysis Agent (FastAPI + YOLOv8)

> **But** : service FastAPI qui charge un modèle YOLOv8 (document layout) et renvoie :
> - des **JSON** de détections (`/infer`),
> - une **image annotée** (`/infer_image`),
> - un **fichier YOLO .txt** compatible label (`/infer_yolo_txt`).
>
> **Python requis** : **3.11** (fortement recommandé)

---

## 1) Aperçu rapide

- **Framework API** : FastAPI  
- **Modèle** : Ultralytics YOLOv8, poids dans `weights/best.pt`  
- **Classes** : lues depuis `data/data.yaml`  
- **Seuils/classes à exclure** : `config/thresholds.json`  
- **CORS** ouvert par défaut (pour intégration plateforme)

Endpoints principaux :
- `GET /health` : ping + info classes/poids
- `POST /infer` : JSON (bbox + conf + classes)
- `POST /infer_image` : image PNG annotée
- `POST /infer_yolo_txt` : réponse texte au format YOLO

---

## 2) Structure du projet

```text
repo_root/
├─ app/
│  ├─ __init__.py
│  ├─ main.py            # FastAPI (points d'entrée /health, /infer, /infer_image, /infer_yolo_txt)
│  ├─ utils.py           # lecture classes YAML + thresholds.json
│  ├─ postprocess.py     # filtres par classe + règles simples layout
│  └─ draw.py            # rendu des boxes sur l'image
├─ weights/
│  └─ best.pt            # <-- vos poids YOLOv8 finaux
├─ data/
│  └─ data.yaml          # <-- YAML Ultralytics avec la liste des classes (names: [...])
├─ config/
│  └─ thresholds.json    # <-- seuils par classe + éventuelles classes à exclure
├─ requirements.txt
└─ README.md
