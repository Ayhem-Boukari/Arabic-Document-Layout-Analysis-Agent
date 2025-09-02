import io
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    PlainTextResponse,
    HTMLResponse,
)
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel
from ultralytics import YOLO

from .utils import load_class_names, load_thresholds
from .postprocess import filter_by_class_conf, simple_layout_rules
from .draw import draw_detections

# -------------------------------------------------
# Chemins & constantes
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "best.pt"
DATA_YAML = ROOT / "data" / "data.yaml"
THRESH = ROOT / "config" / "thresholds.json"

TAGS_METADATA = [
    {
        "name": "Health",
        "description": "Vérification du service et métadonnées du modèle.",
    },
    {
        "name": "Inference",
        "description": "Exécuter l’inférence YOLO et obtenir des résultats en **JSON**, **PNG annoté** ou **TXT (format YOLO)**.",
    },
]

LONG_DESCRIPTION = """
**Objectif**  
Pré-annoter automatiquement des images de *journaux/documents arabes ou bilingues* avec leurs **zones de mise en page** (Title, Text, Image, Table, Footer, etc.), afin d’accélérer l’annotation manuelle dans DATAUP.

**Entrées**  
- **Image**: PNG/JPG (champ `file`)  
- **Paramètres optionnels**:  
  - `imgsz` *(int, défaut 1280)*: taille d’inférence (plus grand = meilleurs petits objets, plus lent)  
  - `iou` *(float, défaut 0.5)*: NMS IoU  
  - `conf_min` *(float, défaut 0.001)*: seuil de confiance minimal global (les seuils spécifiques par classe sont appliqués ensuite)

**Sorties**  
- **`POST /infer`** → **JSON** avec la taille de l’image et la liste des détections :  
  `cls_id`, `cls_name`, `conf`, `x1,y1,x2,y2` (pixels), `cx,cy,w,h` (normalisés YOLO)  
- **`POST /infer_image`** → **PNG** de l’image **annotée** (boîtes + libellés)  
- **`POST /infer_yolo_txt`** → fichier texte **compatible YOLO** : `cls cx cy w h` par ligne (normalisé)

**Notes**  
- Les **seuils par classe** et **règles de post-traitement** améliorent la qualité (moins de bruit, moins de confusions).  
- Le modèle est chargé **sur CPU** par défaut (peut être plus lent).  
"""

# -------------------------------------------------
# Application FastAPI (docs custom)
# -------------------------------------------------
app = FastAPI(
    title="Arabic Document Layout Analysis Agent",
    version="1.0.0",
    description=LONG_DESCRIPTION,
    openapi_url="/openapi.json",
    docs_url=None,     # on remplace /docs par une version custom ci-dessous
    redoc_url=None,    # idem pour /redoc
    openapi_tags=TAGS_METADATA,
    contact={"name": "DATAUP Agent", "email": "support@example.com"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    terms_of_service="https://example.com/terms",  # optionnel
)

# CORS (plateforme DATAUP, front, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre si besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Chargement modèle & classes
# -------------------------------------------------
CLASS_NAMES = load_class_names(str(DATA_YAML))
CLASS_CONF_LIST, EXCLUDE_SET = load_thresholds(str(THRESH), CLASS_NAMES)
MODEL = YOLO(str(WEIGHTS))  # CPU par défaut

# -------------------------------------------------
# Modèles Pydantic (schémas)
# -------------------------------------------------
class Detection(BaseModel):
    cls_name: str
    cls_id: int
    conf: float
    x1: int; y1: int; x2: int; y2: int
    cx: float; cy: float; w: float; h: float

class PredictResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection]

# -------------------------------------------------
# Utilitaire conversion YOLO -> dicts
# -------------------------------------------------
def _ultra_to_dicts(result, img_w, img_h):
    dets = []
    b = result.boxes
    if b is None or len(b) == 0:
        return dets
    xyxy = b.xyxy.cpu().numpy()
    conf = b.conf.cpu().numpy()
    cls  = b.cls.cpu().numpy().astype(int)
    for i in range(xyxy.shape[0]):
        x1,y1,x2,y2 = xyxy[i]
        w = (x2-x1)/img_w
        h = (y2-y1)/img_h
        cx = (x1+x2)/(2*img_w)
        cy = (y1+y2)/(2*img_h)
        idx = int(cls[i])
        dets.append({
            "cls_id": idx,
            "cls_name": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"cls_{idx}",
            "conf": float(conf[i]),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "xywhn": [float(cx), float(cy), float(w), float(h)],
        })
    return dets

# -------------------------------------------------
# Page d’accueil simple et lisible
# -------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    return HTMLResponse(
        f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Arabic Document Layout Analysis Agent</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg:#0b1020; --card:#131a2b; --text:#e9eefb; --muted:#b8c2e0; --accent:#5aa9ff;
    }}
    body {{ background:var(--bg); color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; margin:0; }}
    .wrap {{ max-width:880px; margin:64px auto; padding:0 20px; }}
    .card {{
      background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      border:1px solid rgba(255,255,255,0.09);
      border-radius:20px; padding:28px 28px;
      box-shadow:0 12px 30px rgba(0,0,0,0.35);
    }}
    h1 {{ margin:0 0 12px 0; font-size:28px; letter-spacing:.2px; }}
    p  {{ color:var(--muted); line-height:1.6; }}
    code, pre {{ background:#0d1326; color:#dfe7ff; padding:2px 6px; border-radius:6px; }}
    .row {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:18px; }}
    a.btn {{
      display:inline-block; padding:10px 14px; border-radius:12px; text-decoration:none; color:#081120;
      background:var(--accent); font-weight:600;
    }}
    .btn.ghost {{ background:transparent; color:var(--text); border:1px solid rgba(255,255,255,0.15); }}
    ul {{ margin:10px 0 0 18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Arabic Document Layout Analysis Agent</h1>
      <p>
        API de pré-annotation: envoyez une image et recevez des boîtes <em>layout</em>
        en JSON, une image annotée (PNG) ou du texte compatible YOLO.
      </p>
      <div class="row">
        <a class="btn" href="/docs">Swagger Docs</a>
        <a class="btn ghost" href="/redoc">ReDoc</a>
        <a class="btn ghost" href="/health">Health</a>
      </div>
      <h3 style="margin-top:22px;">Endpoints rapides</h3>
      <ul>
        <li><code>POST /infer</code> → JSON structuré</li>
        <li><code>POST /infer_image</code> → PNG annoté</li>
        <li><code>POST /infer_yolo_txt</code> → lignes YOLO <code>cls cx cy w h</code></li>
      </ul>
    </div>
  </div>
</body>
</html>
        """
    )

# -------------------------------------------------
# Docs Swagger & ReDoc custom
# -------------------------------------------------
@app.get("/docs", include_in_schema=False)
def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Arabic Document Layout Analysis Agent — API Docs",
        swagger_ui_parameters={
            "docExpansion": "list",              # sections ouvertes en liste
            "defaultModelsExpandDepth": -1,      # cache la section "Schemas"
            "displayRequestDuration": True,      # temps des requêtes
            "tryItOutEnabled": True,             # bouton Try it out activé
        },
    )

@app.get("/redoc", include_in_schema=False)
def redoc_ui():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title="Arabic Document Layout Analysis Agent — ReDoc",
    )

# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get(
    "/health",
    tags=["Health"],
    summary="Health check & infos modèle",
    response_description="Statut de l’API et configuration du modèle chargé."
)
def health():
    return {"status": "ok", "classes": CLASS_NAMES, "weights": str(WEIGHTS)}

@app.post(
    "/infer",
    tags=["Inference"],
    summary="Inférence → JSON",
    description="Upload d’une image, renvoie les détections en JSON (pixels + coordonnées normalisées YOLO).",
    response_model=PredictResponse,
)
async def infer(
    file: UploadFile = File(...),
    imgsz: int = Form(1280),
    iou: float = Form(0.5),
    conf_min: float = Form(0.001)
):
    buf = await file.read()
    npimg = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    res = MODEL.predict(img, imgsz=imgsz, iou=iou, conf=conf_min, verbose=False)[0]
    raw = _ultra_to_dicts(res, w, h)
    filt = filter_by_class_conf(raw, CLASS_CONF_LIST, EXCLUDE_SET, CLASS_NAMES)
    final = simple_layout_rules(filt, w, h)

    payload = {
        "width": w, "height": h,
        "detections": [{
            "cls_name": d["cls_name"],
            "cls_id": d["cls_id"],
            "conf": d["conf"],
            "x1": int(d["xyxy"][0]), "y1": int(d["xyxy"][1]),
            "x2": int(d["xyxy"][2]), "y2": int(d["xyxy"][3]),
            "cx": d["xywhn"][0], "cy": d["xywhn"][1],
            "w": d["xywhn"][2], "h": d["xywhn"][3],
        } for d in final]
    }
    return JSONResponse(payload)

@app.post(
    "/infer_image",
    tags=["Inference"],
    summary="Inférence → Image annotée (PNG)",
    description="Upload d’une image, renvoie la même image avec boîtes & libellés dessinés."
)
async def infer_image(
    file: UploadFile = File(...),
    imgsz: int = Form(1280),
    iou: float = Form(0.5),
    conf_min: float = Form(0.001)
):
    buf = await file.read()
    npimg = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    res = MODEL.predict(img, imgsz=imgsz, iou=iou, conf=conf_min, verbose=False)[0]
    raw = _ultra_to_dicts(res, w, h)
    filt = filter_by_class_conf(raw, CLASS_CONF_LIST, EXCLUDE_SET, CLASS_NAMES)
    final = simple_layout_rules(filt, w, h)

    out = draw_detections(img.copy(), final)
    ok, png = cv2.imencode(".png", out)
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post(
    "/infer_yolo_txt",
    tags=["Inference"],
    summary="Inférence → Lignes YOLO (TXT)",
    description="Upload d’une image, renvoie un texte où chaque ligne est `cls cx cy w h` (coordonnées normalisées YOLO)."
)
async def infer_yolo_txt(
    file: UploadFile = File(...),
    imgsz: int = Form(1280),
    iou: float = Form(0.5),
    conf_min: float = Form(0.001)
):
    buf = await file.read()
    npimg = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    res = MODEL.predict(img, imgsz=imgsz, iou=iou, conf=conf_min, verbose=False)[0]
    raw = _ultra_to_dicts(res, w, h)
    filt = filter_by_class_conf(raw, CLASS_CONF_LIST, EXCLUDE_SET, CLASS_NAMES)
    final = simple_layout_rules(filt, w, h)

    lines = []
    for d in final:
        cx, cy, ww, hh = d["xywhn"]
        lines.append(f'{d["cls_id"]} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}')
    content = "\n".join(lines) + ("\n" if lines else "")
    return PlainTextResponse(content)
