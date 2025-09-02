# app/draw.py
from typing import List, Dict, Tuple
import cv2
import numpy as np
import colorsys

# Palette BGR fixe par classe (OpenCV = BGR)
# -> Ajuste librement si tu veux d'autres couleurs.
FIXED_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "Header": (255, 0, 0),                 # Bleu
    "Title": (0, 0, 255),                  # Rouge
    "Text": (0, 170, 0),                   # Vert soutenu
    "Table": (0, 255, 255),                # Jaune (BGR)
    "Image": (255, 0, 255),                # Magenta
    "Footer": (255, 255, 0),               # Cyan
    "Stamp or Signature": (0, 140, 255),   # Orange
    "Caption": (147, 20, 255),             # Rose/Violet
    "Keyvalue": (0, 255, 127),             # Vert clair
    "List-item": (255, 128, 0),            # Ciel/bleu clair
    "Check-box": (128, 128, 128),          # Gris
    # S'il y a d'autres classes inattendues, un fallback sera utilisé (voir plus bas).
}

def _hash_color_bgr(name: str) -> Tuple[int, int, int]:
    """
    Couleur de secours déterministe basée sur le nom (HSV -> BGR).
    Évite d'avoir du noir/blanc en choisissant saturation/valeur élevées.
    """
    h = (abs(hash(name)) % 360) / 360.0
    s, v = 0.75, 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # Convertir en BGR (0-255)
    return (int(b * 255), int(g * 255), int(r * 255))

def _get_color(cls_name: str) -> Tuple[int, int, int]:
    return FIXED_PALETTE.get(cls_name, _hash_color_bgr(cls_name))

def _text_color_for(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Choisit du texte noir ou blanc selon la luminance perçue.
    """
    b, g, r = bgr
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if y > 170 else (255, 255, 255)

def _draw_label(img, x1, y1, text, color):
    """
    Dessine un fond coloré + texte lisible.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    th = 2
    (tw, th_text), _ = cv2.getTextSize(text, font, fs, th)
    # Fond du label
    x2 = x1 + tw + 8
    y2 = y1 - th_text - 8
    if y2 < 0:  # Si trop haut, bascule sous la box
        y2 = y1 + th_text + 8
        y1_lab = y1
    else:
        y1_lab = y1 - 2
    cv2.rectangle(img, (x1, y2), (x2, y1_lab), color, thickness=-1)
    # Texte
    tcolor = _text_color_for(color)
    ty = y2 + th_text + 3 if y2 >= 0 else y1 + th_text + 3
    cv2.putText(img, text, (x1 + 4, ty), font, fs, tcolor, th, cv2.LINE_AA)

def draw_detections(
    img: np.ndarray,
    dets: List[dict],
    class_palette: Dict[str, Tuple[int, int, int]] = None,
    fill_alpha: float = 0.15,
    thickness: int = 2,
    font_scale: float = 0.6,
    show_legend: bool = True
) -> np.ndarray:
    """
    Dessine des boxes colorées par classe + labels sur l'image.
    - dets: liste de dicts [{"cls_name","conf","xyxy":[x1,y1,x2,y2], ...}, ...]
    - class_palette: palette personnalisée (optionnel)
    """
    if class_palette is None:
        class_palette = FIXED_PALETTE

    overlay = img.copy()

    # 1) Remplissage semi-translucide + contour
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cls_name = d.get("cls_name", "unknown")
        conf = d.get("conf", 0.0)
        color = class_palette.get(cls_name, _get_color(cls_name))

        # Remplissage translucide
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

    # Fusion du remplissage
    img = cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0)

    # 2) Contours + labels
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cls_name = d.get("cls_name", "unknown")
        conf = float(d.get("conf", 0.0))
        color = class_palette.get(cls_name, _get_color(cls_name))

        # Contour
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)

        # Label
        label = f"{cls_name} {conf:.2f}"
        _draw_label(img, x1, y1, label, color)

    # 3) Légende (facultative) avec les classes présentes sur l'image
    if show_legend and len(dets) > 0:
        present = []
        seen = set()
        for d in dets:
            name = d.get("cls_name", "unknown")
            if name not in seen:
                present.append(name)
                seen.add(name)

        # Zone de légende à gauche
        x0, y0 = 12, 12
        h_line = 22
        for i, name in enumerate(present[:20]):  # éviter d'en mettre trop
            color = class_palette.get(name, _get_color(name))
            y = y0 + i * h_line
            # Carré couleur
            cv2.rectangle(img, (x0, y), (x0 + 16, y + 16), color, thickness=-1)
            # Texte
            tcolor = _text_color_for(color)
            cv2.putText(img, name, (x0 + 22, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tcolor, 1, cv2.LINE_AA)

    return img
