from typing import List, Dict

def filter_by_class_conf(dets: List[Dict], class_conf_list, exclude_names, class_names):
    out = []
    for d in dets:
        if d["cls_name"] in exclude_names:
            continue
        thr = class_conf_list[d["cls_id"]]
        if d["conf"] >= thr:
            out.append(d)
    return out

def simple_layout_rules(dets: List[Dict], img_w: int, img_h: int):
    if not dets:
        return dets

    for d in dets:
        x1,y1,x2,y2 = d["xyxy"]
        d["_area"] = (x2-x1)*(y2-y1)

    # Header doit commencer dans bande haute 10% de l'image
    top_band = 0.1 * img_h
    keep = []
    for d in dets:
        if d["cls_name"] == "Header":
            if d["xyxy"][1] <= top_band:
                keep.append(d)
        else:
            keep.append(d)
    dets = keep

    # Conflit Title vs Text: si IoU>0.5, on garde Title, on retire Text
    def iou(a,b):
        ax1,ay1,ax2,ay2 = a["xyxy"]; bx1,by1,bx2,by2 = b["xyxy"]
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw, ih = max(0,ix2-ix1), max(0,iy2-iy1)
        inter = iw*ih
        if inter<=0: return 0.0
        ua = a["_area"] + b["_area"] - inter
        return inter/ua if ua>0 else 0.0

    titles = [d for d in dets if d["cls_name"]=="Title"]
    texts  = [d for d in dets if d["cls_name"]=="Text"]
    drop_ids = set()
    for t in titles:
        for x in texts:
            if iou(t,x) > 0.5:
                drop_ids.add(id(x))
    dets = [d for d in dets if not (d["cls_name"]=="Text" and id(d) in drop_ids)]

    for d in dets:
        d.pop("_area", None)

    return dets
