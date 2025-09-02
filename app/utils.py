from pathlib import Path
import yaml, json
from typing import List

def load_class_names(data_yaml_path: str) -> List[str]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    return names

def load_thresholds(thresholds_json_path: str, class_names: list[str]):
    with open(thresholds_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    default_conf = float(cfg.get("default_conf", 0.35))
    per_class = cfg.get("class_thresholds", {})
    excl = set(cfg.get("exclude_classes", []))
    conf_by_idx = []
    for name in class_names:
        conf_by_idx.append(float(per_class.get(name, default_conf)))
    return conf_by_idx, excl
