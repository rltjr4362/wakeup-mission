# src/data.py
"""Open Images V7에서 10종을 받아 YOLO 형식으로 내보낸다."""
import os
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

from . import config as C


def load_split(split: str, per_class: int, seed: int = 51) -> fo.Dataset:
    """클래스마다 따로 받아(per_class장) 하나로 합친다. 같은 사진은 한 번만 들어간다."""
    name = f"wakeup-{split}-{per_class}"
    if fo.dataset_exists(name):
        return fo.load_dataset(name)
    merged = fo.Dataset(name, persistent=True)
    for c in C.CLASSES:
        part = foz.load_zoo_dataset(
            "open-images-v7", split=split, label_types=["detections"],
            classes=list(c.oi_labels), max_samples=per_class, seed=seed, shuffle=True,
            dataset_name=f"oi-{split}-{c.key}-{per_class}")
        merged.merge_samples(part)          # filepath 기준으로 합침 (중복 제거)
    return merged


def to_mission_view(ds: fo.Dataset):
    """10종에 해당하는 박스만 남기고, Open Images 라벨을 우리 클래스 키로 바꾼다."""
    view = ds.filter_labels("ground_truth", F("label").is_in(list(C.OI_TO_KEY)), only_matches=True)
    return view.map_labels("ground_truth", C.OI_TO_KEY)


def export_yolo(view, out_dir: Path, split: str):
    """YOLO 형식: out_dir/images/{split}, out_dir/labels/{split}, out_dir/dataset.yaml"""
    view.export(export_dir=str(out_dir), dataset_type=fo.types.YOLOv5Dataset,
                label_field="ground_truth", split=split, classes=C.KEYS)


def count_table(view):
    """클래스별 사진 수 · 박스 수 (pandas DataFrame)."""
    import pandas as pd
    boxes = view.count_values("ground_truth.detections.label")
    rows = []
    for c in C.CLASSES:
        n_img = len(view.match(F("ground_truth.detections").filter(F("label") == c.key).length() > 0))
        rows.append(dict(id=c.id, 클래스=c.ko, 키=c.key, 사진=n_img, 박스=boxes.get(c.key, 0),
                         구분="기존(COCO)" if c.coco_id is not None else "새로"))
    return pd.DataFrame(rows)


def make_old_subset(yolo_dir: Path, out_dir: Path):
    """기존 4종이 있는 val 사진만 모아 두 벌의 라벨을 만든다.
    - out_dir/ours : 우리 번호(0~3) → 파인튜닝 모델 평가용
    - out_dir/coco : COCO 번호(64, 25, 73, 79) → 사전학습 COCO YOLO 평가용
    두 벌이 같은 사진을 쓰므로 기준선과 공정하게 비교된다."""
    old_ids = {c.id: c.coco_id for c in C.OLD}
    coco_names = _coco_names()
    for variant in ("ours", "coco"):
        (out_dir / variant / "images" / "val").mkdir(parents=True, exist_ok=True)
        (out_dir / variant / "labels" / "val").mkdir(parents=True, exist_ok=True)
    for lab in sorted((yolo_dir / "labels" / "val").glob("*.txt")):
        rows = [r.split() for r in lab.read_text().splitlines() if r.strip()]
        rows = [r for r in rows if int(r[0]) in old_ids]
        if not rows:
            continue
        img = next((yolo_dir / "images" / "val").glob(lab.stem + ".*"))
        for variant, mapping in (("ours", {k: k for k in old_ids}), ("coco", old_ids)):
            dst = out_dir / variant / "images" / "val" / img.name
            if not dst.exists():
                try:
                    os.link(img, dst)           # 하드링크 (용량 안 씀)
                except OSError:
                    shutil.copy(img, dst)
            text = "\n".join(" ".join([str(mapping[int(r[0])])] + r[1:]) for r in rows) + "\n"
            (out_dir / variant / "labels" / "val" / lab.name).write_text(text)
    _write_yaml(out_dir / "ours.yaml", out_dir / "ours", C.KEYS)
    _write_yaml(out_dir / "coco.yaml", out_dir / "coco", coco_names)


def _coco_names():
    from ultralytics import YOLO
    names = YOLO("yolo11n.pt").names
    return [names[i] for i in range(len(names))]


def _write_yaml(path: Path, root: Path, names):
    lines = [f"path: {root.resolve()}", "train: images/val", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")
