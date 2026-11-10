# src/crops.py
"""실험 E: 앱 장면처럼 '물건 하나를 크게, 가운데에' 자른 사진(crop)을 만든다.

앱은 스캔 틀(정사각형)을 320×320으로 줄여 모델에 넣는다. 그런데 학습 데이터(Open Images)는 장면 전체 사진이다.
그래서 학습 사진에서 박스마다 정사각형으로 잘라 '앱이 보는 사진'을 더 만든다.

- 목표 박스(13종)를 가운데에 둔 crop → 그 안에 40% 이상 보이는 목표 박스를 라벨로
- 네거티브 박스(폰 · 리모컨 · 손 …)를 가운데에 둔 crop → 라벨은 그 안의 목표 박스뿐 (보통 없음 = 배경)
"""
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import dedupe_rows

MIN_AREA = 0.01           # 사진 면적의 1% 미만 박스는 crop 중심으로 쓰지 않음
MIN_VISIBLE = 0.4         # crop 안에 박스 면적의 40% 이상 보여야 라벨로 남김
SIZE = 320                # 앱 입력 크기


# ---------- 기하 (순수 함수) ----------

def square_at(img_w, img_h, box_px, fill, shift=(0.0, 0.0)):
    """박스(x1, y1, x2, y2 픽셀)를 가운데에 두고 긴 변이 한 변의 fill배인 정사각형 → (x, y, 한 변).
    shift=(dx, dy): 중심을 한 변의 dx, dy배만큼 옮김 (사람이 정확히 가운데에 대지 않는 것 흉내).
    사진 밖으로 나가면 안쪽으로 밀고, 사진이 더 작으면 들어가는 가장 큰 정사각형 (src.clean.crop_square와 같은 규칙)."""
    x1, y1, x2, y2 = box_px
    side = max(1, round(max(x2 - x1, y2 - y1) / fill))
    side = min(side, img_w, img_h)
    cx = (x1 + x2) / 2 + shift[0] * side
    cy = (y1 + y2) / 2 + shift[1] * side
    x = min(max(round(cx - side / 2), 0), img_w - side)
    y = min(max(round(cy - side / 2), 0), img_h - side)
    return x, y, side


def visible_fraction(box_px, crop):
    """박스 면적 중 crop(x, y, 한 변) 안에 들어온 비율."""
    x1, y1, x2, y2 = box_px
    cx, cy, s = crop
    ix = max(0.0, min(x2, cx + s) - max(x1, cx))
    iy = max(0.0, min(y2, cy + s) - max(y1, cy))
    area = (x2 - x1) * (y2 - y1)
    return ix * iy / area if area > 0 else 0.0


def rows_in_crop(targets_px, crop, min_visible=MIN_VISIBLE):
    """목표 박스 [(cls, x1, y1, x2, y2) 픽셀] → crop 안의 YOLO 행 [(cls, cx, cy, w, h) 0~1].
    min_visible 이상 보이는 박스만, crop 경계로 잘라서."""
    cx0, cy0, s = crop
    rows = []
    for k, x1, y1, x2, y2 in targets_px:
        if visible_fraction((x1, y1, x2, y2), crop) < min_visible:
            continue
        a1, b1 = max(x1, cx0) - cx0, max(y1, cy0) - cy0
        a2, b2 = min(x2, cx0 + s) - cx0, min(y2, cy0 + s) - cy0
        rows.append((int(k), (a1 + a2) / 2 / s, (b1 + b2) / 2 / s, (a2 - a1) / s, (b2 - b1) / s))
    return rows


def norm_to_px(box, img_w, img_h):
    """Open Images/FiftyOne 정규화 [x, y, w, h] → 픽셀 (x1, y1, x2, y2)."""
    x, y, w, h = box
    return x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h


def half_of(image_id: str) -> str:
    """사진 이름 → 'tune' 또는 'test' (md5 해시로 정해서 항상 같고, 대략 반반)."""
    return "tune" if int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


def plan_crops(img_w, img_h, targets_px, centers, rng, n_per=None, fill_range=(0.35, 0.8),
               shift_max=0.1, fixed_fill=None):
    """crop 계획 목록. centers: [(이름, 박스px)] 중심으로 쓸 박스들. n_per(이름) → 그 박스로 만들 crop 수 (기본 1).
    fixed_fill이 있으면 무작위 없이 그 fill로 가운데에 (평가용).
    반환: [dict(center=이름, crop=(x, y, 한 변), rows=[YOLO 행])]"""
    plans = []
    for name, box in centers:
        for _ in range(n_per(name) if n_per else 1):
            if fixed_fill is not None:
                fill, shift = fixed_fill, (0.0, 0.0)
            else:
                fill = float(rng.uniform(*fill_range))
                shift = (float(rng.uniform(-shift_max, shift_max)), float(rng.uniform(-shift_max, shift_max)))
            crop = square_at(img_w, img_h, box, fill, shift)
            plans.append(dict(center=name, crop=crop, rows=rows_in_crop(targets_px, crop)))
    return plans


# ---------- FiftyOne → 단순 기록 ----------

def records_from_dataset(ds, oi_to_key, keys, neg_labels=()):
    """FiftyOne 데이터셋 → [dict(filepath, targets=[(cls, x, y, w, h) 정규화], negs=[(라벨, x, y, w, h)])].
    목표 라벨(oi_to_key)은 클래스 번호로 바꾸고, neg_labels는 네거티브 박스로 따로 모은다. 나머지 박스는 버린다."""
    cls_of = {k: i for i, k in enumerate(keys)}
    paths = ds.values("filepath")
    labels = ds.values("ground_truth.detections.label")
    boxes = ds.values("ground_truth.detections.bounding_box")
    out = []
    for fp, labs, bbs in zip(paths, labels, boxes):
        t, n = [], []
        for lab, bb in zip(labs or [], bbs or []):
            if lab in oi_to_key:
                t.append((cls_of[oi_to_key[lab]], *bb))
            elif lab in neg_labels:
                n.append((lab, *bb))
        out.append(dict(filepath=fp, targets=t, negs=n))
    return out


def merge_records(*lists):
    """filepath가 같은 기록은 하나만 (먼저 나온 것)."""
    seen, out = set(), []
    for lst in lists:
        for r in lst:
            if r["filepath"] not in seen:
                seen.add(r["filepath"]); out.append(r)
    return out


def _yolo_rows(targets):
    """정규화 [x, y, w, h] 목표 박스 → YOLO 행, 같은 클래스끼리 IoU 0.7 이상 겹치면 하나만."""
    return dedupe_rows([(k, x + w / 2, y + h / 2, w, h) for k, x, y, w, h in targets], 0.7)


def _write_label(path, rows):
    path.write_text("".join(f"{k} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for k, cx, cy, w, h in rows))


def write_yaml(out_dir: Path, keys, train="images/train", val="images/val"):
    lines = [f"path: {Path(out_dir).resolve()}", f"train: {train}", f"val: {val}", "names:"]
    lines += [f"  {i}: {k}" for i, k in enumerate(keys)]
    (Path(out_dir) / "dataset.yaml").write_text("\n".join(lines) + "\n")


def _fresh(out_dir, split):
    img_dir, lab_dir = Path(out_dir) / "images" / split, Path(out_dir) / "labels" / split
    shutil.rmtree(img_dir, ignore_errors=True); shutil.rmtree(lab_dir, ignore_errors=True)
    img_dir.mkdir(parents=True); lab_dir.mkdir(parents=True)
    return img_dir, lab_dir


def export_full(records, out_dir, split, max_side=640):
    """사진 전체를 YOLO 형식으로 (긴 변을 max_side로 줄여 저장 → 320 학습 때 읽기가 빠름).
    네거티브 박스는 내보내지 않는다 = 배경. 끝나면 labels/<split>.done. 반환: 사진 수."""
    import cv2
    out_dir = Path(out_dir)
    done = out_dir / "labels" / f"{split}.done"
    if done.exists():
        return len(list((out_dir / "labels" / split).glob("*.txt")))
    img_dir, lab_dir = _fresh(out_dir, split)
    for r in records:
        stem = Path(r["filepath"]).stem
        img = cv2.imread(r["filepath"])
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            k = max_side / max(h, w)
            img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        _write_label(lab_dir / f"{stem}.txt", _yolo_rows(r["targets"]))
    done.touch()
    return len(records)


def export_crops(records, out_dir, split, seed=0, n_per_class=None, neg_per_image=3, fixed_fill=None,
                 min_area=MIN_AREA, size=SIZE, table_name=None, keys=None, pos=True, neg=True,
                 pos_per_class=None):
    """기록마다 crop을 만들어 out_dir/images/<split>, labels/<split>에 저장하고 표를 돌려준다.
    - 목표 박스(면적 ≥ min_area)마다 n_per_class.get(cls, 1)장
    - 네거티브 박스(면적 ≥ min_area) 중 큰 것부터 neg_per_image개, 각 1장
    표 열: file, image, kind(pos/neg), target(목표 클래스, neg는 -1), center(클래스 키 또는 네거티브 라벨), n_rows, half
    pos_per_class: 사진 한 장에서 클래스마다 큰 박스부터 이 개수만 중심으로 씀 (None = 모두; 책장처럼 박스가 많은 사진이 crop을 독차지하지 않게).
    pos/neg=False면 그 종류의 박스는 crop 중심으로 쓰지 않는다 (라벨 계산에는 목표 박스를 늘 씀).
    끝나면 labels/<split>.done + <split>_crops.csv (다음엔 표만 읽음)."""
    import cv2
    out_dir = Path(out_dir)
    done = out_dir / "labels" / f"{split}.done"
    table = out_dir / f"{table_name or split}_crops.csv"
    if done.exists():
        return pd.read_csv(table)
    img_dir, lab_dir = _fresh(out_dir, split)
    rng = np.random.default_rng(seed)
    n_per_class = n_per_class or {}
    rows_out = []
    for r in records:
        stem = Path(r["filepath"]).stem
        big_t = [t for t in r["targets"] if t[3] * t[4] >= min_area] if pos else []
        if pos_per_class is not None:
            big_t = sorted(big_t, key=lambda t: -t[3] * t[4])
            big_t = [t for i, t in enumerate(big_t) if sum(u[0] == t[0] for u in big_t[:i]) < pos_per_class]
        big_n = sorted([n for n in r["negs"] if n[3] * n[4] >= min_area], key=lambda n: -n[3] * n[4])[:neg_per_image] if neg else []
        if not big_t and not big_n:
            continue
        img = cv2.imread(r["filepath"])
        h, w = img.shape[:2]
        targets_px = [(k, *norm_to_px(b, w, h)) for k, *b in r["targets"]]
        centers = [(("pos", k), norm_to_px(b, w, h)) for k, *b in big_t]
        centers += [(("neg", lab), norm_to_px(b, w, h)) for lab, *b in big_n]
        n_per = lambda name: n_per_class.get(name[1], 1) if name[0] == "pos" else 1
        plans = plan_crops(w, h, targets_px, centers, rng, n_per=n_per, fixed_fill=fixed_fill)
        for j, p in enumerate(plans):
            x, y, s = p["crop"]
            crop = cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)
            name = f"{stem}_c{j}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows = dedupe_rows(p["rows"], 0.7)
            _write_label(lab_dir / f"{name}.txt", rows)
            kind, what = p["center"]
            rows_out.append(dict(file=f"{name}.jpg", image=stem, kind=kind,
                                 target=int(what) if kind == "pos" else -1,
                                 center=(keys[int(what)] if keys else str(what)) if kind == "pos" else what,
                                 n_rows=len(rows), half=half_of(stem)))
    df = pd.DataFrame(rows_out, columns=["file", "image", "kind", "target", "center", "n_rows", "half"])
    df.to_csv(table, index=False)
    done.touch()
    return df


def link_split(src_dirs, dst_dir, split):
    """여러 YOLO 폴더의 images/labels/<split>을 dst_dir 하나로 하드링크해 합친다 (학습용 train = 전체 사진 + crop)."""
    dst_dir = Path(dst_dir)
    img_dst, lab_dst = _fresh(dst_dir, split)
    n = 0
    for src in src_dirs:
        for lab in (Path(src) / "labels" / split).glob("*.txt"):
            img = Path(src) / "images" / split / (lab.stem + ".jpg")
            for a, b in ((img, img_dst / img.name), (lab, lab_dst / lab.name)):
                try:
                    os.link(a, b)
                except OSError:
                    shutil.copy(a, b)
            n += 1
    return n
