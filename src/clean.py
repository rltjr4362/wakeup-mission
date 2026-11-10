# src/clean.py
"""실험 C: 데이터 청소. 앱 장면(물건 하나, 크게, 가운데)과 다른 박스를 지운다.

지우는 박스: 묶음 박스(IsGroupOf) · 그림/사진 속 물건(IsDepiction) · 사진 면적의 1% 미만.
라벨만 지우면 모델이 그 물건을 '배경'으로 배우므로(빠진 라벨 함정), 지운 박스 자리를
YOLO 레터박스 색(114 회색)으로 칠한다. 남긴 박스 안의 픽셀은 칠하지 않는다.
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import dedupe_rows
from .evaluate import mission_passed

GRAY = (114, 114, 114)
EPS = 1e-6          # 0.5*10 = 5.0000000001 같은 부동소수 오차로 한 픽셀 더 칠하지 않게


def _flag(v) -> bool:
    """Open Images 속성값(None / bool / 0·1 / -1)을 bool로. 1(True)일 때만 True."""
    return v is True or v == 1


ATTRS = ("IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside")


def is_unknown(d) -> bool:
    """Open Images CSV는 '모름'을 -1로 적는데 FiftyOne은 이를 True로 읽는다.
    그래서 다섯 속성이 모두 True면 실제로는 '모름'으로 본다."""
    return all(_flag(getattr(d, a, None)) for a in ATTRS)


def should_drop(is_group_of, is_depiction, area, min_area=0.01, unknown=False) -> bool:
    """앱 장면과 다른 박스면 True. area는 사진 면적 대비 비율(w*h, 0~1).
    속성이 '모름'(unknown)이면 묶음 · 그림 표시를 믿지 않고 면적 규칙만 쓴다."""
    if unknown:
        return area < min_area
    return _flag(is_group_of) or _flag(is_depiction) or area < min_area


def box_px(box, h, w):
    """정규화 [x, y, w, h] → 픽셀 (x1, y1, x2, y2). 바깥쪽으로 반올림(floor/ceil)하고 사진 안으로 자른다."""
    x, y, bw, bh = box
    x1 = min(max(math.floor(x * w + EPS), 0), w)
    y1 = min(max(math.floor(y * h + EPS), 0), h)
    x2 = min(max(math.ceil((x + bw) * w - EPS), 0), w)
    y2 = min(max(math.ceil((y + bh) * h - EPS), 0), h)
    return x1, y1, x2, y2


def paint_mask(h, w, drop_boxes, keep_boxes) -> np.ndarray:
    """칠할 곳 = (지울 박스들의 합집합) − (남길 박스들의 합집합)."""
    drop = np.zeros((h, w), bool)
    for b in drop_boxes:
        x1, y1, x2, y2 = box_px(b, h, w)
        drop[y1:y2, x1:x2] = True
    for b in keep_boxes:
        x1, y1, x2, y2 = box_px(b, h, w)
        drop[y1:y2, x1:x2] = False
    return drop


def split_dets(dets, min_area):
    """우리 10종 박스 → (남길 박스, 지울 박스). 클래스 키가 아닌 라벨은 무시."""
    kept, dropped = [], []
    for d in dets:
        if d.label not in C.BY_KEY:
            continue
        x, y, bw, bh = d.bounding_box
        if should_drop(getattr(d, "IsGroupOf", None), getattr(d, "IsDepiction", None), bw * bh, min_area, unknown=is_unknown(d)):
            dropped.append(d)
        else:
            kept.append(d)
    return kept, dropped


def _dets_of(sample):
    gt = sample.ground_truth
    return gt.detections if gt is not None else []


def clean_and_export(view, out_dir: Path, split: str, min_area=0.01) -> dict:
    """view의 사진을 청소해서 YOLO 형식으로 내보낸다.
    out_dir/images/{split}/*.jpg, out_dir/labels/{split}/*.txt, out_dir/dataset.yaml
    끝까지 되면 out_dir/labels/{split}.done 을 남기고, 다음엔 저장한 통계만 읽어 온다."""
    import cv2
    out_dir = Path(out_dir)
    img_dir, lab_dir = out_dir / "images" / split, out_dir / "labels" / split
    done = out_dir / "labels" / f"{split}.done"
    stats_file = out_dir / "labels" / f"{split}_stats.json"
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    write_yaml(out_dir)
    if done.exists():
        return json.loads(stats_file.read_text())
    shutil.rmtree(img_dir, ignore_errors=True)                       # 중단된 반쪽 폴더는 지우고 처음부터
    shutil.rmtree(lab_dir, ignore_errors=True)
    img_dir.mkdir(parents=True); lab_dir.mkdir(parents=True)

    stats = dict(images=0, kept={k: 0 for k in C.KEYS}, dropped={k: 0 for k in C.KEYS}, all_dropped=0)
    for sample in view:
        img = cv2.imread(sample.filepath)
        h, w = img.shape[:2]
        kept, dropped = split_dets(_dets_of(sample), min_area)

        # 1) 지운 박스 자리를 회색으로 (남긴 박스 안은 그대로)
        mask = paint_mask(h, w, [d.bounding_box for d in dropped], [d.bounding_box for d in kept])
        img[mask] = GRAY

        # 2) 남긴 박스 → YOLO 행 (cls, cx, cy, w, h) → 겹친 박스 제거
        rows = []
        for d in kept:
            x, y, bw, bh = d.bounding_box
            rows.append((C.BY_KEY[d.label].id, x + bw / 2, y + bh / 2, bw, bh))
        rows = dedupe_rows(rows, 0.7)

        stem = Path(sample.filepath).stem
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lab_dir / f"{stem}.txt").write_text(
            "".join(f"{k} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for k, cx, cy, bw, bh in rows))

        stats["images"] += 1
        for r in rows:
            stats["kept"][C.KEYS[r[0]]] += 1
        for d in dropped:
            stats["dropped"][d.label] += 1
        if dropped and not rows:
            stats["all_dropped"] += 1                                # 사진 전체가 지워짐 → 빈 라벨 (회색 = 아무것도 없음)

    stats_file.write_text(json.dumps(stats, ensure_ascii=False))
    done.touch()
    return stats


def write_yaml(out_dir: Path):
    """out_dir/dataset.yaml (path는 절대 경로, 클래스 0~9)."""
    lines = [f"path: {Path(out_dir).resolve()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {k}" for i, k in enumerate(C.KEYS)]
    (Path(out_dir) / "dataset.yaml").write_text("\n".join(lines) + "\n")


def stats_table(stats: dict) -> pd.DataFrame:
    """clean_and_export 통계 → 클래스별 남김 · 지움 표."""
    rows = []
    for c in C.CLASSES:
        k, d = stats["kept"][c.key], stats["dropped"][c.key]
        rows.append(dict(id=c.id, 클래스=c.ko, 남김=k, 지움=d, **{"지움%": round(100 * d / (k + d), 1) if k + d else 0.0}))
    return pd.DataFrame(rows)


def reason_table(view, min_area=0.01) -> pd.DataFrame:
    """클래스별로 박스 중 묶음 · 그림 · 작음 · (셋 중 하나라도 = 지움) 비율(%). 중복 제거 전 원래 박스 기준."""
    counts = {k: dict(박스=0, 묶음=0, 그림=0, 작음=0, 지움=0) for k in C.KEYS}
    for sample in view:
        for d in _dets_of(sample):
            if d.label not in counts:
                continue
            x, y, bw, bh = d.bounding_box
            n = counts[d.label]
            unknown = is_unknown(d)                              # '모름'이면 묶음 · 그림 표시를 세지 않음
            group = _flag(getattr(d, "IsGroupOf", None)) and not unknown
            depict = _flag(getattr(d, "IsDepiction", None)) and not unknown
            small = bw * bh < min_area
            n["박스"] += 1
            n["묶음"] += group
            n["그림"] += depict
            n["작음"] += small
            n["지움"] += group or depict or small
    rows = []
    for c in C.CLASSES:
        n = counts[c.key]
        pct = lambda v: round(100 * v / n["박스"], 1) if n["박스"] else 0.0
        rows.append({"id": c.id, "클래스": c.ko, "키": c.key, "박스": n["박스"],
                     "묶음%": pct(n["묶음"]), "그림%": pct(n["그림"]), "작음%": pct(n["작음"]), "지움%": pct(n["지움"])})
    return pd.DataFrame(rows)


# ---------- 앱 기준 통과율 ----------

def crop_square(img_w, img_h, box_px, fill):
    """박스(x1, y1, x2, y2 픽셀)를 가운데에 두고, 긴 변이 정사각형 한 변의 fill배가 되도록 자른다.
    사진 밖으로 나가면 안쪽으로 밀고, 사진이 더 작으면 들어가는 가장 큰 정사각형. → (x, y, 한 변)"""
    x1, y1, x2, y2 = box_px
    side = round(max(x2 - x1, y2 - y1) / fill)
    side = min(side, img_w, img_h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x = min(max(round(cx - side / 2), 0), img_w - side)
    y = min(max(round(cy - side / 2), 0), img_h - side)
    return x, y, side


def mission_crops(yolo_dir, split, out_dir, fill=0.6, min_area=0.01, size=320) -> pd.DataFrame:
    """YOLO 라벨의 박스마다 '앱처럼' 물건을 가운데에 크게 잘라 size×size 사진을 만든다.
    out_dir/<사진이름>_<박스번호>.jpg, 표는 out_dir/crops.csv. 끝나면 out_dir/done."""
    import cv2
    yolo_dir, out_dir = Path(yolo_dir), Path(out_dir)
    done, table = out_dir / "done", out_dir / "crops.csv"
    if done.exists():
        return pd.read_csv(table)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    rows = []
    for lab in sorted((yolo_dir / "labels" / split).glob("*.txt")):
        lines = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        img = cv2.imread(str(next((yolo_dir / "images" / split).glob(lab.stem + ".*"))))
        h, w = img.shape[:2]
        for k, line in enumerate(lines):
            cls = int(line[0])
            cx, cy, bw, bh = map(float, line[1:5])
            if bw * bh < min_area:
                continue
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            x, y, s = crop_square(w, h, box, fill)
            crop = cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)
            name = f"{lab.stem}_{k}.jpg"
            cv2.imwrite(str(out_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(dict(file=name, target=cls, cls_key=C.KEYS[cls]))
    df = pd.DataFrame(rows, columns=["file", "target", "cls_key"])
    df.to_csv(table, index=False)
    done.touch()
    return df


def _np(t):
    """torch 텐서(mps/cpu)든 numpy든 numpy로."""
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def mission_pass_rate(model, crops_df, crops_dir, conf=0.5, iou=0.45, min_side=0.4,
                      imgsz=320, device="cpu", batch=32, class_map=None) -> pd.DataFrame:
    """잘라 둔 사진마다 웹앱과 같은 규칙(src.evaluate.mission_passed)으로 통과 여부.
    열: file, target, cls_key, passed, pred_key(주인공 박스 클래스, 없으면 '-'), score
    class_map: 모델의 클래스 번호 → 우리 번호 (예: COCO 그대로 쓰는 모델 {64: 0, 25: 1, 73: 2, 79: 3}).
               주어지면 표에 없는 클래스(사람 등)는 버린다 = 미션 클래스만 찾도록 설정한 앱."""
    from .postprocess import pick_main
    crops_dir = Path(crops_dir)
    records = crops_df.to_dict("records")
    out = []
    for i in range(0, len(records), batch):
        chunk = records[i:i + batch]
        paths = [str(crops_dir / r["file"]) for r in chunk]
        results = model.predict(paths, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        for r, res in zip(chunk, results):
            b = res.boxes
            dets = []
            for (x1, y1, x2, y2), s, k in zip(_np(b.xyxy), _np(b.conf), _np(b.cls)):
                k = int(k)
                if class_map is not None:
                    if k not in class_map:
                        continue
                    k = class_map[k]
                dets.append(dict(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), score=float(s), cls=k))
            main = pick_main(dets, imgsz, min_side)
            out.append(dict(file=r["file"], target=int(r["target"]), cls_key=r["cls_key"],
                            passed=mission_passed(dets, int(r["target"]), imgsz, min_side),
                            pred_key=C.KEYS[main["cls"]] if main else "-",
                            score=round(main["score"], 3) if main else None))
    return pd.DataFrame(out, columns=["file", "target", "cls_key", "passed", "pred_key", "score"])
