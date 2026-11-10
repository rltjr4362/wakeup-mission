# src/missioneval.py
"""실험 E 평가: 앱 규칙(주인공 박스)으로 통과율 · 잘못 통과율 · 클래스별 신뢰도 기준 · P/R/F1.

예측은 가장 낮은 기준(0.4)으로 한 번만 하고 박스를 모두 저장한다. 기준을 올리는 것은 저장한 박스를 거르기만 하면 된다
(같은 클래스 안에서 점수 높은 박스가 먼저 남는 NMS라서, 거른 뒤 결과 = 그 기준으로 다시 예측한 결과).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .postprocess import pick_main

GRID = tuple(round(0.4 + 0.05 * i, 2) for i in range(11))     # 0.40, 0.45, …, 0.90


def thr_of(thr, cls):
    """thr: 숫자 하나(모든 클래스 같음) 또는 클래스 번호 → 기준 (dict/list)."""
    return thr if isinstance(thr, (int, float)) else thr[cls]


def apply_thr(dets, thr):
    """클래스별 기준 이상인 박스만."""
    return [d for d in dets if d["score"] >= thr_of(thr, d["cls"])]


def main_cls(dets, thr, input_size=320, min_side=0.4) -> int:
    """기준으로 거른 뒤 주인공 박스의 클래스. 없으면 -1."""
    m = pick_main(apply_thr(dets, thr), input_size, min_side)
    return -1 if m is None else int(m["cls"])


def mains(dets_list, thr, input_size=320, min_side=0.4) -> np.ndarray:
    return np.array([main_cls(d, thr, input_size, min_side) for d in dets_list], dtype=int)


def false_pass(neg_mains, n_classes) -> np.ndarray:
    """네거티브 crop 중 주인공 박스가 클래스 k인 비율 (= 미션이 k일 때 잘못 통과)."""
    neg_mains = np.asarray(neg_mains)
    if len(neg_mains) == 0:
        return np.zeros(n_classes)
    return np.array([(neg_mains == k).mean() for k in range(n_classes)])


def choose_thresholds(neg_dets, n_classes, grid=GRID, max_fp=0.02, input_size=320, min_side=0.4) -> dict:
    """클래스마다 '잘못 통과율 ≤ max_fp'인 가장 낮은 기준. 없으면 grid의 가장 큰 값.
    잘못 통과율은 모든 클래스에 같은 기준 t를 썼을 때로 잰다 (클래스끼리 서로 영향을 주는 것을 단순화)."""
    fp = {t: false_pass(mains(neg_dets, t, input_size, min_side), n_classes) for t in grid}
    out = {}
    for k in range(n_classes):
        ok = [t for t in grid if fp[t][k] <= max_fp]
        out[k] = min(ok) if ok else max(grid)
    return out


def prf(targets, preds, n_classes) -> pd.DataFrame:
    """crop 하나 = 문제 하나. targets: 정답 클래스(네거티브는 -1), preds: 주인공 박스 클래스(없음 -1).
    클래스 k: TP = 정답 k를 k로, FP = 정답이 k가 아닌데 k로, FN = 정답 k를 k가 아닌 것으로. 열: TP, FP, FN, P, R, F1"""
    targets, preds = np.asarray(targets), np.asarray(preds)
    rows = []
    for k in range(n_classes):
        tp = int(((targets == k) & (preds == k)).sum())
        fp = int(((targets != k) & (preds == k)).sum())
        fn = int(((targets == k) & (preds != k)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append(dict(id=k, TP=tp, FP=fp, FN=fn, P=p, R=r, F1=f1))
    return pd.DataFrame(rows).set_index("id")


def predict_dets(model, files, crops_dir, conf=0.4, iou=0.45, imgsz=320, device="cpu", batch=32):
    """crop 사진마다 ultralytics 예측 박스 [dict(x1, y1, x2, y2, score, cls)] 목록."""
    out = []
    files = list(files)
    for i in range(0, len(files), batch):
        paths = [str(Path(crops_dir) / f) for f in files[i:i + batch]]
        for res in model.predict(paths, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False):
            b = res.boxes
            xyxy, sc, cl = (t.cpu().numpy() for t in (b.xyxy, b.conf, b.cls))
            out.append([dict(x1=float(a), y1=float(b_), x2=float(c), y2=float(d), score=float(s), cls=int(k))
                        for (a, b_, c, d), s, k in zip(xyxy, sc, cl)])
    return out
