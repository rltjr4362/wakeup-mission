# src/postprocess.py
"""YOLO ONNX 출력 후처리. docs/postprocess.js와 같은 동작이어야 한다 (웹앱이 할 계산을 파이썬으로 검증)."""
import numpy as np


def decode(output, conf=0.5, input_size=320):
    """[1, 4+nc, N] → 신뢰도 conf 이상인 박스 목록 (각 위치에서 점수가 가장 높은 클래스 하나). 박스는 [0, input_size]로 자른다(ultralytics와 동일)."""
    o = output[0]
    scores = o[4:]                       # [nc, N]
    cls = scores.argmax(0)
    best = scores.max(0)
    dets = []
    for i in np.where(best >= conf)[0]:
        cx, cy, w, h = o[:4, i]
        dets.append(dict(x1=float(np.clip(cx - w / 2, 0, input_size)), y1=float(np.clip(cy - h / 2, 0, input_size)),
                         x2=float(np.clip(cx + w / 2, 0, input_size)), y2=float(np.clip(cy + h / 2, 0, input_size)),
                         score=float(best[i]), cls=int(cls[i])))
    return dets


def iou(a, b):
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, iou_thres=0.45):
    """점수 높은 순으로 고르고, 같은 클래스끼리 많이 겹치는 박스는 버린다."""
    keep = []
    for d in sorted(dets, key=lambda d: d["score"], reverse=True):
        if all(k["cls"] != d["cls"] or iou(k, d) < iou_thres for k in keep):
            keep.append(d)
    return keep


def pick_main(dets, input_size, min_side=0.4):
    """긴 변이 스캔 틀(=입력 전체) 한 변의 min_side 이상인 박스 중 면적이 가장 큰 것. 없으면 None."""
    best, best_area = None, 0.0
    for d in dets:
        area = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) / (input_size * input_size)
        long_side = max(d["x2"] - d["x1"], d["y2"] - d["y1"]) / input_size
        if long_side >= min_side and area > best_area:
            best, best_area = d, area
    return best


def scan_crop(w, h, ratio=0.7):
    """프레임 가운데 정사각형 스캔 틀 (x, y, 한 변). 짧은 변의 ratio배."""
    s = round(ratio * min(w, h))
    return (w - s) // 2, (h - s) // 2, s
