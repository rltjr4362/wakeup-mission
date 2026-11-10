# src/evaluate.py
"""평가 도우미: 클래스별 지표 표, 직접 찍은 사진으로 미션 통과율."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .postprocess import decode, nms, pick_main, scan_crop

INPUT, CONF, IOU, MIN_AREA, SCAN_RATIO = 320, 0.5, 0.45, 0.15, 0.7   # 웹앱 app.js와 같은 값


def per_class_table(metrics, names) -> pd.DataFrame:
    """ultralytics val() 결과 → 클래스별 P, R, mAP50, mAP50-95 표."""
    b = metrics.box
    rows = []
    for j, ci in enumerate(b.ap_class_index):
        rows.append(dict(id=int(ci), 클래스=names[int(ci)], P=b.p[j], R=b.r[j], mAP50=b.ap50[j], **{"mAP50-95": b.ap[j]}))
    return pd.DataFrame(rows).round(3)


def mission_passed(dets, target, input_size=INPUT, min_area=MIN_AREA) -> bool:
    """웹앱 판정과 같은 규칙: 틀 안의 '주인공 박스'가 목표 클래스여야 통과."""
    main = pick_main(dets, input_size, min_area)
    return main is not None and main["cls"] == target


def preprocess(img_bgr):
    """웹앱과 같은 전처리: 가운데 스캔 틀 자르기 → 320 → RGB → 0~1 → [1,3,320,320]."""
    import cv2
    h, w = img_bgr.shape[:2]
    x, y, s = scan_crop(w, h, SCAN_RATIO)
    crop = cv2.resize(img_bgr[y:y + s, x:x + s], (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None].copy(), crop


def run_onnx(session, tensor):
    out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return nms(decode(out, CONF), IOU)


def my_photos_table(session, photo_dir: Path) -> pd.DataFrame:
    """my_photos/<클래스키>/*.jpg 를 미션으로 보고 통과 여부를 표로."""
    import cv2
    rows = []
    for c in C.CLASSES:
        for p in sorted((photo_dir / c.key).glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            tensor, _ = preprocess(cv2.imread(str(p)))
            dets = run_onnx(session, tensor)
            main = pick_main(dets, INPUT, MIN_AREA)
            rows.append(dict(미션=c.ko, 파일=p.name, 통과=mission_passed(dets, c.id),
                             잡힌것=C.CLASSES[main["cls"]].ko if main else "-",
                             점수=round(main["score"], 2) if main else None))
    return pd.DataFrame(rows)
