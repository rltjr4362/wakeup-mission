# src/evaluate.py
"""평가 도우미: 클래스별 지표 표, 직접 찍은 사진으로 미션 통과율."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .postprocess import decode, nms, pick_main, scan_crop

INPUT, CONF, IOU, MIN_AREA, SCAN_RATIO = 320, 0.5, 0.45, 0.15, 0.7   # 웹앱 app.js와 같은 값
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}                            # 소문자로 비교 (대소문자 무관)


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
    """웹앱과 비슷한 전처리: 가운데 스캔 틀 자르기 → 320 → RGB → 0~1 → [1,3,320,320].

    자르기는 웹앱의 **근사**다. 웹앱은 화면에 보이는 영역(object-fit: cover)의 짧은 변 × 0.7을 쓰므로
    폰에서는 틀이 더 좁을 수 있다. 여기서는 사진 전체의 짧은 변 × 0.7을 쓴다.
    큰 사진을 크게 줄이므로 INTER_AREA로 줄인다 (LINEAR는 계단 현상이 생김)."""
    import cv2
    h, w = img_bgr.shape[:2]
    x, y, s = scan_crop(w, h, SCAN_RATIO)
    crop = cv2.resize(img_bgr[y:y + s, x:x + s], (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None].copy(), crop


def run_onnx(session, tensor):
    out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return nms(decode(out, CONF), IOU)


def my_photos_table(session, photo_dir: Path) -> pd.DataFrame:
    """my_photos/<클래스키>/*.jpg|jpeg|png (대소문자 무관) 를 미션으로 보고 통과 여부를 표로.

    스캔 틀은 웹앱의 근사(사진 짧은 변 × 0.7, 위 preprocess 참고)이므로
    물건을 **가운데에, 꽤 크게** 찍어야 웹앱 결과와 비슷하다.
    읽을 수 없는 파일(HEIC 등)은 건너뛰고 경고만 출력한다. 사진이 없으면 빈 표."""
    import cv2
    rows = []
    for c in C.CLASSES:
        for p in sorted((photo_dir / c.key).glob("*")):
            if p.suffix.lower() not in PHOTO_SUFFIXES:
                continue
            img = cv2.imread(str(p))
            if img is None:
                print(f"⚠️ 읽을 수 없어 건너뜀: {p.name} (HEIC는 JPG/PNG로 바꿔 넣으세요)")
                continue
            tensor, _ = preprocess(img)
            dets = run_onnx(session, tensor)
            main = pick_main(dets, INPUT, MIN_AREA)
            rows.append(dict(미션=c.ko, 파일=p.name, 통과=mission_passed(dets, c.id),
                             잡힌것=C.CLASSES[main["cls"]].ko if main else "-",
                             점수=round(main["score"], 2) if main else None))
    return pd.DataFrame(rows, columns=["미션", "파일", "통과", "잡힌것", "점수"])
