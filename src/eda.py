"""EDA: FiftyOne view의 박스를 한 줄 = 박스 하나인 pandas 표로 펼친다. 이후 그림은 모두 이 표를 쓴다."""
import pandas as pd

from . import config as C
from .clean import _flag, should_drop

ATTRS = ["IsGroupOf", "IsDepiction", "IsOccluded", "IsTruncated", "IsInside"]


def _box_row(split, sample, d, min_area):
    """박스 하나 → dict. 좌표는 정규화(0~1), aspect는 픽셀 기준 가로/세로."""
    x, y, w, h = d.bounding_box
    img_w, img_h = sample.metadata.width, sample.metadata.height
    row = dict(split=split, sample_id=sample.id, filepath=sample.filepath, img_w=img_w, img_h=img_h,
               cls=d.label, x=x, y=y, w=w, h=h, area=w * h,
               aspect=(w * img_w) / (h * img_h) if h > 0 else float("nan"),
               cx=x + w / 2, cy=y + h / 2)
    for a in ATTRS:
        row[a] = _flag(getattr(d, a, None))                 # None · 없음 → False
    # Open Images CSV의 -1(= 모름)을 FiftyOne은 True로 읽는다 → 다섯 속성이 모두 True면 사실은 "모름"
    row["attr_unknown"] = all(row[a] for a in ATTRS)
    row["drop"] = should_drop(row["IsGroupOf"], row["IsDepiction"], row["area"], min_area, unknown=row["attr_unknown"])
    return row


def box_table(views: dict, min_area=0.01) -> pd.DataFrame:
    """{split: view} → 박스 표. 우리 10종 라벨만 (겹친 박스 제거 전 원래 박스).
    sample.metadata(width, height)가 있어야 한다 → 노트북에서 view.compute_metadata()."""
    rows = []
    for split, view in views.items():
        for sample in view:
            gt = sample.ground_truth
            for d in (gt.detections if gt is not None else []):
                if d.label in C.BY_KEY:
                    rows.append(_box_row(split, sample, d, min_area))
    df = pd.DataFrame(rows)
    for col in ATTRS + ["attr_unknown", "drop"]:
        df[col] = df[col].astype(bool)
    return df


def class_summary(df: pd.DataFrame) -> pd.DataFrame:
    """split · 클래스별 사진 수(그 클래스 박스가 하나라도 있는 사진), 박스 수, 사진당 박스 수.
    10종이 모두 나오도록 없는 클래스는 0."""
    rows = []
    for split in df["split"].unique():
        part = df[df["split"] == split]
        for key in C.KEYS:
            boxes = part[part["cls"] == key]
            n_img = boxes["sample_id"].nunique()
            rows.append(dict(split=split, cls=key, images=n_img, boxes=len(boxes),
                             boxes_per_image=len(boxes) / n_img if n_img else 0.0))
    return pd.DataFrame(rows)


def cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """클래스 × 클래스: 두 클래스가 함께 있는 사진 수. 대각선 = 그 클래스가 있는 사진 수."""
    m = pd.DataFrame(0, index=C.KEYS, columns=C.KEYS)
    for _, classes in df.groupby("sample_id")["cls"]:
        present = sorted(set(classes))
        for a in present:
            for b in present:
                m.loc[a, b] += 1
    return m
