# tests/test_clean.py
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pandas as pd

from src import config as C
from src.clean import (GRAY, should_drop, paint_mask, crop_square, clean_and_export,
                       reason_table, mission_crops, mission_pass_rate)


# ---------- should_drop ----------

def test_should_drop_rules():
    assert should_drop(False, False, 0.2) is False
    assert should_drop(True, False, 0.2) is True          # 묶음 박스
    assert should_drop(False, True, 0.2) is True          # 그림·사진 속 물건
    assert should_drop(False, False, 0.005) is True       # 너무 작음 (< 1%)
    assert should_drop(False, False, 0.01) is False       # 경계값은 남김


def test_should_drop_treats_none_and_ints():
    assert should_drop(None, None, 0.2) is False
    assert should_drop(1, 0, 0.2) is True
    assert should_drop(0, 0, 0.2) is False
    assert should_drop(False, False, 0.05, min_area=0.1) is True


# ---------- paint_mask ----------

def test_paint_mask_drop_only():
    m = paint_mask(10, 20, [[0.0, 0.0, 0.5, 0.5]], [])
    assert m.shape == (10, 20) and m.dtype == bool
    assert m[:5, :10].all() and m.sum() == 50


def test_paint_mask_keeps_kept_box_pixels():
    # 사진 전체가 지울 박스, 가운데는 남길 박스 → 가운데는 원래 픽셀
    m = paint_mask(10, 10, [[0, 0, 1, 1]], [[0.2, 0.2, 0.3, 0.3]])
    assert not m[2:5, 2:5].any()
    assert m.sum() == 100 - 9


def test_paint_mask_rounds_outward_and_clips():
    # x 0.15*10=1.5 → floor 1, x2 0.45*10=4.5 → ceil 5 ; 범위 밖은 잘림
    m = paint_mask(10, 10, [[0.15, 0.9, 0.3, 0.5]], [])
    assert m[9, 1:5].all() and m.sum() == 4


# ---------- crop_square ----------

def test_crop_square_centered():
    # 박스 100×50, fill 0.5 → 한 변 200, 박스 중심 (500, 400) 기준
    assert crop_square(1000, 800, (450, 375, 550, 425), 0.5) == (400, 300, 200)


def test_crop_square_shifts_inside():
    assert crop_square(1000, 800, (0, 0, 100, 100), 0.5) == (0, 0, 200)
    assert crop_square(1000, 800, (900, 700, 1000, 800), 0.5) == (800, 600, 200)


def test_crop_square_too_big_uses_largest_fitting():
    # 한 변 1000이 필요하지만 짧은 변이 800 → 800, 박스 중심에 맞추되 안으로
    assert crop_square(1000, 800, (300, 300, 900, 500), 0.6) == (200, 0, 800)


# ---------- clean_and_export (가짜 FiftyOne view) ----------

def _det(label, box, group=False, depiction=False):
    return NS(label=label, bounding_box=box, IsGroupOf=group, IsDepiction=depiction)


def _fake_view(tmp_path):
    img = np.full((100, 200, 3), 255, np.uint8)
    p1, p2 = tmp_path / "src" / "a.jpg", tmp_path / "src" / "b.jpg"
    p1.parent.mkdir()
    cv2.imwrite(str(p1), img); cv2.imwrite(str(p2), img)
    s1 = NS(filepath=str(p1), ground_truth=NS(detections=[
        _det("book", [0.0, 0.0, 1.0, 1.0], group=True),       # 책장 전체 → 지움
        _det("book", [0.25, 0.25, 0.5, 0.5]),                  # 큰 책 → 남김
        _det("book", [0.255, 0.25, 0.5, 0.5]),                 # 같은 책 중복 → dedupe
    ]))
    s2 = NS(filepath=str(p2), ground_truth=NS(detections=[
        _det("shoe", [0.0, 0.0, 0.05, 0.05]),                  # 작음 → 지움 (사진 전체가 지워짐)
    ]))
    return [s1, s2]


def test_clean_and_export_writes_images_labels_yaml(tmp_path):
    out = tmp_path / "clean"
    stats = clean_and_export(_fake_view(tmp_path), out, "train")
    book = C.BY_KEY["book"].id
    assert stats["images"] == 2 and stats["all_dropped"] == 1
    assert stats["kept"]["book"] == 1 and stats["dropped"]["book"] == 1 and stats["dropped"]["shoe"] == 1
    lab = (out / "labels" / "train" / "a.txt").read_text()
    assert lab == f"{book} 0.500000 0.500000 0.500000 0.500000\n"
    assert (out / "labels" / "train" / "b.txt").read_text() == ""
    a = cv2.imread(str(out / "images" / "train" / "a.jpg"))
    assert abs(int(a[5, 5, 0]) - GRAY[0]) <= 3            # 지운 곳은 회색
    assert a[50, 100].min() >= 250                        # 남긴 박스 안은 원래 흰색
    assert (out / "labels" / "train.done").exists()
    yaml = (out / "dataset.yaml").read_text()
    assert f"path: {out.resolve()}" in yaml and "train: images/train" in yaml and "  9: glasses" in yaml


def test_clean_and_export_skips_when_done_and_redoes_partial(tmp_path):
    out = tmp_path / "clean"
    view = _fake_view(tmp_path)
    first = clean_and_export(view, out, "val")
    again = clean_and_export([], out, "val")              # 끝난 표시가 있으면 저장된 통계를 그대로
    assert again == first
    (out / "labels" / "val.done").unlink()                # 중단된 척 → 다시 처음부터
    (out / "images" / "val" / "junk.jpg").write_bytes(b"x")
    redo = clean_and_export(view, out, "val")
    assert redo == first and not (out / "images" / "val" / "junk.jpg").exists()


def test_reason_table_percentages(tmp_path):
    df = reason_table(_fake_view(tmp_path)).set_index("키")
    assert df.loc["book", "박스"] == 3
    assert df.loc["book", "묶음%"] == round(100 / 3, 1)
    assert df.loc["book", "지움%"] == round(100 / 3, 1)
    assert df.loc["shoe", "작음%"] == 100.0
    assert df.loc["mouse", "박스"] == 0


# ---------- mission_crops / mission_pass_rate ----------

def _yolo_dir(tmp_path):
    d = tmp_path / "yolo"
    (d / "images" / "val").mkdir(parents=True); (d / "labels" / "val").mkdir(parents=True)
    cv2.imwrite(str(d / "images" / "val" / "x.jpg"), np.zeros((400, 600, 3), np.uint8))
    (d / "labels" / "val" / "x.txt").write_text("2 0.5 0.5 0.2 0.3\n8 0.1 0.1 0.05 0.05\n")
    return d


def test_mission_crops_skips_small_boxes(tmp_path):
    out = tmp_path / "crops"
    df = mission_crops(_yolo_dir(tmp_path), "val", out, size=64)
    assert list(df.columns) == ["file", "target", "cls_key"]
    assert df.to_dict("records") == [dict(file="x_0.jpg", target=2, cls_key="book")]
    assert cv2.imread(str(out / "x_0.jpg")).shape == (64, 64, 3)
    assert (out / "done").exists()
    assert mission_crops(tmp_path / "nope", "val", out, size=64).equals(df)   # 끝난 표시 → 다시 안 만듦


class _FakeBoxes:
    def __init__(self, rows):
        a = np.array(rows, np.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]


class FakeModel:
    """파일 이름에 'good'이 있으면 책(2)을 크게, 아니면 아무것도 못 찾는 가짜 YOLO."""
    def predict(self, paths, **kw):
        assert kw["imgsz"] == 320
        return [NS(boxes=_FakeBoxes([[40, 40, 280, 280, 0.9, 2]] if "good" in str(p) else [])) for p in paths]


def test_mission_pass_rate(tmp_path):
    crops = pd.DataFrame([dict(file="good_0.jpg", target=2, cls_key="book"),
                          dict(file="bad_0.jpg", target=2, cls_key="book"),
                          dict(file="good_1.jpg", target=8, cls_key="shoe")])
    df = mission_pass_rate(FakeModel(), crops, tmp_path, batch=2)
    assert df["passed"].tolist() == [True, False, False]
    assert df["pred_key"].tolist() == ["book", "-", "book"]
