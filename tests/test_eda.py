from types import SimpleNamespace as NS

import pytest

from src.eda import box_table, cooccurrence, class_summary


def _det(label, box, **attrs):
    return NS(label=label, bounding_box=box, **attrs)


def _sample(path, w, h, dets):
    return NS(id=path, filepath=path, metadata=NS(width=w, height=h), ground_truth=NS(detections=dets))


def _views():
    train = [
        _sample("a.jpg", 200, 100, [
            _det("book", [0.0, 0.0, 1.0, 1.0], IsGroupOf=True),            # 묶음 → drop
            _det("shoe", [0.1, 0.2, 0.05, 0.1], IsGroupOf=False),          # 면적 0.005 → drop
            _det("sock", [0.25, 0.25, 0.5, 0.5], IsGroupOf=False, IsDepiction=None, IsOccluded=1),
        ]),
        _sample("b.jpg", 100, 100, [
            _det("shoe", [0.0, 0.0, 0.5, 0.5]),                            # 속성 없음 → False
            _det("shoe", [0.5, 0.5, 0.5, 0.5], IsDepiction=True),          # 그림 → drop
            _det("Laptop", [0.0, 0.0, 1.0, 1.0]),                          # 우리 클래스 아님 → 무시
        ]),
    ]
    val = [_sample("c.jpg", 50, 100, [_det("door", [0.4, 0.1, 0.2, 0.8])])]
    return {"train": train, "val": val}


def test_box_table_columns_and_rows():
    df = box_table(_views())
    assert len(df) == 6
    for col in ["split", "sample_id", "filepath", "img_w", "img_h", "cls", "x", "y", "w", "h",
                "area", "aspect", "cx", "cy", "IsGroupOf", "IsDepiction", "IsOccluded",
                "IsTruncated", "IsInside", "drop"]:
        assert col in df.columns
    assert list(df["split"]) == ["train"] * 5 + ["val"]


def test_box_table_geometry():
    df = box_table(_views())
    sock = df[df.cls == "sock"].iloc[0]
    assert sock.area == pytest.approx(0.25)
    assert sock.cx == pytest.approx(0.5) and sock.cy == pytest.approx(0.5)
    assert sock.aspect == pytest.approx(2.0)          # 0.5*200 / (0.5*100)
    door = df[df.cls == "door"].iloc[0]
    assert door.aspect == pytest.approx((0.2 * 50) / (0.8 * 100))


def test_box_table_flags_and_drop():
    df = box_table(_views())
    for col in ["IsGroupOf", "IsDepiction", "IsOccluded", "IsTruncated", "IsInside", "drop"]:
        assert df[col].dtype == bool
    assert list(df["drop"]) == [True, True, False, False, True, False]
    sock = df[df.cls == "sock"].iloc[0]
    assert bool(sock.IsOccluded) and not sock.IsDepiction       # 1 → True, None → False


def test_box_table_min_area():
    df = box_table(_views(), min_area=0.3)
    assert df[df.cls == "sock"].iloc[0]["drop"]                 # 0.25 < 0.3


def test_class_summary_counts():
    df = box_table(_views())
    s = class_summary(df).set_index(["split", "cls"])
    assert s.loc[("train", "shoe"), "images"] == 2
    assert s.loc[("train", "shoe"), "boxes"] == 3
    assert s.loc[("train", "shoe"), "boxes_per_image"] == pytest.approx(1.5)
    assert s.loc[("train", "mouse"), "images"] == 0             # 10종 모두 나옴
    assert len(s) == 20


def test_cooccurrence_counts_images_with_both():
    df = box_table(_views())
    m = cooccurrence(df[df.split == "train"])
    assert m.loc["sock", "shoe"] == 1 and m.loc["shoe", "sock"] == 1
    assert m.loc["shoe", "shoe"] == 2                           # 대각선 = 그 클래스가 있는 사진 수
    assert m.loc["door", "shoe"] == 0
    assert list(m.index) == list(m.columns) and len(m) == 10


def test_box_table_marks_unknown_attrs():
    # Open Images CSV의 -1(모름)은 FiftyOne에서 다섯 속성이 모두 True로 들어온다 → attr_unknown
    all_true = dict(IsGroupOf=True, IsDepiction=True, IsOccluded=True, IsTruncated=True, IsInside=True)
    views = {"train": [_sample("d.jpg", 100, 100, [
        _det("toothbrush", [0.2, 0.2, 0.5, 0.5], **all_true),
        _det("toothbrush", [0.2, 0.2, 0.5, 0.5], IsGroupOf=True, IsDepiction=True),
    ])]}
    df = box_table(views)
    assert df["attr_unknown"].dtype == bool
    assert list(df["attr_unknown"]) == [True, False]
    assert list(df["drop"]) == [False, True]         # 모름 → 면적 규칙만 (25%라 남김), 진짜 묶음 → 지움
