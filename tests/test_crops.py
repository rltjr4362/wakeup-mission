import cv2
import numpy as np
import pytest

from src.crops import (square_at, visible_fraction, rows_in_crop, norm_to_px, half_of, plan_crops,
                       merge_records, export_full, export_crops, link_split)
from src.clean import crop_square


def test_square_at_matches_crop_square_without_shift():
    for box, fill in [((450, 375, 550, 425), 0.5), ((0, 0, 100, 100), 0.5), ((900, 700, 1000, 800), 0.5)]:
        assert square_at(1000, 800, box, fill) == crop_square(1000, 800, box, fill)


def test_square_at_shift_moves_center_and_stays_inside():
    assert square_at(1000, 800, (450, 375, 550, 425), 0.5, shift=(0.1, -0.1)) == (420, 280, 200)
    x, y, s = square_at(1000, 800, (0, 0, 100, 100), 0.5, shift=(-0.3, -0.3))
    assert (x, y, s) == (0, 0, 200)


def test_square_at_caps_to_image():
    assert square_at(300, 200, (0, 0, 300, 200), 0.5) == (50, 0, 200)


def test_visible_fraction():
    assert visible_fraction((0, 0, 10, 10), (0, 0, 100)) == 1.0
    assert visible_fraction((90, 0, 110, 10), (0, 0, 100)) == 0.5
    assert visible_fraction((200, 200, 210, 210), (0, 0, 100)) == 0.0


def test_rows_in_crop_clips_and_drops():
    crop = (100, 100, 200)                                 # 100~300
    targets = [(3, 150, 150, 250, 250),                    # 다 보임
               (5, 250, 100, 350, 200),                    # 반 보임 → 잘라서 남김
               (7, 280, 100, 380, 200)]                    # 20% → 버림
    rows = rows_in_crop(targets, crop)
    assert [r[0] for r in rows] == [3, 5]
    assert rows[0] == pytest.approx((3, 0.5, 0.5, 0.5, 0.5))
    assert rows[1] == pytest.approx((5, 0.875, 0.25, 0.25, 0.5))


def test_norm_to_px():
    assert norm_to_px([0.1, 0.2, 0.5, 0.25], 200, 100) == pytest.approx((20, 20, 120, 45))


def test_half_of_is_deterministic_and_balanced():
    names = [f"img{i}" for i in range(2000)]
    halves = [half_of(n) for n in names]
    assert halves == [half_of(n) for n in names]
    assert set(halves) == {"tune", "test"}
    assert 0.45 < halves.count("tune") / len(halves) < 0.55


def test_plan_crops_random_fill_in_range_and_n_per():
    rng = np.random.default_rng(0)
    box = (400, 300, 500, 400)                             # 한 변 100
    plans = plan_crops(1000, 800, [(1, *box)], [(("pos", 1), box)], rng, n_per=lambda n: 3)
    assert len(plans) == 3
    for p in plans:
        x, y, s = p["crop"]
        assert 100 / 0.8 - 1 <= s <= 100 / 0.35 + 1
        assert p["rows"] and p["rows"][0][0] == 1
    assert len({p["crop"] for p in plans}) == 3            # 서로 다른 crop


def test_plan_crops_fixed_fill_is_centered():
    box = (400, 300, 500, 400)
    [p] = plan_crops(1000, 800, [], [(("neg", "Pen"), box)], None, fixed_fill=0.5)
    assert p["crop"] == (350, 250, 200) and p["rows"] == []


def test_merge_records_dedupes_by_path():
    a = [dict(filepath="a", targets=[1], negs=[])]
    b = [dict(filepath="a", targets=[], negs=[2]), dict(filepath="b", targets=[], negs=[])]
    assert [r["filepath"] for r in merge_records(a, b)] == ["a", "b"]


def _img(tmp_path, name, w=400, h=300):
    p = tmp_path / "src" / f"{name}.jpg"
    p.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(p), np.full((h, w, 3), 128, np.uint8))
    return str(p)


def test_export_full_resizes_and_skips_negative_boxes(tmp_path):
    fp = _img(tmp_path, "a", 1280, 640)
    recs = [dict(filepath=fp, targets=[(2, 0.1, 0.1, 0.2, 0.2)], negs=[("Pen", 0.5, 0.5, 0.2, 0.2)])]
    out = tmp_path / "full"
    assert export_full(recs, out, "train", max_side=640) == 1
    assert cv2.imread(str(out / "images/train/a.jpg")).shape[:2] == (320, 640)
    assert (out / "labels/train/a.txt").read_text() == "2 0.200000 0.200000 0.200000 0.200000\n"


def test_export_crops_pos_and_neg(tmp_path):
    fp = _img(tmp_path, "a")
    recs = [dict(filepath=fp, targets=[(5, 0.1, 0.1, 0.2, 0.2), (1, 0.0, 0.0, 0.05, 0.05)],    # 두 번째는 1% 미만
                 negs=[("Pen", 0.6, 0.6, 0.3, 0.3)])]
    out = tmp_path / "crops"
    df = export_crops(recs, out, "train", seed=0, n_per_class={5: 3})
    assert sorted(df["kind"].tolist()) == ["neg", "pos", "pos", "pos"]
    assert set(df[df.kind == "pos"]["target"]) == {5}
    assert (df[df.kind == "neg"]["target"] == -1).all() and (df[df.kind == "neg"]["center"] == "Pen").all()
    for f in df["file"]:
        assert cv2.imread(str(out / "images/train" / f)).shape[:2] == (320, 320)
    # 다시 부르면 저장한 표를 그대로
    assert export_crops(recs, out, "train").equals(df)


def test_link_split_merges(tmp_path):
    for d, name in (("a", "x"), ("b", "y")):
        (tmp_path / d / "images/train").mkdir(parents=True)
        (tmp_path / d / "labels/train").mkdir(parents=True)
        (tmp_path / d / "images/train" / f"{name}.jpg").write_bytes(b"1")
        (tmp_path / d / "labels/train" / f"{name}.txt").write_text("")
    assert link_split([tmp_path / "a", tmp_path / "b"], tmp_path / "m", "train") == 2
    assert sorted(p.name for p in (tmp_path / "m/images/train").iterdir()) == ["x.jpg", "y.jpg"]
