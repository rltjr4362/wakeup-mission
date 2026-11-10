# tests/test_postprocess.py
import numpy as np
from src.postprocess import decode, nms, pick_main, scan_crop, iou

def make_output(boxes, nc=3, n=5):
    """boxes: [(i, cx, cy, w, h, cls, score)] → [1, 4+nc, n] 배열"""
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, cx, cy, w, h, cls, score in boxes:
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = score
    return out

def test_decode_keeps_confident_and_converts_to_corners():
    out = make_output([(1, 100, 100, 40, 20, 2, 0.9), (3, 50, 50, 10, 10, 0, 0.3)])
    dets = decode(out, conf=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert (d["x1"], d["y1"], d["x2"], d["y2"]) == (80, 90, 120, 110)
    assert d["cls"] == 2 and abs(d["score"] - 0.9) < 1e-6

def test_iou():
    a = dict(x1=0, y1=0, x2=10, y2=10)
    b = dict(x1=5, y1=0, x2=15, y2=10)
    assert abs(iou(a, b) - 50 / 150) < 1e-9

def test_nms_removes_overlap_same_class_only():
    a = dict(x1=0, y1=0, x2=10, y2=10, score=0.9, cls=0)
    b = dict(x1=1, y1=0, x2=11, y2=10, score=0.8, cls=0)   # a와 많이 겹침 → 제거
    c = dict(x1=1, y1=0, x2=11, y2=10, score=0.7, cls=1)   # 다른 클래스 → 유지
    kept = nms([b, a, c], iou_thres=0.45)
    assert kept == [a, c]

def test_pick_main_largest_above_min_side():
    small = dict(x1=0, y1=0, x2=50, y2=50, score=0.9, cls=0)      # 긴 변 16%
    big = dict(x1=0, y1=0, x2=200, y2=200, score=0.6, cls=1)      # 62%
    assert pick_main([small, big], input_size=320, min_side=0.4) == big
    assert pick_main([small], input_size=320, min_side=0.4) is None

def test_pick_main_thin_long_box_qualifies():
    thin = dict(x1=0, y1=0, x2=200, y2=20, score=0.8, cls=0)      # 긴 변 62%, 면적 4%
    square = dict(x1=0, y1=0, x2=100, y2=100, score=0.9, cls=1)   # 긴 변 31%
    assert pick_main([thin], input_size=320, min_side=0.4) == thin
    assert pick_main([square], input_size=320, min_side=0.4) is None
    tall = dict(x1=0, y1=0, x2=20, y2=200, score=0.8, cls=0)      # 세로로 긴 것도 통과
    assert pick_main([tall], input_size=320, min_side=0.4) == tall

def test_pick_main_larger_area_wins_among_qualifiers():
    thin = dict(x1=0, y1=0, x2=200, y2=20, score=0.9, cls=0)
    big = dict(x1=0, y1=0, x2=150, y2=150, score=0.5, cls=1)      # 긴 변 47%, 면적 더 큼
    assert pick_main([thin, big], input_size=320, min_side=0.4) == big
    assert pick_main([big, thin], input_size=320, min_side=0.4) == big

def test_scan_crop_centered_square():
    assert scan_crop(1280, 720, ratio=0.7) == (388, 108, 504)


from src.evaluate import mission_passed

def test_mission_passed_needs_target_class_as_main_box():
    big_sock = dict(x1=0, y1=0, x2=200, y2=200, score=0.8, cls=7)
    big_shoe = dict(x1=0, y1=0, x2=250, y2=250, score=0.9, cls=8)
    assert mission_passed([big_sock], target=7, input_size=320, min_side=0.4)
    assert not mission_passed([big_sock, big_shoe], target=7, input_size=320, min_side=0.4)  # 더 큰 게 신발
    assert not mission_passed([], target=7, input_size=320, min_side=0.4)


def test_decode_clips_to_input_frame():
    out = make_output([(0, 10, 100, 40, 20, 0, 0.9), (1, 315, 100, 20, 20, 0, 0.9)])
    a, b = decode(out, conf=0.5)
    assert a["x1"] == 0 and b["x2"] == 320
    assert decode(out, conf=0.5, input_size=300)[1]["x2"] == 300
