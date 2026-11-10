import numpy as np
import pytest

from src.missioneval import GRID, apply_thr, main_cls, mains, false_pass, choose_thresholds, prf


def d(cls, score, side=200):
    return dict(x1=0, y1=0, x2=side, y2=side, score=score, cls=cls)


def test_grid():
    assert GRID[0] == 0.4 and GRID[-1] == 0.9 and len(GRID) == 11


def test_apply_thr_flat_and_per_class():
    dets = [d(0, 0.6), d(1, 0.45)]
    assert apply_thr(dets, 0.5) == [dets[0]]
    assert apply_thr(dets, {0: 0.7, 1: 0.4}) == [dets[1]]


def test_main_cls_filters_before_picking_main():
    big_mouse, small_cup = d(0, 0.55, 300), d(10, 0.9, 200)
    assert main_cls([big_mouse, small_cup], 0.5) == 0          # 큰 박스가 주인공
    assert main_cls([big_mouse, small_cup], {0: 0.8, 10: 0.5}) == 10
    assert main_cls([d(0, 0.9, 50)], 0.5) == -1                # 너무 작음 (긴 변 < 40%)
    assert main_cls([], 0.5) == -1


def test_false_pass_and_choose_thresholds():
    # 네거티브 4장: 마우스 0.6, 마우스 0.8, 컵 0.45, 아무것도 없음
    neg = [[d(0, 0.6)], [d(0, 0.8)], [d(1, 0.45)], []]
    assert list(false_pass(mains(neg, 0.4), 2)) == [0.5, 0.25]
    thr = choose_thresholds(neg, 3, max_fp=0.0)
    assert thr == {0: 0.85, 1: 0.5, 2: 0.4}
    thr = choose_thresholds([[d(0, 0.95)]], 1, max_fp=0.0)
    assert thr == {0: 0.9}                                     # 없으면 최대값
    assert list(false_pass([], 2)) == [0, 0]


def test_prf():
    targets = [0, 0, 1, -1, -1]
    preds   = [0, 1, 1, 0, -1]
    t = prf(targets, preds, 3)
    assert t.loc[0, ["TP", "FP", "FN"]].tolist() == [1, 1, 1]
    assert t.loc[0, "P"] == 0.5 and t.loc[0, "R"] == 0.5 and t.loc[0, "F1"] == pytest.approx(0.5)
    assert t.loc[1, "P"] == 0.5 and t.loc[1, "R"] == 1.0
    assert t.loc[2, "F1"] == 0.0
