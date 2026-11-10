# tests/test_config.py
from src import config as C

def test_ten_classes_in_order():
    assert [c.key for c in C.CLASSES] == [
        "mouse", "umbrella", "book", "toothbrush",
        "hat", "door", "toilet_paper", "sock", "shoe", "glasses"]
    assert [c.id for c in C.CLASSES] == list(range(10))

def test_every_class_has_open_images_label():
    for c in C.CLASSES:
        assert c.oi_labels, c.key
    # 합치기 표: Open Images 라벨 → 우리 클래스 키
    assert C.OI_TO_KEY["Fedora"] == "hat"
    assert C.OI_TO_KEY["Sunglasses"] == "glasses"
    assert C.OI_TO_KEY["High heels"] == "shoe"
    assert set(C.OI_TO_KEY.values()) == {c.key for c in C.CLASSES}

def test_old_classes_have_coco_ids():
    assert {c.key: c.coco_id for c in C.CLASSES if c.coco_id is not None} == {
        "mouse": 64, "umbrella": 25, "book": 73, "toothbrush": 79}
    assert [c.key for c in C.OLD] == ["mouse", "umbrella", "book", "toothbrush"]

def test_mission_text_uses_right_particle():
    assert C.mission_text(C.BY_KEY["book"]) == "📖 책을 비춰라"
    assert C.mission_text(C.BY_KEY["mouse"]) == "🖱️ 마우스를 비춰라"

def test_js_export_contains_all_classes():
    js = C.to_js()
    assert js.startswith("export const CLASSES = [")
    assert js.count('"key"') == 10
